"""Storage layer tests — real SQLite operations, no mocks.

Covers: fresh initialization, migrations, analysis run lifecycle, project
isolation, experiment CRUD/failure handling, corruption/missing-data errors,
secret redaction and cross-restart persistence.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from storage.database import (
    configure_database,
    get_database,
    reset_database,
    resolve_database_url,
)
from storage.migrations import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def storage_db(tmp_path):
    """Point the process-global database at a fresh file-based SQLite file."""
    reset_database()
    db_path = tmp_path / "test_storage.db"
    configure_database(f"sqlite+aiosqlite:///{db_path}")
    yield db_path
    reset_database()


@pytest.fixture
def client(storage_db):
    """TestClient over the real router, backed by the fixture database."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def _user_version(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _table_names(db_path: Path) -> set:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _run(coro):
    """Run async store helpers from sync tests on their own event loop."""
    return asyncio.run(coro)


async def _init_schema():
    await get_database().init_schema()


# ---------------------------------------------------------------------------
# Fresh database / initialization
# ---------------------------------------------------------------------------

def test_fresh_database_initialization(storage_db):
    assert not storage_db.exists()
    _run(_init_schema())
    assert storage_db.exists()
    assert _user_version(storage_db) == SCHEMA_VERSION
    tables = _table_names(storage_db)
    for expected in ("projects", "analyses", "experiments", "dataset_reports", "recommendations"):
        assert expected in tables


def test_initialization_is_idempotent(storage_db):
    _run(_init_schema())
    _run(_init_schema())  # second run must be a no-op
    assert _user_version(storage_db) == SCHEMA_VERSION


def test_missing_database_recreated_on_next_init(tmp_path):
    db_path = tmp_path / "deleted.db"
    configure_database(f"sqlite+aiosqlite:///{db_path}")
    _run(_init_schema())
    db_path.unlink()
    reset_database()
    configure_database(f"sqlite+aiosqlite:///{db_path}")
    _run(_init_schema())
    assert db_path.exists()
    assert _user_version(db_path) == SCHEMA_VERSION
    reset_database()


def test_database_directory_created(tmp_path):
    nested = tmp_path / "deep" / "dir" / "agent.db"
    configure_database(f"sqlite+aiosqlite:///{nested}")
    _run(_init_schema())
    assert nested.exists()
    reset_database()


def test_resolve_database_url_anchors_relative_paths():
    from storage.database import BACKEND_ROOT

    resolved = resolve_database_url("sqlite+aiosqlite:///./llm_training_agent.db")
    assert resolved.startswith("sqlite+aiosqlite:///")
    assert "llm_training_agent.db" in resolved
    # Relative path must not depend on CWD: it is anchored to the backend root.
    assert BACKEND_ROOT.name in resolved
    # In-memory URLs pass through unchanged.
    assert resolve_database_url("sqlite+aiosqlite:///:memory:") == "sqlite+aiosqlite:///:memory:"


# ---------------------------------------------------------------------------
# Migrations (legacy schema → current, data-preserving)
# ---------------------------------------------------------------------------

_LEGACY_SCHEMA_SQL = """
CREATE TABLE projects (
    project_id VARCHAR(36) PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    project_path VARCHAR(1024) NOT NULL UNIQUE,
    framework VARCHAR(100),
    framework_version VARCHAR(50),
    base_model VARCHAR(255),
    tokenizer VARCHAR(255),
    created_at DATETIME,
    updated_at DATETIME
);
CREATE TABLE analyses (
    analysis_id VARCHAR(36) PRIMARY KEY,
    project_id VARCHAR(36) NOT NULL REFERENCES projects (project_id),
    started_at DATETIME,
    completed_at DATETIME,
    duration INTEGER,
    health_score FLOAT,
    readiness_score FLOAT,
    status VARCHAR(50)
);
CREATE TABLE experiments (
    experiment_id VARCHAR(36) PRIMARY KEY,
    project_id VARCHAR(36) NOT NULL REFERENCES projects (project_id),
    dataset_version VARCHAR(255) NOT NULL,
    model VARCHAR(255) NOT NULL,
    tokenizer VARCHAR(255),
    hyperparameters JSON,
    metrics JSON,
    notes TEXT,
    created_at DATETIME,
    tags JSON,
    artifacts JSON
);
"""


def test_legacy_schema_is_migrated_and_data_preserved(tmp_path):
    """A database created by the originally shipped schema must be upgraded
    additively (columns added, existing rows untouched)."""
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_LEGACY_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO projects (project_id, project_name, project_path) "
            "VALUES ('p1', 'old-project', ?)",
            (str(db_path.parent / "old-project"),),
        )
        conn.execute(
            "INSERT INTO analyses (analysis_id, project_id, status) "
            "VALUES ('a1', 'p1', 'completed')"
        )
        conn.execute(
            "INSERT INTO experiments (experiment_id, project_id, dataset_version, model) "
            "VALUES ('e1', 'p1', 'v1', 'llama3.2')"
        )
        conn.commit()
    finally:
        conn.close()

    configure_database(f"sqlite+aiosqlite:///{db_path}")
    _run(_init_schema())
    assert _user_version(db_path) == SCHEMA_VERSION

    conn = sqlite3.connect(db_path)
    try:
        # Existing data untouched.
        assert conn.execute("SELECT project_name FROM projects").fetchone()[0] == "old-project"
        assert conn.execute("SELECT status FROM analyses WHERE analysis_id='a1'").fetchone()[0] == "completed"
        assert conn.execute("SELECT model FROM experiments WHERE experiment_id='e1'").fetchone()[0] == "llama3.2"
        # New v2 columns exist with sane defaults.
        analysis_cols = {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
        assert {"context", "report", "error", "partial_failures", "config_snapshot"} <= analysis_cols
        experiment_cols = {row[1] for row in conn.execute("PRAGMA table_info(experiments)")}
        assert {"status", "estimates", "hardware", "analysis_id", "error"} <= experiment_cols
        # Legacy experiment got the default status instead of NULL.
        assert conn.execute("SELECT status FROM experiments WHERE experiment_id='e1'").fetchone()[0] == "estimated"
        # Previously unused tables are created too.
        assert "chat_sessions" in _table_names(db_path)
    finally:
        conn.close()
    reset_database()


def test_newer_schema_version_is_rejected(tmp_path):
    """A database from a future version must be refused, not modified."""
    from core.errors import StorageError

    db_path = tmp_path / "future.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_LEGACY_SCHEMA_SQL)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
    finally:
        conn.close()

    configure_database(f"sqlite+aiosqlite:///{db_path}")
    with pytest.raises(StorageError) as exc_info:
        _run(_init_schema())
    assert exc_info.value.error_code == "STORAGE_SCHEMA_MISMATCH"
    reset_database()


def test_migrating_twice_is_safe(tmp_path):
    db_path = tmp_path / "twice.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_LEGACY_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()

    configure_database(f"sqlite+aiosqlite:///{db_path}")
    _run(_init_schema())
    reset_database()
    configure_database(f"sqlite+aiosqlite:///{db_path}")
    _run(_init_schema())
    assert _user_version(db_path) == SCHEMA_VERSION
    reset_database()


# ---------------------------------------------------------------------------
# Analysis runs: lifecycle, multiple runs, statuses
# ---------------------------------------------------------------------------

def _sample_run(project_path: str, name: str = "proj"):
    """Create + complete a realistic run; returns its analysis_id."""
    from storage.analysis_store import STATUS_COMPLETED, complete_run, start_run

    async def flow():
        run_id = await start_run(
            project_path,
            project_name=name,
            config_snapshot={"provider": "ollama", "model": "llama3.2"},
        )
        context = {
            "project_name": name,
            "project_path": project_path,
            "project_health_score": 71.5,
            "training_readiness_score": 40.0,
            "dataset_paths": ["data/train.jsonl"],
        }
        report = {"executive_summary": f"Report for {name}", "project_health_score": 71.5}
        results = {
            "dataset": {
                "dataset_name": "train.jsonl",
                "quality_score": 0.8,
                "duplicate_percentage": 1.0,
                "token_count": 1234,
                "findings": ["finding-1"],
                "recommendations": ["rec-1"],
            },
            "prompt": {"template_name": "system.txt", "clarity_score": 0.7, "detected_issues": ["ambiguous"]},
            "hyperparameters": {"learning_rate": 0.0002, "batch_size": 8, "epochs": 3, "optimizer": "adamw"},
            "model": {"selected_model": "llama3.2", "strengths": ["fast"], "weaknesses": ["small"]},
            "prediction": {"confidence": "medium", "likely_failure_modes": ["overfitting"]},
            "cost": {"estimated_training_time": "4h", "estimated_gpu_hours": 4.0},
        }
        recommendations = [
            {
                "recommendation_id": "r-1",
                "title": "Increase epochs",
                "description": "The model is undertrained.",
                "evidence": "loss still decreasing",
                "severity": "high",
                "confidence": "medium",
            }
        ]
        saved = await complete_run(
            run_id,
            context=context,
            report=report,
            results=results,
            recommendations=recommendations,
            partial_failures=[],
            health_score=71.5,
            readiness_score=40.0,
            status=STATUS_COMPLETED,
        )
        assert saved
        return run_id

    return _run(flow())


def test_analysis_run_lifecycle(storage_db):
    from storage.analysis_store import get_run, list_runs

    run_id = _sample_run(r"C:\work\proj-a", "proj-a")
    runs = _run(list_runs(r"C:\work\proj-a"))
    assert len(runs) == 1
    assert runs[0]["analysisId"] == run_id
    assert runs[0]["status"] == "completed"
    assert runs[0]["startedAt"] and runs[0]["completedAt"]
    assert runs[0]["durationSeconds"] is not None

    detail = _run(get_run(run_id))
    assert detail["report"]["executive_summary"] == "Report for proj-a"
    assert detail["context"]["project_health_score"] == 71.5
    assert detail["results"]["dataset"]["dataset_name"] == "train.jsonl"
    assert detail["results"]["prediction"]["confidence"] == "medium"
    assert len(detail["recommendations"]) == 1
    assert detail["config"]["model"] == "llama3.2"
    assert detail["partialFailures"] == []


def test_multiple_runs_never_overwrite(storage_db):
    from storage.analysis_store import (
        STATUS_COMPLETED,
        complete_run,
        get_run,
        list_runs,
        start_run,
    )

    async def flow():
        ids = []
        for i in range(3):
            run_id = await start_run(r"C:\work\multi", project_name="multi")
            await complete_run(
                run_id,
                context={"project_name": "multi", "run": i},
                report={"executive_summary": f"run {i}"},
                results={},
                recommendations=[],
                status=STATUS_COMPLETED,
            )
            ids.append(run_id)
        return ids

    ids = _run(flow())
    assert len(set(ids)) == 3
    runs = _run(list_runs(r"C:\work\multi"))
    assert len(runs) == 3
    # Every run keeps its own report — nothing is overwritten.
    assert _run(get_run(ids[0]))["report"]["executive_summary"] == "run 0"
    assert _run(get_run(ids[-1]))["report"]["executive_summary"] == "run 2"


def test_failed_run_is_recorded_with_error(storage_db):
    from storage.analysis_store import fail_run, get_run, list_runs, start_run

    async def flow():
        run_id = await start_run(r"C:\work\will-fail", project_name="will-fail")
        await fail_run(run_id, RuntimeError("scanner exploded: sk-abcdef123456"))
        return run_id

    run_id = _run(flow())
    runs = _run(list_runs(r"C:\work\will-fail"))
    assert runs[0]["status"] == "failed"
    assert runs[0]["failureCount"] == 1
    detail = _run(get_run(run_id))
    assert detail["error"] is not None
    assert "exploded" in detail["error"]
    # Credential-shaped token was redacted before persistence.
    assert "sk-abcdef123456" not in detail["error"]
    assert "[REDACTED]" in detail["error"]


def test_partial_run_not_marked_successful(storage_db):
    from storage.analysis_store import (
        STATUS_PARTIAL,
        complete_run,
        get_run,
        list_runs,
        run_failure,
        start_run,
    )

    async def flow():
        run_id = await start_run(r"C:\work\partial", project_name="partial")
        saved = await complete_run(
            run_id,
            context={"project_name": "partial"},
            report={"executive_summary": "degraded"},
            results={"dataset": {"dataset_name": "d", "error": "llm down"}},
            recommendations=[],
            partial_failures=[
                run_failure("dataset", "llm down", critical=True),
                run_failure("gpu_time_estimate", "no driver", critical=False),
            ],
            status=STATUS_PARTIAL,
        )
        assert saved
        return run_id

    run_id = _run(flow())
    runs = _run(list_runs(r"C:\work\partial"))
    assert runs[0]["status"] == "partial"
    assert runs[0]["failureCount"] == 2
    detail = _run(get_run(run_id))
    assert {f["step"] for f in detail["partialFailures"]} == {"dataset", "gpu_time_estimate"}


# ---------------------------------------------------------------------------
# Project isolation & identity
# ---------------------------------------------------------------------------

def test_project_isolation_same_display_name(storage_db):
    """Two different folders with the SAME name must stay separate."""
    _sample_run(r"C:\clients\A\my-project", "my-project")
    _sample_run(r"C:\clients\B\my-project", "my-project")

    from storage.analysis_store import list_runs

    runs_a = _run(list_runs(r"C:\clients\A\my-project"))
    runs_b = _run(list_runs(r"C:\clients\B\my-project"))
    assert len(runs_a) == 1
    assert len(runs_b) == 1
    assert runs_a[0]["analysisId"] != runs_b[0]["analysisId"]
    assert runs_a[0]["projectName"] == "my-project"
    assert runs_b[0]["projectName"] == "my-project"
    # The stored identities differ even though display names match.
    assert runs_a[0]["projectPath"] != runs_b[0]["projectPath"]
    # Unknown project gets nothing (never another project's data).
    assert _run(list_runs(r"C:\clients\Z\my-project")) == []


def test_project_identity_is_case_insensitive_on_windows_paths(storage_db):
    """C:\\Proj and c:\\proj must resolve to the SAME project."""
    _sample_run(r"C:\Work\MyProj", "myproj")
    from storage.analysis_store import list_runs

    runs = _run(list_runs(r"c:\work\myproj"))
    assert len(runs) == 1


def test_duplicate_project_rows_not_created(storage_db):
    _sample_run(r"C:\work\dup", "dup")
    _sample_run(r"C:\work\dup", "dup")

    conn = sqlite3.connect(_current_db())
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM projects WHERE project_path LIKE '%work%dup%'"
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def _current_db() -> Path:
    from storage.database import get_database
    from storage.database import _sqlite_file_path

    path = _sqlite_file_path(get_database().database_url)
    assert path is not None
    return path


def test_secret_values_redacted_before_persistence(storage_db):
    """Analyzer findings that quote a project file's credentials must be
    scrubbed before they touch the database."""
    from storage.analysis_store import (
        STATUS_COMPLETED,
        complete_run,
        get_run,
        start_run,
    )

    async def flow():
        run_id = await start_run(r"C:\work\leaky", project_name="leaky")
        await complete_run(
            run_id,
            context={"project_name": "leaky", "note": "OPENAI_API_KEY=sk-1234567890abcdef"},
            report={"executive_summary": "found OPENAI_API_KEY=sk-1234567890abcdef"},
            results={
                "dataset": {
                    "dataset_name": "d",
                    "findings": ["config.yaml contains api_key = sk-1234567890abcdef"],
                }
            },
            recommendations=[
                {"recommendation_id": "r", "title": "Rotate key", "description": "token ghp_abcdefghijklmnopqrstuvwx leaked"}
            ],
            status=STATUS_COMPLETED,
        )
        return run_id

    run_id = _run(flow())
    detail = _run(get_run(run_id))
    blob = str(detail)
    assert "sk-1234567890abcdef" not in blob
    assert "ghp_abcdefghijklmnopqrstuvwx" not in blob
    assert "[REDACTED]" in blob


# ---------------------------------------------------------------------------
# Experiments: create / list / get / delete / compare / failure handling
# ---------------------------------------------------------------------------

def _experiment_body(project_path: str, **overrides):
    body = {
        "projectPath": project_path,
        "model": "llama3.2",
        "tokenizer": "llama3.2-tokenizer",
        "fineTuningMethod": "lora",
        "datasetVersion": "v1",
        "hyperparameters": {"learningRate": 0.0002, "epochs": 3},
        "hardware": {"gpu": "RTX 4090", "vramGb": 24},
        "estimates": {"trainingSeconds": 14400, "source": "gpu-time-estimator"},
        "metrics": {},
        "status": "estimated",
        "notes": "baseline run",
        "tags": ["baseline"],
    }
    body.update(overrides)
    return body


def test_experiment_create_and_retrieve(client):
    resp = client.post("/api/v1/experiments", json=_experiment_body(r"C:\work\exp"))
    assert resp.status_code == 201
    created = resp.json()
    assert created["model"] == "llama3.2"
    assert created["fineTuningMethod"] == "lora"
    assert created["status"] == "estimated"
    assert created["estimates"]["trainingSeconds"] == 14400
    assert created["metrics"] == {}  # no actual metrics were known — none invented
    assert created["actualTrainingTimeSeconds"] is None

    fetched = client.get(f"/api/v1/experiments/{created['experimentId']}")
    assert fetched.status_code == 200
    assert fetched.json()["experimentId"] == created["experimentId"]
    assert fetched.json()["hyperparameters"] == {"learningRate": 0.0002, "epochs": 3}


def test_experiment_requires_model(client):
    resp = client.post(
        "/api/v1/experiments",
        json={"projectPath": r"C:\work\exp", "model": ""},
    )
    assert resp.status_code in (400, 422)


def test_experiment_actual_vs_estimated_and_failed_status(client):
    # Failed experiment with no fabricated results.
    resp = client.post(
        "/api/v1/experiments",
        json=_experiment_body(
            r"C:\work\exp",
            status="failed",
            error="CUDA out of memory after 2 steps",
        ),
    )
    assert resp.status_code == 201
    record = resp.json()
    assert record["status"] == "failed"
    assert "CUDA out of memory" in record["error"]
    assert record["metrics"] == {}

    # Completed experiment with actual, user-reported results.
    resp2 = client.post(
        "/api/v1/experiments",
        json=_experiment_body(
            r"C:\work\exp",
            status="completed",
            metrics={"evalLoss": 1.23, "accuracy": 0.81},
            actualTrainingTimeSeconds=15000.0,
        ),
    )
    assert resp2.status_code == 201
    done = resp2.json()
    assert done["status"] == "completed"
    assert done["metrics"] == {"evalLoss": 1.23, "accuracy": 0.81}
    assert done["actualTrainingTimeSeconds"] == 15000.0
    # Estimated values remain distinguishable from actual ones.
    assert done["estimates"] == {"trainingSeconds": 14400, "source": "gpu-time-estimator"}


def test_experiment_listing_is_project_isolated(client):
    a = client.post("/api/v1/experiments", json=_experiment_body(r"C:\clients\A\shared-name")).json()
    b = client.post("/api/v1/experiments", json=_experiment_body(r"C:\clients\B\shared-name")).json()

    list_a = client.get("/api/v1/experiments", params={"projectPath": r"C:\clients\A\shared-name"})
    list_b = client.get("/api/v1/experiments", params={"projectPath": r"C:\clients\B\shared-name"})
    assert list_a.status_code == 200 and list_b.status_code == 200
    ids_a = {e["experimentId"] for e in list_a.json()["experiments"]}
    ids_b = {e["experimentId"] for e in list_b.json()["experiments"]}
    assert ids_a == {a["experimentId"]}
    assert ids_b == {b["experimentId"]}
    assert ids_a.isdisjoint(ids_b)

    # Unknown project: empty, not someone else's records.
    none = client.get("/api/v1/experiments", params={"projectPath": r"C:\clients\Z\shared-name"})
    assert none.json()["experiments"] == []


def test_experiments_require_project_path(client):
    assert client.get("/api/v1/experiments").status_code == 400


def test_experiment_duplicate_creates_distinct_records(client):
    first = client.post("/api/v1/experiments", json=_experiment_body(r"C:\work\dup-exp")).json()
    second = client.post("/api/v1/experiments", json=_experiment_body(r"C:\work\dup-exp")).json()
    assert first["experimentId"] != second["experimentId"]
    listed = client.get("/api/v1/experiments", params={"projectPath": r"C:\work\dup-exp"}).json()
    assert len(listed["experiments"]) == 2


def test_experiment_get_delete_404(client):
    assert client.get("/api/v1/experiments/nope").status_code == 404
    assert client.delete("/api/v1/experiments/nope").status_code == 404

    created = client.post("/api/v1/experiments", json=_experiment_body(r"C:\work\del")).json()
    deleted = client.delete(f"/api/v1/experiments/{created['experimentId']}")
    assert deleted.status_code == 200
    assert client.get(f"/api/v1/experiments/{created['experimentId']}").status_code == 404
    # Explicit delete only — nothing else was pruned.
    listed = client.get("/api/v1/experiments", params={"projectPath": r"C:\work\del"}).json()
    assert listed["experiments"] == []


def test_experiment_compare(client):
    e1 = client.post("/api/v1/experiments", json=_experiment_body(r"C:\work\cmp")).json()
    e2 = client.post(
        "/api/v1/experiments",
        json=_experiment_body(r"C:\work\cmp", model="mistral-7b"),
    ).json()

    resp = client.post(
        "/api/v1/experiments/compare",
        json={"experimentIds": [e1["experimentId"], e2["experimentId"]], "projectPath": r"C:\work\cmp"},
    )
    assert resp.status_code == 200
    experiments = resp.json()["experiments"]
    assert [e["experimentId"] for e in experiments] == [e1["experimentId"], e2["experimentId"]]
    assert {e["model"] for e in experiments} == {"llama3.2", "mistral-7b"}
    # Comparison-ready fields retained on both records.
    for record in experiments:
        assert "hyperparameters" in record
        assert "hardware" in record
        assert "estimates" in record
        assert "fineTuningMethod" in record

    # Fewer than two ids is a validation error.
    assert client.post(
        "/api/v1/experiments/compare", json={"experimentIds": [e1["experimentId"]]}
    ).status_code == 400
    # Missing ids surface as a precise 400, not silent omission.
    missing = client.post(
        "/api/v1/experiments/compare",
        json={"experimentIds": [e1["experimentId"], "does-not-exist"]},
    )
    assert missing.status_code == 400
    assert "does-not-exist" in str(missing.json())


def test_experiment_analysis_link_must_belong_to_same_project(client):
    """An analysis from project B can never be attached to project A's experiment."""
    run_id = _sample_run(r"C:\clients\A\link", "link-a")
    body = _experiment_body(r"C:\clients\B\link", analysisId=run_id)
    resp = client.post("/api/v1/experiments", json=body)
    assert resp.status_code == 400
    assert "different project" in str(resp.json())

    # The same link works from the owning project.
    ok = client.post(
        "/api/v1/experiments",
        json=_experiment_body(r"C:\clients\A\link", analysisId=run_id),
    )
    assert ok.status_code == 201
    assert ok.json()["analysisId"] == run_id