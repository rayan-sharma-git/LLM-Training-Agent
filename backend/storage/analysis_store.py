"""Analysis-run persistence: project identity, run lifecycle, retrieval.

This module is the single place that writes/reads analysis history from
SQLite.  Design rules:

* **Project identity** is the *normalized absolute project path*
  (``chat.store.normalize_project_path`` — case-folded on Windows), never the
  display folder name.  Two different folders called ``my-project`` are two
  different projects; ``C:\\Proj`` and ``c:\\proj`` are the same project.
* **One row per run.**  A new analysis never overwrites a previous run;
  retrieval always filters by project and (when given) run id.
* **Statuses**: ``running`` → ``completed`` | ``partial`` | ``failed``.
  ``partial`` means the run finished but a critical step failed; a run is
  never reported as successful when critical processing failed.
* **Everything written is secret-scrubbed** via ``core.redaction`` — analyzer
  findings may quote project files, which can contain credentials.
* Storage failures raise :class:`~core.errors.StorageError`; this module
  never fabricates empty or placeholder results.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import desc, select

from chat.store import normalize_project_path
from core.errors import StorageError
from core.redaction import scrub, scrub_obj
from storage.database import get_database, get_session, storage_error_from
from storage.models import Analysis, Prediction, Project

logger = logging.getLogger(__name__)

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

#: Analyzer/pipeline steps whose failure makes the run *partial* rather than
#: successful.  Advisory steps (cost, prediction, GPU estimate, optional
#: cleaning) are recorded as non-critical warnings.
CRITICAL_STEPS = frozenset({"dataset", "prompt", "hyperparameters", "model", "report"})


def _jsonable(value: Any) -> Any:
    """Convert pydantic models / dataclasses / dicts into JSON-safe data."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _iso(value: Optional[datetime]) -> Optional[str]:
    """ISO-8601 (UTC) rendering of a stored timestamp."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def run_failure(step: str, error: Any, critical: bool) -> Dict[str, Any]:
    """Structured partial-failure entry (scrubbed)."""
    return {
        "step": str(step),
        "error": scrub(str(error))[:1000],
        "critical": bool(critical),
    }


async def get_or_create_project(
    session,
    project_path: str,
    project_name: Optional[str] = None,
    framework: Optional[str] = None,
    framework_version: Optional[str] = None,
    base_model: Optional[str] = None,
    tokenizer: Optional[str] = None,
) -> str:
    """Return the project id for *project_path*, creating the row if needed.

    Identity = normalized absolute path (see module docstring).  The display
    name is only metadata; it never determines identity, so two different
    folders sharing a name can never merge into one project.
    """
    identity = normalize_project_path(project_path)
    result = await session.execute(select(Project).where(Project.project_path == identity))
    project = result.scalar_one_or_none()

    if project is None:
        project = Project(
            project_id=str(uuid.uuid4()),
            project_name=project_name or os.path.basename(identity.rstrip("/\\")) or identity,
            project_path=identity,
            framework=framework,
            framework_version=framework_version,
            base_model=base_model,
            tokenizer=tokenizer,
        )
        session.add(project)
        try:
            await session.flush()
        except Exception:
            # Concurrent insert of the same path: reuse the winner's row.
            await session.rollback()
            result = await session.execute(
                select(Project).where(Project.project_path == identity)
            )
            project = result.scalar_one_or_none()
            if project is None:
                raise
    return project.project_id


async def upsert_project_metadata(
    *,
    project_path: str,
    project_name: Optional[str] = None,
    framework: Optional[str] = None,
    framework_version: Optional[str] = None,
    base_model: Optional[str] = None,
    tokenizer: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert or refresh a project's metadata row (used by /project/refresh)."""
    try:
        async with get_session() as session:
            project_id = await get_or_create_project(
                session,
                project_path,
                project_name=project_name,
                framework=framework,
                framework_version=framework_version,
                base_model=base_model,
                tokenizer=tokenizer,
            )
            project = await session.get(Project, project_id)
            if project is not None:
                for key, value in {
                    "framework": framework,
                    "framework_version": framework_version,
                    "base_model": base_model,
                    "tokenizer": tokenizer,
                }.items():
                    if value:
                        setattr(project, key, value)
            await session.commit()
            return {
                "projectId": project_id,
                "projectPath": normalize_project_path(project_path),
                "projectName": project.project_name if project else project_name,
            }
    except StorageError:
        raise
    except Exception as error:
        raise storage_error_from(error, get_database().database_url) from error


async def start_run(
    project_path: str,
    *,
    project_name: Optional[str] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a ``running`` analysis row and return its id.

    The config snapshot records the configuration the run was performed with
    (provider/model names, framework, dataset paths — never credentials).
    """
    try:
        async with get_session() as session:
            project_id = await get_or_create_project(session, project_path, project_name=project_name)
            analysis_id = str(uuid.uuid4())
            session.add(
                Analysis(
                    analysis_id=analysis_id,
                    project_id=project_id,
                    status=STATUS_RUNNING,
                    config_snapshot=scrub_obj(_jsonable(config_snapshot) or {}),
                )
            )
            await session.commit()
            logger.info("Analysis run %s started for %s", analysis_id, normalize_project_path(project_path))
            return analysis_id
    except StorageError:
        raise
    except Exception as error:
        raise storage_error_from(error, get_database().database_url) from error


def _as_float(value: Any) -> Optional[float]:
    """Best-effort numeric coercion — returns None instead of guessing."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_int(value: Any) -> Optional[int]:
    number = _as_float(value)
    return int(number) if number is not None else None


async def complete_run(
    analysis_id: str,
    *,
    context: Any = None,
    report: Any = None,
    results: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[Any]] = None,
    partial_failures: Optional[List[Dict[str, Any]]] = None,
    health_score: Optional[float] = None,
    readiness_score: Optional[float] = None,
    status: str = STATUS_COMPLETED,
) -> bool:
    """Persist a finished run (status ``completed``/``partial``) transactionally.

    Analyzer outputs are secret-scrubbed, then stored twice: queryable summary
    scalars in the legacy columns and the full JSON in ``payload`` so no field
    is lost to column mismatches.  Returns False when the run row no longer
    exists.
    """
    from storage.models import (
        CostEstimateRecord,
        DatasetReport,
        HyperparameterReport,
        ModelReport,
        PromptReport,
        RecommendationRecord,
    )

    if status not in (STATUS_COMPLETED, STATUS_PARTIAL):
        raise ValueError(f"Invalid terminal status for complete_run: {status}")

    results = {
        name: scrub_obj(_jsonable(value))
        for name, value in (results or {}).items()
    }
    context_data = scrub_obj(_jsonable(context)) if context is not None else None
    report_data = scrub_obj(_jsonable(report)) if report is not None else None
    failures = scrub_obj(_jsonable(partial_failures) or [])

    try:
        async with get_session() as session:
            analysis = await session.get(Analysis, analysis_id)
            if analysis is None:
                return False

            now = datetime.now(timezone.utc)
            started = analysis.started_at
            if started is not None and started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)

            analysis.status = status
            analysis.completed_at = now
            analysis.duration = int((now - started).total_seconds()) if started else None
            analysis.health_score = _as_float(health_score)
            analysis.readiness_score = _as_float(readiness_score)
            analysis.context = context_data
            analysis.report = report_data
            analysis.partial_failures = failures or None

            dataset = results.get("dataset") or {}
            if dataset:
                session.add(
                    DatasetReport(
                        analysis_id=analysis_id,
                        dataset_name=str(dataset.get("dataset_name") or "unknown"),
                        quality_score=_as_float(dataset.get("quality_score")),
                        duplicate_percentage=_as_float(dataset.get("duplicate_percentage")),
                        token_count=_as_int(dataset.get("token_count")),
                        findings=dataset.get("findings"),
                        recommendations=dataset.get("recommendations"),
                        payload=dataset,
                    )
                )

            prompt = results.get("prompt") or {}
            if prompt:
                session.add(
                    PromptReport(
                        analysis_id=analysis_id,
                        template_name=str(prompt.get("template_name") or "unknown"),
                        clarity_score=_as_float(prompt.get("clarity_score")),
                        ambiguity_score=_as_float(prompt.get("ambiguity_score")),
                        consistency_score=_as_float(prompt.get("consistency_score")),
                        findings=prompt.get("detected_issues"),
                        recommendations=prompt.get("recommendations"),
                        payload=prompt,
                    )
                )

            hp = results.get("hyperparameters") or {}
            if hp:
                session.add(
                    HyperparameterReport(
                        analysis_id=analysis_id,
                        learning_rate=_as_float(hp.get("learning_rate")),
                        batch_size=_as_int(hp.get("batch_size")),
                        epochs=_as_int(hp.get("epochs")),
                        optimizer=hp.get("optimizer"),
                        scheduler=hp.get("scheduler"),
                        lora_rank=_as_int(hp.get("lora_rank")),
                        lora_alpha=_as_int(hp.get("lora_alpha")),
                        findings=hp.get("findings"),
                        recommendations=hp.get("recommendations"),
                        payload=hp,
                    )
                )

            model_result = results.get("model") or {}
            if model_result:
                session.add(
                    ModelReport(
                        analysis_id=analysis_id,
                        model_name=str(model_result.get("selected_model") or "unknown"),
                        strengths=model_result.get("strengths"),
                        weaknesses=model_result.get("weaknesses"),
                        payload=model_result,
                    )
                )

            prediction = results.get("prediction") or {}
            if prediction:
                from storage.models import Prediction as PredictionRecord

                session.add(
                    PredictionRecord(
                        analysis_id=analysis_id,
                        predicted_failure_modes=prediction.get("likely_failure_modes"),
                        payload=prediction,
                    )
                )

            cost = results.get("cost") or {}
            if cost:
                session.add(
                    CostEstimateRecord(
                        analysis_id=analysis_id,
                        training_time=cost.get("estimated_training_time"),
                        gpu_hours=_as_float(cost.get("estimated_gpu_hours")),
                        vram=cost.get("estimated_vram_usage"),
                        storage=cost.get("estimated_storage_requirement"),
                        checkpoint_size=cost.get("estimated_checkpoint_size"),
                        payload=cost,
                    )
                )

            for raw in recommendations or []:
                rec = scrub_obj(_jsonable(raw)) or {}
                if not rec:
                    continue
                external_id = str(rec.get("recommendation_id") or "")
                # Recommendation IDs emitted by an analyzer are only unique
                # within one report.  The legacy table uses a global primary
                # key, so scope the stored key to this analysis while retaining
                # the original external ID in payload for API/report clients.
                stored_id = str(
                    uuid.uuid5(uuid.UUID(analysis_id), external_id or str(uuid.uuid4()))
                )
                session.add(
                    RecommendationRecord(
                        recommendation_id=stored_id,
                        analysis_id=analysis_id,
                        severity=rec.get("severity"),
                        confidence=_as_float(rec.get("confidence")),
                        title=str(rec.get("title") or "Untitled recommendation")[:255],
                        description=rec.get("description"),
                        evidence=rec.get("evidence"),
                        status="pending",
                        payload=rec,
                    )
                )

            await session.commit()
            logger.info(
                "Analysis run %s persisted with status=%s (%d partial failures)",
                analysis_id,
                status,
                len(failures or []),
            )
            return True
    except StorageError:
        raise
    except Exception as error:
        raise storage_error_from(error, get_database().database_url) from error


async def fail_run(analysis_id: Optional[str], error: Any) -> bool:
    """Mark a run ``failed`` with a scrubbed error message. Best-effort."""
    if not analysis_id:
        return False
    try:
        async with get_session() as session:
            analysis = await session.get(Analysis, analysis_id)
            if analysis is None:
                return False
            now = datetime.now(timezone.utc)
            started = analysis.started_at
            if started is not None and started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            analysis.status = STATUS_FAILED
            analysis.completed_at = now
            analysis.duration = int((now - started).total_seconds()) if started else None
            analysis.error = scrub(str(error))[:4000]
            await session.commit()
            logger.warning("Analysis run %s marked failed", analysis_id)
            return True
    except StorageError:
        raise
    except Exception as storage_problem:
        raise storage_error_from(storage_problem, get_database().database_url) from storage_problem


# ---------------------------------------------------------------------------
# Retrieval (always project-scoped — a different project never sees these rows)
# ---------------------------------------------------------------------------

def _summary(analysis: Analysis, project: Optional[Project]) -> Dict[str, Any]:
    failures = analysis.partial_failures or []
    return {
        "analysisId": analysis.analysis_id,
        "projectPath": project.project_path if project else None,
        "projectName": project.project_name if project else None,
        "status": analysis.status,
        "startedAt": _iso(analysis.started_at),
        "completedAt": _iso(analysis.completed_at),
        "durationSeconds": analysis.duration,
        "healthScore": analysis.health_score,
        "readinessScore": analysis.readiness_score,
        "failureCount": len(failures) + (1 if analysis.status == STATUS_FAILED else 0),
    }


async def list_runs(project_path: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Newest-first run summaries for one project (never cross-project)."""
    try:
        async with get_session() as session:
            identity = normalize_project_path(project_path)
            result = await session.execute(
                select(Project).where(Project.project_path == identity)
            )
            project = result.scalar_one_or_none()
            if project is None:
                return []
            runs = await session.execute(
                select(Analysis)
                .where(Analysis.project_id == project.project_id)
                .order_by(desc(Analysis.started_at), desc(Analysis.analysis_id))
                .limit(max(1, min(int(limit or 20), 200)))
            )
            return [_summary(analysis, project) for analysis in runs.scalars().all()]
    except StorageError:
        raise
    except Exception as error:
        raise storage_error_from(error, get_database().database_url) from error


def _detail(
    analysis: Analysis,
    project: Optional[Project],
    children: Sequence[Any],
) -> Dict[str, Any]:
    """Assemble a full run record from the analysis row + child rows."""
    by_table = {getattr(row, "__tablename__", None): row for row in children}
    results: Dict[str, Any] = {}
    for table, name in (
        ("dataset_reports", "dataset"),
        ("prompt_reports", "prompt"),
        ("hyperparameter_reports", "hyperparameters"),
        ("model_reports", "model"),
        ("predictions", "prediction"),
        ("cost_estimates", "cost"),
    ):
        row = by_table.get(table)
        if row is not None:
            results[name] = row.payload if row.payload is not None else {}

    recommendations = [
        row.payload if row.payload is not None else {"title": row.title}
        for row in children
        if getattr(row, "__tablename__", None) == "recommendations"
    ]

    return {
        **_summary(analysis, project),
        "error": analysis.error,
        "config": analysis.config_snapshot,
        "context": analysis.context,
        "report": analysis.report,
        "partialFailures": analysis.partial_failures or [],
        "results": results,
        "recommendations": recommendations,
    }


async def _load_children(session, analysis_id: str) -> List[Any]:
    """Load every child row of a run (table-tagged via ``__tablename__``)."""
    from storage.models import (
        CostEstimateRecord,
        DatasetReport,
        HyperparameterReport,
        ModelReport,
        PromptReport,
        RecommendationRecord,
    )

    children: List[Any] = []
    for model in (
        DatasetReport,
        PromptReport,
        HyperparameterReport,
        ModelReport,
        CostEstimateRecord,
        RecommendationRecord,
    ):
        result = await session.execute(select(model).where(model.analysis_id == analysis_id))
        children.extend(result.scalars().all())

    prediction = await session.execute(
        select(Prediction).where(Prediction.analysis_id == analysis_id)
    )
    children.extend(prediction.scalars().all())
    return children


async def get_run(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Full stored record for one run (None when it does not exist)."""
    try:
        async with get_session() as session:
            analysis = await session.get(Analysis, analysis_id)
            if analysis is None:
                return None
            project = await session.get(Project, analysis.project_id)
            children = await _load_children(session, analysis_id)
            return _detail(analysis, project, children)
    except StorageError:
        raise
    except Exception as error:
        raise storage_error_from(error, get_database().database_url) from error


async def get_latest_run(
    project_path: str,
    *,
    require_field: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Latest run of a project, optionally requiring a payload field.

    ``require_field`` ("context"/"report") skips runs that failed before that
    payload was produced, so callers never receive a half-empty record.
    Returns None when nothing is stored — callers turn that into an explicit
    404, never into fabricated empty data.
    """
    try:
        async with get_session() as session:
            identity = normalize_project_path(project_path)
            result = await session.execute(
                select(Project).where(Project.project_path == identity)
            )
            project = result.scalar_one_or_none()
            if project is None:
                return None
            runs = await session.execute(
                select(Analysis)
                .where(Analysis.project_id == project.project_id)
                .order_by(desc(Analysis.started_at), desc(Analysis.analysis_id))
            )
            candidates = list(runs.scalars().all())
            if require_field:
                candidates = [run for run in candidates if getattr(run, require_field, None)]
            if not candidates:
                return None
            analysis = candidates[0]
            children = await _load_children(session, analysis.analysis_id)
            return _detail(analysis, project, children)
    except StorageError:
        raise
    except Exception as error:
        raise storage_error_from(error, get_database().database_url) from error


__all__ = [
    "CRITICAL_STEPS",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_PARTIAL",
    "STATUS_RUNNING",
    "complete_run",
    "fail_run",
    "get_latest_run",
    "get_or_create_project",
    "get_run",
    "list_runs",
    "run_failure",
    "start_run",
    "upsert_project_metadata",
]