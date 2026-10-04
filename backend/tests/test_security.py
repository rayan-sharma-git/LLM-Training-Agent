"""Security regression tests for trust boundaries and secret handling."""
from __future__ import annotations


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from core.security import (
    resolve_dataset_path,
    resolve_output_dir,
    resolve_workspace_child,
    validate_provider_url,
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_traversal_and_absolute_escape_are_rejected(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))
    with pytest.raises(Exception):
        resolve_workspace_child(str(root), "../outside.txt")
    with pytest.raises(ValueError):
        resolve_dataset_path(str(root), "../secret.jsonl")
    with pytest.raises(ValueError):
        resolve_dataset_path(str(root), str(tmp_path / "outside.jsonl"))
    with pytest.raises(ValueError):
        resolve_output_dir(str(root), "../outside", "cleaned")


def test_symlink_escape_is_rejected_when_supported(tmp_path, monkeypatch):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(outside / "secret.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))
    with pytest.raises(Exception):
        resolve_workspace_child(str(root), "link.txt")


def test_provider_url_policy_blocks_metadata_and_plain_remote_http():
    assert validate_provider_url("http://127.0.0.1:11434", allow_remote=False)
    assert validate_provider_url("https://api.example.com/v1")
    for value in ("http://169.254.169.254/latest/meta-data", "http://10.0.0.1/admin", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            validate_provider_url(value)


def test_api_rejects_oversized_proposal_and_does_not_modify_file(client, tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    target = root / "app.py"
    target.write_text("safe\n", encoding="utf-8")
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))
    response = client.post("/api/v1/files/propose", json={"projectRoot": str(root), "filePath": "app.py", "proposedContent": "x" * 2_000_001})
    assert response.status_code == 422
    assert target.read_text(encoding="utf-8") == "safe\n"


def test_malicious_dataset_path_is_not_read(client, tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"prompt":"secret","response":"leak"}\n', encoding="utf-8")
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))
    response = client.post("/api/v1/dataset/clean", json={"projectPath": str(root), "datasetPaths": [str(outside)], "useLlm": False})
    assert response.status_code in (400, 403, 422)
    assert not (root / "cleaned").exists()


def test_dataset_analyze_refuses_paths_outside_the_workspace(client, tmp_path, monkeypatch):
    """``/dataset/analyze`` must not read a file outside an authorized root."""
    root = tmp_path / "project"
    root.mkdir()
    secret = tmp_path / "secret.jsonl"
    secret.write_text('{"prompt":"TOPSECRET","response":"leak"}\n', encoding="utf-8")
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))

    # Absolute path outside the authorized root.
    absolute = client.post("/api/v1/dataset/analyze", json={"datasetPath": str(secret)})
    assert absolute.status_code == 403
    assert "TOPSECRET" not in absolute.text

    # Relative traversal escaping the authorized root.
    traversal = client.post(
        "/api/v1/dataset/analyze",
        json={"projectPath": str(root), "datasetPath": "../secret.jsonl"},
    )
    assert traversal.status_code == 403
    assert "TOPSECRET" not in traversal.text


def test_dataset_analyze_requires_a_dataset_path(client, tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))
    response = client.post("/api/v1/dataset/analyze", json={})
    assert response.status_code == 400


def test_dataset_analyze_accepts_a_workspace_dataset(client, tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    dataset = root / "train.jsonl"
    dataset.write_text('{"prompt":"hi","response":"yo"}\n', encoding="utf-8")
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))
    response = client.post(
        "/api/v1/dataset/analyze",
        json={"projectPath": str(root), "datasetPath": "train.jsonl"},
    )
    assert response.status_code == 200
    assert "dataset_name" in response.json()


def test_compare_experiments_does_not_cross_project_boundaries(
    client, tmp_path, monkeypatch
):
    """A projectPath with no stored experiments must not return records."""
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))

    created = [
        client.post(
            "/api/v1/experiments",
            json={"projectPath": str(root), "model": model},
        )
        for model in ("llama3.2", "mistral")
    ]
    assert [r.status_code for r in created] == [201, 201]
    ids = [r.json()["experimentId"] for r in created]

    same = client.post(
        "/api/v1/experiments/compare",
        json={"experimentIds": ids, "projectPath": str(root)},
    )
    assert same.status_code == 200
    assert len(same.json()["experiments"]) == 2

    # Fewer than two ids is a client error.
    assert (
        client.post(
            "/api/v1/experiments/compare",
            json={"experimentIds": ids[:1], "projectPath": str(root)},
        ).status_code
        == 400
    )


def test_websocket_disconnect_tolerates_an_unknown_socket():
    """``disconnect`` must not raise when the socket is already gone."""
    from api.websocket import manager

    class Unregistered:
        pass

    manager.disconnect(Unregistered())  # must not raise

