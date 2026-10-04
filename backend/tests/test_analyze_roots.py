"""The analyze endpoint must refuse projects outside the authorized roots.

This is what keeps Project A and Project B isolated: after the user closes
Project A and opens Project B, Project A must no longer be analyzable.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from core import security


@pytest.fixture()
def client(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hi')\n", encoding="utf-8")

    monkeypatch.setenv(security.ALLOWED_ROOTS_ENV, str(project))
    monkeypatch.delenv(security.ALLOWED_ROOTS_FILE_ENV, raising=False)
    security._roots_cache = None

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    yield TestClient(app), project
    security._roots_cache = None


def test_analyze_accepts_the_open_project(client):
    test_client, project = client
    response = test_client.post(
        "/api/v1/project/analyze", json={"projectPath": str(project)}
    )
    assert response.status_code == 200, response.text


def test_analyze_rejects_a_project_outside_the_authorized_roots(client, tmp_path):
    test_client, _ = client
    outsider = tmp_path / "other-project"
    outsider.mkdir()
    (outsider / "main.py").write_text("print('nope')\n", encoding="utf-8")

    response = test_client.post(
        "/api/v1/project/analyze", json={"projectPath": str(outsider)}
    )
    assert response.status_code == 403
    assert "WORKSPACE_FORBIDDEN" in response.text


def test_analyze_rejects_a_missing_project_path(client):
    test_client, _ = client
    response = test_client.post("/api/v1/project/analyze", json={})
    assert response.status_code in (400, 422)


def test_analyze_rejects_a_nonexistent_folder(client, tmp_path):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/project/analyze",
        json={"projectPath": str(tmp_path / "does-not-exist")},
    )
    assert response.status_code == 404


def test_switching_projects_isolates_them(client, tmp_path, monkeypatch):
    """Project A -> Project B: A becomes unauthorized immediately."""
    test_client, project_a = client
    assert test_client.post(
        "/api/v1/project/analyze", json={"projectPath": str(project_a)}
    ).status_code == 200

    project_b = tmp_path / "projectB"
    project_b.mkdir()
    (project_b / "main.py").write_text("print('b')\n", encoding="utf-8")

    # The extension re-publishes the authorized roots when the workspace
    # changes; here we simulate that by switching the env var.
    monkeypatch.setenv(security.ALLOWED_ROOTS_ENV, str(project_b))
    security._roots_cache = None

    assert test_client.post(
        "/api/v1/project/analyze", json={"projectPath": str(project_b)}
    ).status_code == 200
    assert test_client.post(
        "/api/v1/project/analyze", json={"projectPath": str(project_a)}
    ).status_code == 403

