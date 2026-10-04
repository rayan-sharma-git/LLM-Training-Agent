"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.routes import router
from fastapi import FastAPI


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_project_missing_path(client):
    response = client.post("/api/v1/project/analyze", json={})
    assert response.status_code == 422 or response.status_code == 400


def test_dataset_analyze_missing_path(client):
    response = client.post("/api/v1/dataset/analyze", json={})
    assert response.status_code == 422 or response.status_code == 400