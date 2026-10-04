"""Tests for dynamic authorized-root resolution (Project A <-> Project B switching)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from core import security
from core.security import allowed_roots, resolve_workspace_root


@pytest.fixture(autouse=True)
def _clear_roots_cache():
    """Each test starts with a cold cache so mtime invalidation is exercised."""
    security._roots_cache = None
    yield
    security._roots_cache = None


def test_env_var_is_authoritative(tmp_path, monkeypatch):
    project_a = tmp_path / "ProjectA"
    project_a.mkdir()
    monkeypatch.delenv(security.ALLOWED_ROOTS_FILE_ENV, raising=False)
    monkeypatch.setenv(security.ALLOWED_ROOTS_ENV, str(project_a))
    assert allowed_roots() == [project_a.resolve()]


def test_roots_file_overrides_the_startup_environment(tmp_path, monkeypatch):
    """The whole point of the roots file: Project A -> Project B switching."""
    project_a = tmp_path / "ProjectA"
    project_b = tmp_path / "ProjectB"
    project_a.mkdir()
    project_b.mkdir()

    roots_file = tmp_path / "allowed-roots.txt"
    monkeypatch.setenv(security.ALLOWED_ROOTS_FILE_ENV, str(roots_file))
    # The backend started while Project A was open.
    monkeypatch.setenv(security.ALLOWED_ROOTS_ENV, str(project_a))

    roots_file.write_text(str(project_a), encoding="utf-8")
    assert allowed_roots() == [project_a.resolve()]
    assert resolve_workspace_root(str(project_a))

    # The user closes Project A and opens Project B; the extension rewrites the
    # file. No restart, no reinstall.
    roots_file.write_text(str(project_b), encoding="utf-8")
    assert allowed_roots() == [project_b.resolve()]

    # Project A is now refused: the two projects stay isolated.
    with pytest.raises(Exception) as excinfo:
        resolve_workspace_root(str(project_a))
    assert "WORKSPACE_FORBIDDEN" in str(excinfo.value.detail)


def test_empty_roots_file_denies_everything(tmp_path, monkeypatch):
    """No folder open must not fall back to the backend's own directory."""
    project_b = tmp_path / "ProjectB"
    project_b.mkdir()
    roots_file = tmp_path / "allowed-roots.txt"
    roots_file.write_text("", encoding="utf-8")
    monkeypatch.setenv(security.ALLOWED_ROOTS_FILE_ENV, str(roots_file))
    monkeypatch.setenv(security.ALLOWED_ROOTS_ENV, str(project_b))

    assert allowed_roots() == []
    with pytest.raises(Exception) as excinfo:
        resolve_workspace_root(str(project_b))
    assert "WORKSPACE_FORBIDDEN" in str(excinfo.value.detail)


def test_roots_file_supports_multiple_folders(tmp_path, monkeypatch):
    project_a = tmp_path / "ProjectA"
    project_b = tmp_path / "ProjectB"
    project_a.mkdir()
    project_b.mkdir()
    roots_file = tmp_path / "allowed-roots.txt"
    roots_file.write_text(f"{project_a}{os.pathsep}{project_b}", encoding="utf-8")
    monkeypatch.setenv(security.ALLOWED_ROOTS_FILE_ENV, str(roots_file))
    monkeypatch.delenv(security.ALLOWED_ROOTS_ENV, raising=False)

    assert allowed_roots() == [project_a.resolve(), project_b.resolve()]


def test_missing_roots_file_denies_rather_than_guesses(tmp_path, monkeypatch):
    monkeypatch.setenv(security.ALLOWED_ROOTS_FILE_ENV, str(tmp_path / "nope.txt"))
    monkeypatch.delenv(security.ALLOWED_ROOTS_ENV, raising=False)
    assert allowed_roots() == []


def test_no_configuration_falls_back_to_cwd(monkeypatch):
    """Standalone `uvicorn` usage keeps working."""
    monkeypatch.delenv(security.ALLOWED_ROOTS_ENV, raising=False)
    monkeypatch.delenv(security.ALLOWED_ROOTS_FILE_ENV, raising=False)
    assert allowed_roots() == [Path.cwd().resolve()]


def test_extension_roots_never_include_the_backend_directory(tmp_path, monkeypatch):
    """The agent's own install folder is never an authorized project root."""
    install = tmp_path / "extensions" / "rayansharma.llm-training-agent-1.0.0"
    (install / "backend").mkdir(parents=True)
    project = tmp_path / "ProjectA"
    project.mkdir()

    roots_file = tmp_path / "allowed-roots.txt"
    roots_file.write_text(str(project), encoding="utf-8")
    monkeypatch.setenv(security.ALLOWED_ROOTS_FILE_ENV, str(roots_file))
    monkeypatch.delenv(security.ALLOWED_ROOTS_ENV, raising=False)

    assert allowed_roots() == [project.resolve()]
    with pytest.raises(Exception):
        resolve_workspace_root(str(install))
