"""Workspace-scoped path resolution for the file-change workflow.

This module is the **single** path-resolution entry point for project file
operations.  It deliberately delegates to :mod:`core.security` so that the
backend applies exactly the same workspace boundary rules to file changes as
it already applies to datasets, providers and analysis:

* the project root must live inside an authorized root
  (``LLM_TRAINING_AGENT_ALLOWED_ROOTS``, defaulting to the working directory);
* the requested path must be workspace-relative (``..`` and absolute paths are
  rejected);
* the fully resolved path must still be inside the project, which also defeats
  symlink escapes because resolution happens before the containment check.

Historically this module implemented its own, weaker check, which meant the
``/files/*`` routes enforced different rules from every other route.  The
duplicate implementation has been removed.
"""
from __future__ import annotations

from pathlib import Path

from core.security import resolve_workspace_child, resolve_workspace_root

__all__ = ["resolve_project_file", "resolve_project_root"]


def resolve_project_root(project_root: str) -> Path:
    """Validate *project_root* and return it resolved as an authorized directory."""
    return resolve_workspace_root(project_root)


def resolve_project_file(project_root: Path, file_path: str) -> Path:
    """Resolve a workspace-relative *file_path* safely inside *project_root*.

    Raises ``fastapi.HTTPException`` (400/404) on an invalid or escaping path;
    callers in the API layer already treat that as the canonical error shape.
    """
    return resolve_workspace_child(str(project_root), file_path, must_exist=False)

