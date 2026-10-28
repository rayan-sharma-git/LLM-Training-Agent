"""Persistence of the permission policy, scoped to a project path.

The VS Code extension pushes the user's Settings → Permissions here whenever
they change (and on start-up).  The backend - not the UI - reads the stored
policy on every file or command operation.
"""
from __future__ import annotations

import logging
from typing import Optional

import os
from core.permissions import PermissionPolicy
from storage.database import get_session
from storage.models import Setting

logger = logging.getLogger(__name__)


def project_key(project_path: Optional[str]) -> str:
    """Stable key for a project: symlink-resolved and case-normalised, so the
    path the editor sends and the resolved path the file routes see agree."""
    if not project_path or not str(project_path).strip():
        return "_no_project"
    return os.path.normcase(os.path.realpath(str(project_path).strip()))


def _key(project_path: Optional[str]) -> str:
    return f"permissions::{project_key(project_path)}"


async def load_policy(project_path: Optional[str]) -> PermissionPolicy:
    """The stored policy, or the safe default (read only) when none was saved.

    A storage failure also yields the safe default - never a wider policy.
    """
    try:
        async with get_session() as session:
            row = await session.get(Setting, _key(project_path))
            return PermissionPolicy.from_dict(row.value) if row is not None else PermissionPolicy()
    except Exception as error:
        logger.warning("Could not load permissions, using safe defaults: %s", error)
        return PermissionPolicy()


async def save_policy(project_path: Optional[str], policy: PermissionPolicy) -> None:
    async with get_session() as session:
        row = await session.get(Setting, _key(project_path))
        if row is None:
            session.add(Setting(key=_key(project_path), value=policy.to_dict()))
        else:
            row.value = policy.to_dict()
        await session.commit()


__all__ = ["load_policy", "save_policy"]
