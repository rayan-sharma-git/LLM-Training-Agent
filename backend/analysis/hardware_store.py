"""Per-project hardware profile (cloud vs local GPU, hourly rate) kept in SQLite.

The profile is set when the user *states* their hardware in chat ("I train on a
RunPod A100 at $1.89/hr").  Nothing is ever guessed or defaulted: with no
profile stored the analysis uses the GPU detected on this machine and produces
no rental cost.
"""
from __future__ import annotations

import logging
from typing import Optional

from analysis.project_findings import HardwareProfile
from core.permissions_store import project_key
from storage.database import get_session
from storage.models import Setting

logger = logging.getLogger(__name__)


def _key(project_path: Optional[str]) -> str:
    return f"hardware_profile::{project_key(project_path)}"


async def load_profile(project_path: Optional[str]) -> Optional[HardwareProfile]:
    """Return the stored profile for a project, or ``None`` (never raises)."""
    try:
        async with get_session() as session:
            row = await session.get(Setting, _key(project_path))
            if row is None:
                return None
            profile = HardwareProfile.from_dict(row.value)
            return profile if profile.is_set else None
    except Exception as error:  # storage problems must not break an analysis
        logger.warning("Could not load hardware profile: %s", error)
        return None


async def save_profile(project_path: Optional[str], profile: HardwareProfile) -> bool:
    """Persist (merge into) the project's profile; returns False on a storage error."""
    try:
        async with get_session() as session:
            row = await session.get(Setting, _key(project_path))
            merged = HardwareProfile.from_dict(row.value) if row is not None else HardwareProfile()
            incoming = profile.to_dict()
            data = merged.to_dict()
            for field_name, value in incoming.items():
                if field_name == "mode" and value == "auto":
                    continue
                if value in (None, "", 0):
                    continue
                data[field_name] = value
            if row is None:
                session.add(Setting(key=_key(project_path), value=data))
            else:
                row.value = data
            await session.commit()
        return True
    except Exception as error:
        logger.warning("Could not save hardware profile: %s", error)
        return False


async def clear_profile(project_path: Optional[str]) -> bool:
    try:
        async with get_session() as session:
            row = await session.get(Setting, _key(project_path))
            if row is not None:
                await session.delete(row)
                await session.commit()
        return True
    except Exception as error:
        logger.warning("Could not clear hardware profile: %s", error)
        return False
