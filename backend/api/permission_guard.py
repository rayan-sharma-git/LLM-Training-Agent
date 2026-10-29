"""HTTP-layer guard that applies the stored permission policy to a request."""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

from fastapi import HTTPException

from core import permissions as perms
from core.permissions_store import load_policy, project_key

# One-time grants made by pressing "Allow" on a permission card.  They live in
# memory only (never persisted), expire quickly and are consumed by the first
# matching request, so "Allow" cannot silently turn into a standing permission.
_ALLOW_ONCE_TTL_SECONDS = 120.0
_allow_once: Dict[Tuple[str, str], float] = {}
_allow_lock = threading.Lock()
_GRANTABLE = {perms.READ, perms.WRITE}


def grant_once(project_path: Optional[str], action: str) -> None:
    """Allow the next *action* for the project (read/write only; commands use cards)."""
    if action not in _GRANTABLE:
        raise ValueError("Only reading and writing project files can be allowed once here.")
    with _allow_lock:
        _allow_once[(project_key(project_path), action)] = time.monotonic() + _ALLOW_ONCE_TTL_SECONDS


def _consume_once(project_path: Optional[str], action: str) -> bool:
    key = (project_key(project_path), action)
    with _allow_lock:
        expires = _allow_once.pop(key, None)
    return expires is not None and expires >= time.monotonic()


async def require_permission(project_path: Optional[str], action: str) -> None:
    """Raise HTTP 403 ``PERMISSION_REQUIRED`` unless *action* is enabled for the project.

    This is the enforcement point for routes that touch project files directly,
    so a client that skips the chat UI cannot skip the setting either.
    """
    policy = await load_policy(project_path)
    try:
        perms.enforce(policy, action)
    except perms.PermissionError_ as error:
        if not isinstance(error, perms.PermissionBlocked) and _consume_once(project_path, action):
            return
        raise HTTPException(
            status_code=403,
            detail={"errorCode": "PERMISSION_REQUIRED", "action": error.action, "message": error.message},
        )
