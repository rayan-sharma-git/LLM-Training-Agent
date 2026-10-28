"""Server-side store for questions the agent asked and is waiting on.

Only a card that exists here can be answered, an answer is accepted once, and
the continuation always runs from the payload stored *here* - never from
anything the client sends back - so a forged or replayed request cannot
authorise a different action than the one the user was shown.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from chat.store import normalize_project_path
from storage.database import get_session
from storage.models import AgentPendingAction

PENDING_TTL = timedelta(days=7)


def _as_dict(row: AgentPendingAction) -> Dict[str, Any]:
    return {
        "actionId": row.action_id,
        "projectPath": row.project_path,
        "sessionId": row.session_id,
        "messageId": row.message_id,
        "kind": row.kind,
        "payload": row.payload or {},
        "status": row.status,
        "choice": row.choice,
    }


async def create(
    project_path: Optional[str],
    session_id: Optional[str],
    kind: str,
    payload: Dict[str, Any],
) -> str:
    action_id = str(uuid.uuid4())
    async with get_session() as db:
        db.add(AgentPendingAction(
            action_id=action_id,
            project_path=normalize_project_path(project_path),
            session_id=session_id,
            kind=kind,
            payload=payload,
            status="pending",
            created_at=datetime.now(timezone.utc),
        ))
        await db.commit()
    return action_id


async def attach_message(action_id: str, message_id: str) -> None:
    async with get_session() as db:
        row = await db.get(AgentPendingAction, action_id)
        if row is not None:
            row.message_id = message_id
            await db.commit()


async def get(project_path: Optional[str], action_id: str) -> Optional[Dict[str, Any]]:
    """The action, only when it belongs to *project_path*."""
    async with get_session() as db:
        row = await db.get(AgentPendingAction, action_id)
        if row is None or row.project_path != normalize_project_path(project_path):
            return None
        return _as_dict(row)


async def claim(project_path: Optional[str], action_id: str, choice: str) -> Optional[Dict[str, Any]]:
    """Atomically mark a pending action answered; ``None`` if it cannot be answered.

    A second answer (double click, replay, another window) finds it already
    resolved and gets ``None``.
    """
    async with get_session() as db:
        row = await db.get(AgentPendingAction, action_id)
        if row is None or row.project_path != normalize_project_path(project_path):
            return None
        if row.status != "pending":
            return None
        created = row.created_at
        if created is not None:
            created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - created > PENDING_TTL:
                row.status = "expired"
                await db.commit()
                return None
        row.status = "resolved"
        row.choice = choice
        row.resolved_at = datetime.now(timezone.utc)
        await db.commit()
        return _as_dict(row)


__all__ = ["create", "attach_message", "get", "claim"]
