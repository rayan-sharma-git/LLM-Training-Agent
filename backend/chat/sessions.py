"""Persistent chat sessions (SQLite) scoped to a project path.

Replaces the process-memory conversation list for the chat UI: sessions and
their messages - including rich parts such as change viewers and action cards
- survive closing VS Code, restarting the backend and switching chats.  A
session is only reachable with the project path it was created for, so one
project's conversations never show up in another.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select

from chat.store import normalize_project_path
from storage.database import get_session
from storage.models import ChatMessageRecord, ChatSession, Project

DEFAULT_TITLE = "New chat"
MAX_TITLE_CHARS = 60
MAX_MESSAGE_CHARS = 200_000
MAX_LIST = 200

_FILLER = re.compile(
    r"^(hey|hi|hello|ok(ay)?|please|pls|kindly|could you|can you|would you|will you|i (want|need|would like) (you )?to|"
    r"i'd like (you )?to|let's|lets|just|so|um|and|also)\b[\s,:-]*", re.I)
_GREETING = re.compile(r"^(hi|hello|hey|yo|thanks?|thank you|ok(ay)?|test|\?+|help)[.!? ]*$", re.I)


def _utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    value = _utc(value)
    return value.isoformat() if value else None


def generate_title(text: str) -> Optional[str]:
    """A short title from a user message, or ``None`` when it says nothing useful."""
    cleaned = " ".join((text or "").split())
    if not cleaned or _GREETING.match(cleaned):
        return None
    for _ in range(3):  # strip stacked filler such as "hey, can you please"
        stripped = _FILLER.sub("", cleaned, count=1)
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = re.split(r"(?<=[.!?])\s|\n", cleaned, maxsplit=1)[0].strip(" .,:;!?\"'`")
    if len(cleaned) < 3:
        return None
    if len(cleaned) > MAX_TITLE_CHARS:
        cut = cleaned[:MAX_TITLE_CHARS].rsplit(" ", 1)[0]
        cleaned = (cut or cleaned[:MAX_TITLE_CHARS]).rstrip(" ,;:-") + "…"
    return cleaned[0].upper() + cleaned[1:]


def _message_dict(record: ChatMessageRecord) -> Dict[str, Any]:
    return {
        "messageId": record.message_id,
        "role": record.role,
        "content": record.content,
        "blocks": record.blocks or [],
        "createdAt": _iso(record.timestamp),
    }


async def _project_id(db, project_path: Optional[str]) -> str:
    key = normalize_project_path(project_path)
    row = (await db.execute(select(Project).where(Project.project_path == key))).scalar_one_or_none()
    if row is None:
        name = (key.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "project")[:255]
        row = Project(project_name=name, project_path=key)
        db.add(row)
        await db.flush()
    return row.project_id


async def _owned_session(db, project_path: Optional[str], session_id: str) -> Optional[ChatSession]:
    record = await db.get(ChatSession, session_id)
    if record is None:
        return None
    if record.project_id != await _project_id(db, project_path):
        return None
    return record


async def create_session(project_path: Optional[str], title: Optional[str] = None) -> Dict[str, Any]:
    async with get_session() as db:
        now = datetime.now(timezone.utc)
        record = ChatSession(
            session_id=str(uuid.uuid4()),
            project_id=await _project_id(db, project_path),
            title=title or None,
            created_at=now,
            updated_at=now,
        )
        db.add(record)
        await db.commit()
        return _summary(record, 0)


def _summary(record: ChatSession, count: int) -> Dict[str, Any]:
    return {
        "sessionId": record.session_id,
        "title": record.title or DEFAULT_TITLE,
        "createdAt": _iso(record.created_at),
        "updatedAt": _iso(record.updated_at),
        "messageCount": count,
    }


async def list_sessions(project_path: Optional[str], limit: int = 100) -> List[Dict[str, Any]]:
    """Sessions with at least one message, newest activity first."""
    async with get_session() as db:
        project_id = await _project_id(db, project_path)
        counts = (
            select(ChatMessageRecord.session_id, func.count().label("n"))
            .group_by(ChatMessageRecord.session_id).subquery()
        )
        rows = (await db.execute(
            select(ChatSession, counts.c.n)
            .join(counts, counts.c.session_id == ChatSession.session_id)
            .where(ChatSession.project_id == project_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(max(1, min(int(limit), MAX_LIST)))
        )).all()
        await db.commit()  # persists a Project row created on first access
        return [_summary(record, int(n)) for record, n in rows]


async def get_session_messages(project_path: Optional[str], session_id: str) -> Optional[Dict[str, Any]]:
    async with get_session() as db:
        record = await _owned_session(db, project_path, session_id)
        if record is None:
            return None
        rows = (await db.execute(
            select(ChatMessageRecord).where(ChatMessageRecord.session_id == session_id)
            .order_by(ChatMessageRecord.seq, ChatMessageRecord.timestamp)
        )).scalars().all()
        return {**_summary(record, len(rows)), "messages": [_message_dict(r) for r in rows]}


async def add_message(
    project_path: Optional[str],
    session_id: str,
    role: str,
    content: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Append a message; raises ``LookupError`` when the session is not this project's."""
    async with get_session() as db:
        record = await _owned_session(db, project_path, session_id)
        if record is None:
            raise LookupError("chat session not found")
        top = (await db.execute(
            select(func.max(ChatMessageRecord.seq)).where(ChatMessageRecord.session_id == session_id)
        )).scalar() or 0
        now = datetime.now(timezone.utc)
        message = ChatMessageRecord(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=role if role in ("user", "assistant") else "assistant",
            content=(content or "")[:MAX_MESSAGE_CHARS],
            blocks=blocks or None,
            seq=int(top) + 1,
            timestamp=now,
        )
        db.add(message)
        record.updated_at = now
        await db.commit()
        return _message_dict(message)


async def update_message_blocks(
    project_path: Optional[str], session_id: str, message_id: str, blocks: List[Dict[str, Any]]
) -> bool:
    async with get_session() as db:
        if await _owned_session(db, project_path, session_id) is None:
            return False
        message = await db.get(ChatMessageRecord, message_id)
        if message is None or message.session_id != session_id:
            return False
        message.blocks = list(blocks)
        await db.commit()
        return True


async def get_message(project_path: Optional[str], session_id: str, message_id: str) -> Optional[Dict[str, Any]]:
    async with get_session() as db:
        if await _owned_session(db, project_path, session_id) is None:
            return None
        message = await db.get(ChatMessageRecord, message_id)
        if message is None or message.session_id != session_id:
            return None
        return _message_dict(message)


async def set_title(project_path: Optional[str], session_id: str, title: str) -> None:
    async with get_session() as db:
        record = await _owned_session(db, project_path, session_id)
        if record is not None:
            record.title = title[:200]
            await db.commit()


async def get_title(project_path: Optional[str], session_id: str) -> Optional[str]:
    async with get_session() as db:
        record = await _owned_session(db, project_path, session_id)
        return record.title if record is not None else None


async def delete_session(project_path: Optional[str], session_id: str) -> bool:
    async with get_session() as db:
        record = await _owned_session(db, project_path, session_id)
        if record is None:
            return False
        await db.execute(delete(ChatMessageRecord).where(ChatMessageRecord.session_id == session_id))
        await db.delete(record)
        await db.commit()
        return True


async def history_for_llm(project_path: Optional[str], session_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """The plain-text turns of a session, oldest first, for the language model."""
    data = await get_session_messages(project_path, session_id)
    if not data:
        return []
    turns = [{"role": m["role"], "content": m["content"]} for m in data["messages"] if m["content"]]
    return turns[-limit:]


__all__ = [
    "DEFAULT_TITLE", "generate_title", "create_session", "list_sessions", "get_session_messages",
    "add_message", "update_message_blocks", "get_message", "set_title", "get_title",
    "delete_session", "history_for_llm",
]
