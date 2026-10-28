"""Routes for the chat agent: sessions, messages, answers to cards, permissions."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from agent import service
from chat import sessions
from core.permissions import PermissionPolicy
from core.permissions_store import load_policy, save_policy
from core.redaction import scrub_error
from api import permission_guard

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_PATH_CHARS = 4096


def _project(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > MAX_PATH_CHARS or "\x00" in value:
        raise HTTPException(status_code=422, detail={"errorCode": "INVALID_PROJECT", "message": "projectPath is invalid."})
    return value.strip()


def _session_id(value: Any, required: bool = False) -> Optional[str]:
    if value in (None, ""):
        if required:
            raise HTTPException(status_code=422, detail={"errorCode": "INVALID_SESSION", "message": "sessionId is required."})
        return None
    if not isinstance(value, str) or not all(ch.isalnum() or ch == "-" for ch in value) or len(value) > 64:
        raise HTTPException(status_code=422, detail={"errorCode": "INVALID_SESSION", "message": "sessionId is invalid."})
    return value


def _fail(error: Exception, code: str, message: str) -> HTTPException:
    logger.error("%s: %s", code, scrub_error(error), exc_info=True)
    return HTTPException(status_code=500, detail={"errorCode": code, "message": message})


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------
@router.get("/chat/sessions")
async def list_chat_sessions(projectPath: Optional[str] = None):
    """Previous chats of a project, newest activity first (title, date and time)."""
    try:
        return {"sessions": await sessions.list_sessions(_project(projectPath))}
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "LIST_SESSIONS_FAILED", "The chat list could not be loaded.")


@router.post("/chat/sessions")
async def create_chat_session(request: Dict[str, Any]):
    try:
        return await sessions.create_session(_project(request.get("projectPath")))
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "CREATE_SESSION_FAILED", "The chat could not be created.")


@router.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str, projectPath: Optional[str] = None):
    """A full conversation, including change viewers and answered cards."""
    try:
        data = await sessions.get_session_messages(_project(projectPath), _session_id(session_id, True))
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "GET_SESSION_FAILED", "The chat could not be loaded.")
    if data is None:
        raise HTTPException(status_code=404, detail={"errorCode": "SESSION_NOT_FOUND", "message": "That chat was not found."})
    return data


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str, projectPath: Optional[str] = None):
    try:
        removed = await sessions.delete_session(_project(projectPath), _session_id(session_id, True))
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "DELETE_SESSION_FAILED", "The chat could not be deleted.")
    if not removed:
        raise HTTPException(status_code=404, detail={"errorCode": "SESSION_NOT_FOUND", "message": "That chat was not found."})
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Agent chat
# ---------------------------------------------------------------------------
@router.post("/agent/chat")
async def agent_chat(request: Dict[str, Any]):
    """Send a message. Without ``sessionId`` a new session is created automatically."""
    message = request.get("message")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=422, detail={"errorCode": "EMPTY_MESSAGE", "message": "message must not be empty."})
    try:
        return await service.handle_message(
            _project(request.get("projectPath")), _session_id(request.get("sessionId")), message,
            provider=request.get("provider") if isinstance(request.get("provider"), str) else None)
    except LookupError:
        raise HTTPException(status_code=404, detail={"errorCode": "SESSION_NOT_FOUND", "message": "That chat was not found."})
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "AGENT_CHAT_FAILED", "The message could not be processed.")


@router.post("/agent/actions/{action_id}/respond")
async def agent_respond(action_id: str, request: Dict[str, Any]):
    """Answer a card (Allow / Deny, cleaning options, Revert ...)."""
    choice = request.get("choice")
    if not isinstance(choice, str) or not choice.strip():
        raise HTTPException(status_code=422, detail={"errorCode": "INVALID_CHOICE", "message": "choice is required."})
    try:
        return await service.respond_to_action(
            _project(request.get("projectPath")), _session_id(request.get("sessionId"), True) or "",
            action_id, choice)
    except LookupError as error:
        raise HTTPException(status_code=409, detail={"errorCode": "ACTION_UNAVAILABLE", "message": str(error)})
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "AGENT_ACTION_FAILED", "That answer could not be processed.")


@router.post("/agent/changes/show")
async def agent_show_changes(request: Dict[str, Any]):
    """Post the latest changes into a chat as an inline red/green viewer."""
    try:
        return await service.show_changes(_project(request.get("projectPath")), _session_id(request.get("sessionId")))
    except LookupError:
        raise HTTPException(status_code=404, detail={"errorCode": "SESSION_NOT_FOUND", "message": "That chat was not found."})
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "SHOW_CHANGES_FAILED", "The changes could not be shown.")


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------
@router.get("/agent/permissions")
async def get_permissions(projectPath: Optional[str] = None):
    return (await load_policy(_project(projectPath))).to_dict()


@router.put("/agent/permissions")
async def put_permissions(request: Dict[str, Any]):
    """Store the user's Settings → Permissions for a project (anything but ``true`` is off)."""
    policy = PermissionPolicy.from_dict(request.get("permissions"))
    try:
        await save_policy(_project(request.get("projectPath")), policy)
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "SAVE_PERMISSIONS_FAILED", "The permissions could not be saved.")
    return policy.to_dict()


@router.post("/agent/permissions/allow-once")
async def allow_once(request: Dict[str, Any]):
    """Allow the next read or write for a project after the user pressed Allow on a card."""
    action = request.get("action")
    try:
        permission_guard.grant_once(_project(request.get("projectPath")), str(action))
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"errorCode": "INVALID_ACTION", "message": str(error)})
    return {"status": "granted", "action": action, "scope": "next request only"}


@router.post("/agent/chat/note")
async def agent_note(request: Dict[str, Any]):
    """Store a message the extension produced itself (for example a project analysis).

    Creates the chat on first use, so results of buttons and commands appear in
    the persisted history exactly like answers from the agent.
    """
    content = request.get("content")
    role = request.get("role", "assistant")
    if role not in ("assistant", "user") or not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=422, detail={"errorCode": "INVALID_NOTE", "message": "role and content are required."})
    project = _project(request.get("projectPath"))
    raw_blocks = request.get("blocks")
    # Only passive block types may come from the client; cards that can trigger work stay server-made.
    blocks = [b for b in raw_blocks[:20] if isinstance(b, dict) and b.get("type") in ("commandButtons", "command")] \
        if isinstance(raw_blocks, list) else []
    blocks = blocks or None
    try:
        session_id = _session_id(request.get("sessionId"))
        if session_id and await sessions.get_session_messages(project, session_id) is None:
            raise LookupError("chat session not found")
        if not session_id:
            session_id = (await sessions.create_session(project))["sessionId"]
        message = await sessions.add_message(project, session_id, role, content[:sessions.MAX_MESSAGE_CHARS], blocks)
        title = request.get("title")
        if isinstance(title, str) and title.strip() and not await sessions.get_title(project, session_id):
            await sessions.set_title(project, session_id, title.strip()[: sessions.MAX_TITLE_CHARS])
        data = await sessions.get_session_messages(project, session_id)
        return {"sessionId": session_id, "title": data["title"] if data else sessions.DEFAULT_TITLE, "message": message}
    except LookupError:
        raise HTTPException(status_code=404, detail={"errorCode": "SESSION_NOT_FOUND", "message": "That chat was not found."})
    except HTTPException:
        raise
    except Exception as error:
        raise _fail(error, "NOTE_FAILED", "The message could not be stored.")
