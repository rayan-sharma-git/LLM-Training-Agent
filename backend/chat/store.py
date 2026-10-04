"""Process-scoped chat state: latest analysis snapshot + conversation history.

Why in memory: the REST layer has no wired persistence (the SQLAlchemy models
in ``storage/`` are not used by any route), so chat state lives in this module
instead of introducing a database dependency into the chat path.  State is
scoped to guarantee isolation:

* analysis snapshots are keyed by the *normalized* project path
* conversations are keyed by ``(project path, session id)``

so a conversation started in project A can never be answered with project B's
analysis results or history.  Only analyzer/report output and chat messages are
stored — never credentials.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits (bounded memory, bounded prompts)
# ---------------------------------------------------------------------------

#: Maximum number of projects whose analysis snapshot is retained (LRU).
MAX_PROJECTS = 20
#: Maximum number of concurrent chat sessions retained per project.
MAX_SESSIONS_PER_PROJECT = 8
#: Maximum number of messages retained per session.
MAX_MESSAGES_PER_SESSION = 40
#: Maximum number of characters retained per session (older messages dropped).
MAX_SESSION_CHARS = 24000

_DEFAULT_SCOPE = "_no_project"


def normalize_project_path(project_path: Optional[str]) -> str:
    """Return a stable key for *project_path* (``_no_project`` when absent).

    Using the absolute, case-normalized path means ``C:\\Proj`` and
    ``c:\\proj`` share one snapshot, while different projects never collide.
    """
    if not project_path or not str(project_path).strip():
        return _DEFAULT_SCOPE
    try:
        return os.path.normcase(os.path.abspath(str(project_path).strip()))
    except (OSError, ValueError):  # pragma: no cover - defensive
        return str(project_path).strip()


def normalize_session_id(session_id: Optional[str]) -> str:
    """Return a filesystem-safe session id."""
    value = (session_id or "default").strip()
    safe = "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_"))
    return safe[:64] or "default"


def _history_chars(history: List[Dict[str, str]]) -> int:
    return sum(len(message.get("content", "")) for message in history)


class ChatStore:
    """In-memory analysis snapshots and conversation histories."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._analyses: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._sessions: Dict[str, "OrderedDict[str, List[Dict[str, str]]]"] = {}

    # ------------------------------------------------------------------
    # Analysis snapshots
    # ------------------------------------------------------------------
    def save_analysis(self, project_path: Optional[str], snapshot: Dict[str, Any]) -> str:
        """Store the latest analysis snapshot for a project; returns the key."""
        key = normalize_project_path(project_path)
        record = dict(snapshot)
        record["projectPath"] = str(project_path or "")
        record["analysisTimestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._analyses[key] = record
            self._analyses.move_to_end(key)
            while len(self._analyses) > MAX_PROJECTS:
                evicted, _ = self._analyses.popitem(last=False)
                self._sessions.pop(evicted, None)
        return key

    def get_analysis(self, project_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the stored analysis snapshot for a project (or None)."""
        key = normalize_project_path(project_path)
        with self._lock:
            snapshot = self._analyses.get(key)
            if snapshot is not None:
                self._analyses.move_to_end(key)
            return dict(snapshot) if snapshot else None

    def has_analysis(self, project_path: Optional[str]) -> bool:
        """Return True when an analysis snapshot exists for the project."""
        return self.get_analysis(project_path) is not None

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------
    def get_history(
        self, project_path: Optional[str], session_id: Optional[str]
    ) -> List[Dict[str, str]]:
        """Return a copy of the stored conversation for a project session."""
        key = self._session_key(project_path, session_id)
        scope = normalize_project_path(project_path)
        with self._lock:
            sessions = self._sessions.get(scope)
            if not sessions:
                return []
            return [dict(message) for message in sessions.get(key, [])]

    def append_message(
        self,
        project_path: Optional[str],
        session_id: Optional[str],
        role: str,
        content: str,
    ) -> None:
        """Append one message to a project session, enforcing the caps."""
        key = self._session_key(project_path, session_id)
        scope = normalize_project_path(project_path)
        message = {"role": role, "content": content}
        with self._lock:
            sessions = self._sessions.setdefault(scope, OrderedDict())
            history = sessions.setdefault(key, [])
            history.append(message)
            sessions.move_to_end(key)

            # Cap by message count and total size (drop the oldest turns).
            while len(history) > MAX_MESSAGES_PER_SESSION:
                history.pop(0)
            while len(history) > 1 and _history_chars(history) > MAX_SESSION_CHARS:
                history.pop(0)

            while len(sessions) > MAX_SESSIONS_PER_PROJECT:
                sessions.popitem(last=False)

    def clear_history(self, project_path: Optional[str], session_id: Optional[str]) -> int:
        """Delete one session; returns the number of removed messages."""
        key = self._session_key(project_path, session_id)
        scope = normalize_project_path(project_path)
        with self._lock:
            sessions = self._sessions.get(scope)
            if not sessions:
                return 0
            removed = sessions.pop(key, [])
            if not sessions:
                self._sessions.pop(scope, None)
            return len(removed)

    def reset(self) -> None:
        """Drop all state (used by tests)."""
        with self._lock:
            self._analyses.clear()
            self._sessions.clear()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _session_key(self, project_path: Optional[str], session_id: Optional[str]) -> str:
        return f"{normalize_project_path(project_path)}::{normalize_session_id(session_id)}"


store = ChatStore()


def _dump(value: Any) -> Any:
    """Convert pydantic models / dataclasses / dicts to plain JSON data."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


def build_snapshot(
    context: Any,
    report: Any = None,
    recommendations: Optional[List[Any]] = None,
    *,
    analyzer_results: Optional[Dict[str, Any]] = None,
    prediction: Any = None,
    cost: Any = None,
    gpu_time_estimate: Any = None,
    hardware_detection: Any = None,
) -> Dict[str, Any]:
    """Build an analysis snapshot from real pipeline outputs.

    Only real pipeline results are accepted; missing inputs stay ``None`` so the
    chat prompt can state "not available" instead of inventing values.
    """
    return {
        "context": _dump(context),
        "report": _dump(report) if report is not None else None,
        "recommendations": [_dump(item) for item in (recommendations or [])],
        "analyzers": {
            name: _dump(result) for name, result in (analyzer_results or {}).items()
        },
        "prediction": _dump(prediction) if prediction is not None else None,
        "cost": _dump(cost) if cost is not None else None,
        "gpuTimeEstimate": _dump(gpu_time_estimate) if gpu_time_estimate is not None else None,
        "hardwareDetection": _dump(hardware_detection) if hardware_detection is not None else None,
    }


def save_pipeline_snapshot(
    project_path: Optional[str],
    *,
    context: Any,
    report: Any = None,
    recommendations: Optional[List[Any]] = None,
    analyzer_results: Optional[Dict[str, Any]] = None,
    prediction: Any = None,
    cost: Any = None,
    gpu_time_estimate: Any = None,
    hardware_detection: Any = None,
) -> str:
    """Persist the latest pipeline outputs as the project's chat snapshot.

    Shared by the REST ``/project/analyze`` route and the WebSocket analysis
    pipeline so both feed the chat engine identically (no duplicated wiring).
    Cleaning summaries are excluded — they are large and not useful for chat.
    Never raises: chat context is best-effort and must not fail an analysis.
    """
    try:
        filtered = {
            name: result
            for name, result in (analyzer_results or {}).items()
            if name != "dataset_cleaning"
        }
        return store.save_analysis(
            project_path,
            build_snapshot(
                context,
                report,
                recommendations,
                analyzer_results=filtered,
                prediction=prediction,
                cost=cost,
                gpu_time_estimate=gpu_time_estimate,
                hardware_detection=hardware_detection,
            ),
        )
    except Exception as error:  # pragma: no cover - defensive
        logger.warning(f"Failed to persist chat analysis snapshot: {error}")
        return ""


__all__ = [
    "store",
    "build_snapshot",
    "save_pipeline_snapshot",
    "normalize_project_path",
    "normalize_session_id",
    "MAX_MESSAGES_PER_SESSION",
    "MAX_SESSION_CHARS",
    "MAX_PROJECTS",
    "MAX_SESSIONS_PER_PROJECT",
]

