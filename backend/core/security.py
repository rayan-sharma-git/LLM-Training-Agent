"""Security boundaries for untrusted project, file, request, and provider input."""
from __future__ import annotations

import ipaddress
import os
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException

MAX_PATH_CHARS = 4096
MAX_TEXT_CHARS = 100_000
MAX_PROPOSAL_CHARS = 2_000_000
MAX_DATASET_ENTRIES = 500
MAX_DATASET_FILES = 500
MAX_CHUNK_RECORDS = 200
MAX_CHUNK_CHARS = 100_000

#: Environment variable holding an os.pathsep-separated list of authorized roots.
#: Set by the VS Code extension from the folders that are open *right now*.
ALLOWED_ROOTS_ENV = "LLM_TRAINING_AGENT_ALLOWED_ROOTS"

#: Optional file the extension rewrites whenever the open workspace changes.
#: Lets a single long-lived backend process follow a Project A -> Project B
#: switch without being restarted, while still refusing any other directory.
ALLOWED_ROOTS_FILE_ENV = "LLM_TRAINING_AGENT_ALLOWED_ROOTS_FILE"

_roots_lock = threading.Lock()
_roots_cache: tuple[int, list[Path]] | None = None


def _split_roots(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(os.pathsep) if x.strip()]


def _roots_from_env()->list[Path]|None:
    """Read the authorized roots.

    Precedence:
      1. The roots file, when configured. The extension rewrites it whenever the
         open workspace changes, so this MUST win over the startup environment:
         otherwise a Project A -> Project B switch would never take effect.
      2. The environment variable, as a fallback for a backend started without
         the extension.
      3. ``None`` when neither is configured, so the caller can fall back to the
         process working directory (standalone ``uvicorn`` usage).

    An **empty** result is returned when the extension is running but has no
    folder open, which denies every project rather than silently allowing the
    backend's own directory.
    """
    roots_file = os.environ.get(ALLOWED_ROOTS_FILE_ENV)
    if roots_file:
        try:
            contents = Path(roots_file).read_text(encoding="utf-8")
        except OSError:
            # The extension has not written the file yet: deny rather than guess.
            return []
        return [Path(x).expanduser().resolve() for x in _split_roots(contents)]

    from_env = os.environ.get(ALLOWED_ROOTS_ENV)
    if from_env is not None:
        return [Path(x).expanduser().resolve() for x in _split_roots(from_env)]

    return None


def allowed_roots() -> list[Path]:
    """Directories this backend is permitted to read from and write to.

    The value is re-read (with a short cache) because the set of authorized
    roots changes when the user switches projects in VS Code. Caching only the
    parsed result, never across a roots-file change, keeps Project A and
    Project B strictly isolated.
    """
    global _roots_cache
    roots_file = os.environ.get(ALLOWED_ROOTS_FILE_ENV)
    if not roots_file:
        # Not running under the extension: standalone `uvicorn` usage.
        return _roots_from_env() or [Path.cwd().resolve()]

    try:
        # st_mtime_ns: the coarse st_mtime can be identical for two writes made
        # in quick succession, which would hide a fast Project A -> Project B
        # switch from an already-running backend.
        mtime = Path(roots_file).stat().st_mtime_ns
    except OSError:
        # A roots file is configured but unreadable. Deny rather than fall back
        # to the backend's own directory.
        return _roots_from_env()

    with _roots_lock:
        if _roots_cache is not None and _roots_cache[0] == mtime:
            return list(_roots_cache[1])
        roots = _roots_from_env() or []
        _roots_cache = (mtime, list(roots))
        return list(roots)


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
def resolve_workspace_root(value: str, *, must_exist: bool = True) -> Path:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > MAX_PATH_CHARS
    ):
        raise HTTPException(
            400, detail={"errorCode": "INVALID_PATH", "message": "Invalid project path."}
        )
    candidate = Path(value).expanduser().resolve()
    if must_exist and not candidate.is_dir():
        raise HTTPException(
            404,
            detail={
                "errorCode": "PROJECT_NOT_FOUND",
                "message": "Project directory was not found.",
            },
        )
    if not any(_within(candidate, root) for root in allowed_roots()):
        raise HTTPException(
            403,
            detail={
                "errorCode": "WORKSPACE_FORBIDDEN",
                "message": "The requested directory is outside the authorized workspace.",
            },
        )
    return candidate


def resolve_workspace_child(
    root_value: str, child: str, *, must_exist: bool = False
) -> Path:
    root = resolve_workspace_root(root_value)
    if (
        not isinstance(child, str)
        or not child
        or "\x00" in child
        or len(child) > MAX_PATH_CHARS
    ):
        raise HTTPException(
            400, detail={"errorCode": "INVALID_PATH", "message": "Invalid file path."}
        )
    raw = Path(child)
    if raw.is_absolute() or raw.drive or raw.root:
        raise HTTPException(
            400,
            detail={
                "errorCode": "INVALID_PATH",
                "message": "File paths must be workspace-relative.",
            },
        )
    candidate = (root / raw).resolve()
    if not _within(candidate, root):
        raise HTTPException(
            400,
            detail={
                "errorCode": "PATH_TRAVERSAL",
                "message": "File path escapes the workspace.",
            },
        )
    if must_exist and not candidate.is_file():
        raise HTTPException(
            404,
            detail={"errorCode": "FILE_NOT_FOUND", "message": "The requested file was not found."},
        )
    return candidate
def resolve_dataset_path(project_root: str, dataset_path: str) -> Path:
    root = resolve_workspace_root(project_root)
    if (
        not isinstance(dataset_path, str)
        or not dataset_path
        or "\x00" in dataset_path
        or len(dataset_path) > MAX_PATH_CHARS
    ):
        raise ValueError("Invalid dataset path")
    raw = Path(dataset_path)
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not _within(candidate, root):
        raise ValueError("Dataset path escapes the project directory")
    return candidate


def resolve_output_dir(
    project_root: str, output_dir: Optional[str], default_name: str
) -> Path:
    root = resolve_workspace_root(project_root)
    value = output_dir or default_name
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > MAX_PATH_CHARS
    ):
        raise ValueError("Invalid output directory")
    raw = Path(value)
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not _within(candidate, root) or candidate == root:
        raise ValueError("Output directory must remain inside the project")
    return candidate


def bounded_int(
    value, *, name: str, minimum: int, maximum: int, default: int
) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            422,
            detail={"errorCode": "INVALID_REQUEST", "message": f"{name} must be an integer."},
        ) from exc
    if not minimum <= parsed <= maximum:
        raise HTTPException(
            422,
            detail={
                "errorCode": "INVALID_REQUEST",
                "message": f"{name} is outside the allowed range.",
            },
        )
    return parsed


def bounded_text(
    value, *, name: str, maximum: int = MAX_TEXT_CHARS, required: bool = True
) -> Optional[str]:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or len(value) > maximum
        or "\x00" in value
        or (required and not value.strip())
    ):
        raise HTTPException(
            422,
            detail={
                "errorCode": "INVALID_REQUEST",
                "message": f"{name} is invalid or too large.",
            },
        )
    return value.strip() if required else value
def validate_provider_url(value: str, *, allow_remote: bool = True) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > 2048
    ):
        raise ValueError("Provider URL is invalid")
    parsed = urlparse(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError(
            "Provider URL must be an HTTP(S) URL without credentials or fragments"
        )
    host = parsed.hostname.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(host)
        is_local = address.is_loopback
        if (
            address.is_link_local
            or address.is_unspecified
            or address.is_multicast
            or address.is_reserved
        ):
            raise ValueError("Provider URL points to a disallowed address")
    except ValueError as exc:
        # Re-raise our own rejections; fall through for a plain DNS hostname.
        if exc.args and str(exc.args[0]).startswith("Provider URL"):
            raise
        is_local = host in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not is_local:
        raise ValueError("Remote provider URLs must use HTTPS")
    if not allow_remote and not is_local:
        raise ValueError("Only local provider URLs are allowed")
    return value.strip().rstrip("/")


__all__ = [
    "resolve_workspace_root",
    "resolve_workspace_child",
    "resolve_dataset_path",
    "resolve_output_dir",
    "bounded_int",
    "bounded_text",
    "validate_provider_url",
    "allowed_roots",
]
