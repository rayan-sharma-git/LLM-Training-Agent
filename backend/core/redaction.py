"""Secret redaction helpers.

Secrets must never reach an LLM prompt, a log line, an API response or a
report.  This module is the single place that removes them.

Two complementary strategies are applied:

1. *Known values* — every credential the backend holds (runtime API keys sent
   by the VS Code extension plus environment variables whose name looks like a
   credential) is replaced literally.  This catches secrets that would
   otherwise appear inside URLs, exception messages or file contents.
2. *Credential-shaped strings* — common token formats (``sk-...``, ``ghp_``,
   ``AIza...``, ``Bearer ...``, ``API_KEY=...``) are replaced even when the
   value is unknown to the backend.

Redaction is a safety net, not a proof: it reduces the chance of leaking a
credential, it cannot guarantee that arbitrary secret formats are detected.
Prompt construction therefore also treats all project content as untrusted
(see ``chat/engine.py``).
"""
from __future__ import annotations

import os
import re
from typing import Any, Set

REDACTED = "[REDACTED]"

# Maximum length of a scrubbed error string returned to clients.
MAX_ERROR_CHARS = 300

# Environment variable names containing these markers are treated as secrets.
_SECRET_ENV_MARKERS = (
    "API_KEY",
    "APIKEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTHORIZATION",
    "ACCESS_KEY",
)

# Credential-shaped patterns that must be removed even when unknown.
_VALUE_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{8,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
]

# "api_key = value" / "TOKEN: value" style assignments.
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|apikey|token|secret|password|passwd|credential|authorization)"
    r"\b(\s*[:=]\s*)([\"']?)([^\s\"',;]{6,})\3"
)


def _known_secrets() -> Set[str]:
    """Collect every credential value the backend currently holds."""
    secrets: Set[str] = set()

    try:
        from core import runtime_config

        for value in runtime_config.all_api_keys().values():
            if value:
                secrets.add(value)
    except Exception:  # pragma: no cover - defensive, never block redaction
        pass

    for name, value in os.environ.items():
        if not value:
            continue
        upper = name.upper()
        if any(marker in upper for marker in _SECRET_ENV_MARKERS):
            secrets.add(value)

    # Ignore trivially short values to avoid replacing ordinary words.
    return {value for value in secrets if len(value) >= 6}


def scrub(text: str) -> str:
    """Return *text* with known and credential-shaped secrets removed."""
    if not text:
        return text

    scrubbed = text
    for secret in _known_secrets():
        if secret in scrubbed:
            scrubbed = scrubbed.replace(secret, REDACTED)

    for pattern in _VALUE_PATTERNS:
        scrubbed = pattern.sub(REDACTED, scrubbed)

    scrubbed = _ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", scrubbed
    )
    return scrubbed


def scrub_error(error: BaseException) -> str:
    """Return a scrubbed, bounded description of *error* for clients/logs.

    The exception type is kept (it is useful for support) while any credential
    and any oversized payload is removed.
    """
    detail = scrub(str(error)).strip()
    if len(detail) > MAX_ERROR_CHARS:
        detail = detail[:MAX_ERROR_CHARS].rstrip() + "…"
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def scrub_obj(value: Any, depth: int = 0) -> Any:
    """Recursively scrub strings inside dicts/lists/tuples.

    ``depth`` guards against pathological nesting; deeper values are dropped.
    """
    if depth > 12:
        return None
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {str(key): scrub_obj(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_obj(item, depth + 1) for item in value]
    return value


def contains_known_secret(text: str) -> bool:
    """Return True when *text* contains a credential the backend holds."""
    if not text:
        return False
    return any(secret in text for secret in _known_secrets())


__all__ = [
    "REDACTED",
    "MAX_ERROR_CHARS",
    "scrub",
    "scrub_error",
    "scrub_obj",
    "contains_known_secret",
]
