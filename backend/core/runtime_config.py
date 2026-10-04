"""Runtime configuration store.

The extension sends provider/model/API-key configuration over HTTP to the
local backend. This module stores that state in memory so it can override
the environment-based pydantic settings without writing secrets to disk.

The backend binds to 127.0.0.1 only, so this memory store is only reachable
from the local machine.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

_runtime: Dict[str, Any] = {}
_api_keys: Dict[str, str] = {}


def set_runtime_config(key: str, value: Any) -> None:
    """Set a runtime configuration value (provider, model, base_url, ...)."""
    _runtime[key] = value


def get_runtime_config(key: str, default: Any = None) -> Any:
    """Get a runtime configuration value with fallback to default."""
    return _runtime.get(key, default)


def set_api_key(provider: str, api_key: str) -> None:
    """Securely store an API key in memory for a provider."""
    _api_keys[provider] = api_key


def get_api_key(provider: str) -> Optional[str]:
    """Get the in-memory API key for a provider."""
    return _api_keys.get(provider)


def remove_api_key(provider: str) -> None:
    """Remove the in-memory API key for a provider."""
    _api_keys.pop(provider, None)


def get_active_provider() -> Optional[str]:
    """Get the runtime-selected provider name."""
    return _runtime.get("provider")


def get_active_model() -> Optional[str]:
    """Get the runtime-selected model for the active provider."""
    return _runtime.get("model")


def clear_runtime() -> None:
    """Reset all runtime state (used by tests)."""
    _runtime.clear()
    _api_keys.clear()


def all_api_keys() -> Dict[str, str]:
    """Return a copy of every in-memory API key, keyed by provider name.

    Used by ``core.redaction`` to guarantee no held credential ever reaches
    an LLM prompt, a log line or an error response.
    """
    return dict(_api_keys)