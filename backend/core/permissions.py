"""Project permissions enforced by the agent's execution layer.

Four independent switches decide what the agent may do on its own:

========================  ==================================================
``read_files``            read project files (analysis, search, settings lookup)
``write_files``           change project files (edits, cleaning, merge, revert)
``safe_commands``         run read-only / informational terminal commands
``unrestricted_commands`` run *any* terminal command without asking each time
========================  ==================================================

The policy is stored by the backend (see :mod:`core.permissions_store`) and
every file or command operation goes through :func:`enforce`, so a UI that
forgets to check - or a request that skips the UI - still cannot bypass it.

When an action is not permitted the caller gets a :class:`PermissionRequired`
error carrying a human-readable reason; the chat turns that into an
Allow / Deny card, and an explicit *Allow* authorises that single action.
Catastrophic commands are refused outright (``PermissionBlocked``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

READ = "read"
WRITE = "write"
COMMAND_SAFE = "command_safe"
COMMAND_RISKY = "command_risky"
COMMAND_BLOCKED = "command_blocked"

ACTION_LABELS = {
    READ: "read project files",
    WRITE: "modify project files",
    COMMAND_SAFE: "run a safe terminal command",
    COMMAND_RISKY: "run a terminal command that can change your system",
}


@dataclass(frozen=True)
class PermissionPolicy:
    """The four independent permission switches."""

    read_files: bool = True
    write_files: bool = False
    safe_commands: bool = False
    unrestricted_commands: bool = False

    @classmethod
    def from_dict(cls, data: Any) -> "PermissionPolicy":
        """Build a policy from untrusted input; anything not exactly ``True`` is off."""
        if not isinstance(data, dict):
            return cls()

        def flag(*names: str, default: bool) -> bool:
            for name in names:
                if name in data:
                    return data[name] is True
            return default

        return cls(
            read_files=flag("readFiles", "read_files", default=True),
            write_files=flag("writeFiles", "write_files", default=False),
            safe_commands=flag("safeCommands", "safe_commands", default=False),
            unrestricted_commands=flag("unrestrictedCommands", "unrestricted_commands", default=False),
        )

    def to_dict(self) -> Dict[str, bool]:
        return {
            "readFiles": self.read_files,
            "writeFiles": self.write_files,
            "safeCommands": self.safe_commands,
            "unrestrictedCommands": self.unrestricted_commands,
        }


class PermissionError_(Exception):
    """Base class for permission outcomes (named to avoid shadowing ``PermissionError``)."""

    def __init__(self, action: str, message: str):
        super().__init__(message)
        self.action = action
        self.message = message


class PermissionRequired(PermissionError_):
    """The action is not enabled; the user may approve it once."""


class PermissionBlocked(PermissionError_):
    """The action is never allowed (for example a catastrophic command)."""


def is_permitted(policy: PermissionPolicy, action: str) -> bool:
    """Whether *policy* allows *action* without asking."""
    if action == READ:
        return policy.read_files
    if action == WRITE:
        return policy.write_files
    if action == COMMAND_SAFE:
        return policy.safe_commands or policy.unrestricted_commands
    if action == COMMAND_RISKY:
        return policy.unrestricted_commands
    return False


def enforce(policy: PermissionPolicy, action: str, *, approved: bool = False, detail: str = "") -> None:
    """Raise unless *action* is permitted by *policy* or was explicitly *approved*.

    ``approved`` is only ever set by the pending-action responder after the
    user pressed Allow on the card for this exact action.
    """
    if action == COMMAND_BLOCKED:
        raise PermissionBlocked(action, detail or "This command is blocked because it could destroy data.")
    if is_permitted(policy, action) or approved:
        return
    label = ACTION_LABELS.get(action, action)
    why = {
        READ: "Reading project files is turned off in Settings → Permissions.",
        WRITE: "Modifying project files is turned off in Settings → Permissions.",
        COMMAND_SAFE: "Running safe terminal commands is turned off in Settings → Permissions.",
        COMMAND_RISKY: "This command can change files or your system, and unrestricted "
                       "terminal commands are turned off in Settings → Permissions.",
    }.get(action, "This action is not permitted.")
    raise PermissionRequired(action, f"Permission needed to {label}. {why}" + (f" {detail}" if detail else ""))


__all__ = [
    "READ", "WRITE", "COMMAND_SAFE", "COMMAND_RISKY", "COMMAND_BLOCKED",
    "PermissionPolicy", "PermissionRequired", "PermissionBlocked", "PermissionError_",
    "enforce", "is_permitted",
]
