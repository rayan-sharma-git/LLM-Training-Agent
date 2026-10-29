"""Classify and run terminal commands under the permission policy.

Three tiers:

``safe``
    Read-only / informational commands from a fixed allow-list (``ls``,
    ``git status``, ``nvidia-smi``, ``pip list`` ...), with no shell
    metacharacters and no path leaving the project.  They run with
    ``shell=False``.
``risky``
    Everything else that is not clearly catastrophic: installs, running
    scripts, deleting or moving files, network tools, ``git commit/push``,
    pipes and chains.  Needs the *unrestricted* permission or a per-command
    approval.
``blocked``
    Commands whose only plausible effect is destroying the machine or a whole
    home directory (``rm -rf /``, ``mkfs``, fork bombs, piping a download into
    a shell ...).  Never run, regardless of permissions.

Every run uses the project as working directory, a timeout, an output cap and
an environment with secrets removed; the output is redacted before it is
returned.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.permissions import (
    COMMAND_BLOCKED,
    COMMAND_RISKY,
    COMMAND_SAFE,
    PermissionPolicy,
    enforce,
)
from core.redaction import scrub

MAX_COMMAND_CHARS = 2000
MAX_OUTPUT_CHARS = 20_000
SAFE_TIMEOUT_SECONDS = 30
RISKY_TIMEOUT_SECONDS = 600

_META = re.compile(r"[;&|<>`$\n\r]|\$\(|\(\)")
_SECRET_FILE = re.compile(
    r"(^|[\\/])(\.env(\..*)?|.*\.(pem|key|p12|pfx|crt)|id_(rsa|dsa|ecdsa|ed25519)|"
    r"credentials(\.json)?|\.netrc|\.npmrc|\.pypirc|secrets?\.(json|ya?ml|toml))$", re.I)
_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION|PRIVATE)", re.I)

#: Programs that only read or report.  ``None`` = any arguments are fine,
#: a set = only those sub-commands, a callable = custom argument check.
_SAFE_PROGRAMS: Dict[str, Any] = {
    "ls": None, "dir": None, "pwd": None, "whoami": None, "hostname": None, "date": None,
    "uname": None, "nproc": None, "df": None, "free": None, "uptime": None, "which": None,
    "where": None, "wc": None, "head": None, "tail": None, "cat": None, "type": None,
    "nvidia-smi": None, "tree": None, "stat": None, "file": None, "echo": None,
}
_SAFE_SUBCOMMANDS: Dict[str, set] = {
    "git": {"status", "log", "diff", "show", "branch", "rev-parse", "remote", "tag", "ls-files",
            "blame", "describe", "shortlog"},
    "pip": {"list", "show", "freeze", "check", "--version", "-V"},
    "pip3": {"list", "show", "freeze", "check", "--version", "-V"},
    "python": {"--version", "-V"},
    "python3": {"--version", "-V"},
    "node": {"--version", "-v"},
    "npm": {"--version", "-v", "ls", "list"},
    "conda": {"list", "info", "--version"},
}
#: Flags that turn an otherwise read-only program into a writing one.
_UNSAFE_FLAGS = {
    "git": {"--output", "-o", "--ext-diff", "--textconv", "-c", "--exec", "-d", "-D", "--delete",
            "-m", "-M", "--move", "-f", "--force", "--set-upstream", "-u"},
    "pip": {"--target", "-t", "--user"},
}

_BLOCKED_PATTERNS = [
    (re.compile(r"\brm\s+(-[a-z]*\s+)*-[a-z]*[rf][a-z]*\s+(-[a-z]*\s+)*(--no-preserve-root\s+)?(/|~|\$home|/\*|~/\*|\*)(\s|$)", re.I),
     "recursively deleting the root or home directory"),
    (re.compile(r"\brm\s+.*--no-preserve-root", re.I), "deleting the root directory"),
    (re.compile(r"\bmkfs(\.\w+)?\b", re.I), "formatting a disk"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/(sd|hd|nvme|disk|mmcblk)", re.I), "overwriting a disk device"),
    (re.compile(r">\s*/dev/(sd|hd|nvme|disk|mmcblk)", re.I), "overwriting a disk device"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "a fork bomb"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.I), "shutting the machine down"),
    (re.compile(r"\bformat\s+[a-z]:", re.I), "formatting a drive"),
    (re.compile(r"\b(del|rmdir|rd)\b[^\n]*\s/s\b[^\n]*\b[a-z]:\\?(\s|$)", re.I), "deleting a whole drive"),
    (re.compile(r"\bchmod\s+(-[a-z]+\s+)*[0-7]{3,4}\s+/(\s|$)", re.I), "changing permissions on the root directory"),
    (re.compile(r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", re.I), "running a downloaded script without review"),
    (re.compile(r"\bsudo\s+rm\b", re.I), "deleting files as root"),
]


@dataclass
class Classification:
    tier: str                    # "safe" | "risky" | "blocked"
    action: str                  # permission action constant
    reason: str
    argv: Optional[List[str]] = None  # parsed argv for the safe tier

    def to_dict(self) -> Dict[str, Any]:
        return {"tier": self.tier, "reason": self.reason}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def classify_command(command: str, project_root: Path) -> Classification:
    """Decide the tier of *command* for a project rooted at *project_root*."""
    text = (command or "").strip()
    if not text:
        return Classification("risky", COMMAND_RISKY, "empty command")
    if len(text) > MAX_COMMAND_CHARS or "\x00" in text:
        return Classification("blocked", COMMAND_BLOCKED, "the command is too long or malformed")

    for pattern, why in _BLOCKED_PATTERNS:
        if pattern.search(text):
            return Classification("blocked", COMMAND_BLOCKED, f"it would risk {why}")

    if _META.search(text):
        return Classification("risky", COMMAND_RISKY,
                              "it chains commands, redirects output or uses shell expansion")
    try:
        argv = shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        return Classification("risky", COMMAND_RISKY, "it could not be parsed safely")
    if not argv:
        return Classification("risky", COMMAND_RISKY, "empty command")

    program = Path(argv[0]).name.lower()
    if program.endswith(".exe"):
        program = program[:-4]
    args = argv[1:]

    if program in _SAFE_SUBCOMMANDS:
        sub = next((a for a in args if not a.startswith("-") or a in _SAFE_SUBCOMMANDS[program]), None)
        flags = {a for a in args if a.startswith("-")}
        if sub is None or sub not in _SAFE_SUBCOMMANDS[program]:
            shown = f"{program} {sub}" if sub else program
            return Classification("risky", COMMAND_RISKY, f"'{shown}' can change files or run code")
        if {f.split("=", 1)[0] for f in flags} & _UNSAFE_FLAGS.get(program, set()):
            return Classification("risky", COMMAND_RISKY, "it uses an option that can change files")
        if program == "git" and sub == "branch" and any(not a.startswith("-") for a in args[1:]):
            return Classification("risky", COMMAND_RISKY, "it would create or change a branch")
        if program == "git" and sub in ("remote", "tag") and any(not a.startswith("-") for a in args[1:]):
            return Classification("risky", COMMAND_RISKY, "it would change repository settings")
    elif program not in _SAFE_PROGRAMS:
        return Classification("risky", COMMAND_RISKY, f"'{program}' is not on the safe list")

    # Paths must stay inside the project and must not point at secret files.
    for arg in args:
        if arg.startswith("-") and not _looks_like_path(arg):
            continue
        if _SECRET_FILE.search(arg):
            return Classification("risky", COMMAND_RISKY, "it reads a file that may hold secrets")
        if _looks_like_path(arg) and not _inside((project_root / arg) if not os.path.isabs(arg) else Path(arg), project_root):
            return Classification("risky", COMMAND_RISKY, "it touches a path outside the project")
    return Classification("safe", COMMAND_SAFE, "read-only command on the allow-list", argv=argv)


def _looks_like_path(arg: str) -> bool:
    return arg.startswith(("/", "~", "\\")) or ".." in arg.split("/") or ".." in arg.split("\\") or bool(re.match(r"^[a-zA-Z]:[\\/]", arg))


def _clean_env() -> Dict[str, str]:
    return {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}


def _cap(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    return text[:MAX_OUTPUT_CHARS] + "\n… output truncated", True


def run_command(
    command: str,
    project_root: Path,
    policy: PermissionPolicy,
    *,
    approved: bool = False,
) -> Dict[str, Any]:
    """Run *command* if (and only if) the permission layer allows it.

    Raises :class:`core.permissions.PermissionRequired` /
    :class:`core.permissions.PermissionBlocked`; never runs first and asks later.
    """
    root = Path(project_root)
    classification = classify_command(command, root)
    enforce(policy, classification.action, approved=approved,
            detail=f"({classification.reason}.)" if classification.tier != "safe" else "")

    timeout = SAFE_TIMEOUT_SECONDS if classification.tier == "safe" else RISKY_TIMEOUT_SECONDS
    started = time.monotonic()
    try:
        if classification.tier == "safe" and classification.argv:
            argv = classification.argv
            if os.name == "nt" and Path(argv[0]).name.lower() in ("dir", "type", "echo", "where"):
                argv = ["cmd", "/c", *argv]
            completed = subprocess.run(
                argv, cwd=str(root), env=_clean_env(), capture_output=True, text=True,
                timeout=timeout, shell=False, errors="replace",
            )
        else:
            completed = subprocess.run(
                command, cwd=str(root), env=_clean_env(), capture_output=True, text=True,
                timeout=timeout, shell=True, errors="replace",
            )
        output = (completed.stdout or "") + (completed.stderr or "")
        exit_code, timed_out = completed.returncode, False
    except subprocess.TimeoutExpired as expired:
        out = expired.stdout or b""
        err = expired.stderr or b""
        output = (out.decode("utf-8", "replace") if isinstance(out, bytes) else out) + \
                 (err.decode("utf-8", "replace") if isinstance(err, bytes) else err)
        exit_code, timed_out = None, True
    except FileNotFoundError:
        output, exit_code, timed_out = f"Command not found: {shlex.split(command)[0] if command.split() else command}", 127, False
    except OSError as error:
        output, exit_code, timed_out = f"Could not start the command: {error.strerror or error}", 126, False

    output, truncated = _cap(scrub(output))
    return {
        "command": scrub(command),
        "tier": classification.tier,
        "exitCode": exit_code,
        "timedOut": timed_out,
        "output": output,
        "truncated": truncated,
        "durationMs": int((time.monotonic() - started) * 1000),
        "cwd": str(root),
    }


__all__ = ["classify_command", "run_command", "Classification"]
