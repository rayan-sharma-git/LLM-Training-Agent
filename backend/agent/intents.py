"""Deterministic intent routing for the chat.

Plain regexes - no model call - decide whether a message is a request the agent
can carry out itself (change a setting, clean data, merge, revert, run a
command ...) or an ordinary question for the language model.  Keeping this
deterministic means destructive-looking requests are never guessed at by a
model, and an ordinary question such as "what are epochs?" never edits a file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agent.settings_editor import CONCEPTS, parse_setting_change
from analysis.project_findings import parse_hardware_statement

CHAT = "chat"
SET_SETTING = "set_setting"
REVERT = "revert"
CLEAN = "clean"
MERGE = "merge"
RESTORE = "restore_cleaning"
DISCARD = "discard_cleaning"
RUN_COMMAND = "run_command"
SHOW_CHANGES = "show_changes"
SHOW_PERMISSIONS = "show_permissions"
ANALYZE = "analyze"
HARDWARE = "hardware"


@dataclass
class Intent:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)


_QUESTION = re.compile(r"^\s*(what|why|how|when|where|which|who|should|is|are|does|do|did|can|could|would|will|"
                       r"explain|tell me|describe)\b", re.I)
_PROGRAMS = ("ls", "dir", "pwd", "git", "pip", "pip3", "python", "python3", "nvidia-smi", "cat", "head", "tail",
             "wc", "npm", "node", "pytest", "df", "free", "nproc", "whoami", "tree", "uname", "conda", "echo",
             "type", "which", "where", "stat", "du", "ps", "env", "make")
_DATA_WORDS = r"(?:data(?:set)?s?|records?|files?|jsonl|csv|samples?|folder|output)"


def _is_question(text: str) -> bool:
    return bool(_QUESTION.match(text)) and text.rstrip().endswith("?")


def _command_from(text: str) -> Optional[str]:
    raw = text.strip()
    if raw.startswith("$ ") or raw.startswith("> "):
        return raw[2:].strip() or None
    m = re.match(r"^(?:please\s+)?(?:run|execute|exec)\s+(?:the\s+)?(?:command\s+|cmd\s+)?`+([^`]+)`+\s*[.!]?$", raw, re.I)
    if m:
        return m.group(1).strip()
    m = re.match(r"^(?:please\s+)?(?:run|execute)\s+(?:the\s+)?(?:command|cmd)\s*:?\s*(.+?)\s*$", raw, re.I)
    if m:
        return m.group(1).strip("`\"' ")
    m = re.match(r"^(?:terminal|shell|cmd|bash)\s*:\s*(.+)$", raw, re.I)
    if m:
        return m.group(1).strip("`\"' ")
    m = re.match(r"^(?:please\s+)?(?:run|execute)\s+((?:%s)\b.*)$" % "|".join(map(re.escape, _PROGRAMS)),
                 raw, re.I)
    if m and not _is_question(raw):
        return m.group(1).strip("`\"' .")
    return None


def classify(text: str) -> Intent:
    """Map a chat message to an :class:`Intent`."""
    raw = (text or "").strip()
    if not raw:
        return Intent(CHAT)
    lowered = raw.lower()

    command = _command_from(raw)
    if command:
        return Intent(RUN_COMMAND, {"command": command})

    question = _is_question(raw)

    # --- cleaning lifecycle (merge / restore / discard before clean) -----------
    if not question:
        if re.search(r"\b(merge|merging)\b", lowered) and re.search(
                r"\b(clean(?:ed|ing)?|temp(?:orary)?|output|processed)\b|\boriginals?\b", lowered):
            return Intent(MERGE)
        if re.search(r"\b(undo|revert|roll ?back|restore)\b", lowered) and re.search(
                r"\b(merge|merged|cleaning|clean(?:ed)?)\b|\boriginal (data|dataset|files?)\b", lowered):
            return Intent(RESTORE)
        if re.search(r"\b(discard|delete|remove|drop)\b", lowered) and re.search(
                r"\bclean(?:ed)?\b.*\b(data|dataset|files?|folder|output|copies)\b", lowered):
            return Intent(DISCARD)
        if re.search(r"\b(clean|clean ?up|sanitize|sanitise|preprocess|de-?duplicate|dedupe)\b", lowered) and \
                re.search(rf"\b{_DATA_WORDS}\b", lowered):
            return Intent(CLEAN)

    # --- changes ------------------------------------------------------------------
    if re.search(r"\b(view|show|see|list|display|open)\b.*\b(changes|diffs?|edits|modifications)\b", lowered) or \
            re.search(r"what (did|have) you (change|edit|modify|do)", lowered) or lowered in ("view changes", "changes"):
        return Intent(SHOW_CHANGES)
    if not question and re.search(r"\b(undo|revert|roll ?back)\b", lowered) and not re.search(
            r"\b(merge|merged|clean(?:ed|ing)?)\b", lowered):
        concept = None
        for c in CONCEPTS:
            if re.search(rf"(?<![\w-])(?:{c.phrase})(?![\w-])", lowered):
                concept = c.key
                break
        file_hint = re.search(r"([\w./\\-]+\.\w{1,6})\b", raw)
        return Intent(REVERT, {"concept": concept, "file": file_hint.group(1).replace("\\", "/") if file_hint else None})

    if re.search(r"\b(permissions?|what (are you|can you) allowed)\b", lowered) and \
            re.search(r"\b(show|what|which|list|my|current)\b", lowered):
        return Intent(SHOW_PERMISSIONS)

    if re.match(r"^(?:please\s+)?(?:re-?)?(?:analy[sz]e|run (?:the |a )?(?:project )?analysis)\b", lowered) and not question:
        return Intent(ANALYZE)

    # --- hardware statement ("I train on a RunPod A100 at $1.89/hr") ----------------
    if not question:
        profile = parse_hardware_statement(raw)
        if profile is not None and not parse_setting_change(raw):
            return Intent(HARDWARE, {"profile": profile})

    change = parse_setting_change(raw)
    if change is not None:
        return Intent(SET_SETTING, {"change": change})

    return Intent(CHAT)


def title_for(intent: Intent, text: str) -> Optional[str]:
    """A specific session title for agent actions (``None`` to fall back to the message)."""
    if intent.kind == SET_SETTING:
        change = intent.data["change"]
        return f"Change {change.concept.label}"
    return {
        CLEAN: "Clean dataset",
        MERGE: "Merge cleaned data",
        RESTORE: "Restore original data",
        DISCARD: "Discard cleaned data",
        REVERT: "Revert change",
        SHOW_CHANGES: "Review changes",
        ANALYZE: "Project analysis",
        HARDWARE: "Hardware setup",
        SHOW_PERMISSIONS: "Permissions",
        RUN_COMMAND: ("Run " + intent.data.get("command", ""))[:50],
    }.get(intent.kind)


__all__ = ["Intent", "classify", "title_for", "CHAT", "SET_SETTING", "REVERT", "CLEAN", "MERGE", "RESTORE",
           "DISCARD", "RUN_COMMAND", "SHOW_CHANGES", "SHOW_PERMISSIONS", "ANALYZE", "HARDWARE"]
