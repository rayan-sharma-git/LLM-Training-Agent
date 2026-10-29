"""Structured, Cline-style change blocks for the inline chat diff viewer.

A flat unified diff is awkward to render in a narrow chat sidebar and gives the
UI nothing to hang "Removed" / "Added" labels on.  :func:`build_hunks` turns two
texts into **separate change blocks**: each group of related edits becomes one
hunk holding its surrounding context, the removed lines and the replacement
lines, so ten unrelated edits stay ten small blocks instead of one giant
"all deletions / all additions" wall.

The same hunks drive :func:`reverse_hunks`, which undoes a change on top of the
file's *current* content, so reverting never discards edits the user made
elsewhere in the file after the change was applied.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

CONTEXT_LINES = 2          # lines of context shown around a change
MERGE_GAP = 2 * CONTEXT_LINES  # edits closer than this share one block
MAX_HUNKS = 60             # UI cap; the response says when it truncated
MAX_LINES_PER_BLOCK = 120  # UI cap per removed/added run
MAX_LINE_CHARS = 400


def _split(text: str) -> List[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _clip(line: str) -> str:
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + "…"


@dataclass
class _Change:
    """One contiguous edit: ``old[i1:i2]`` was replaced by ``new[j1:j2]``."""
    i1: int
    i2: int
    j1: int
    j2: int


def _changes(old: List[str], new: List[str]) -> List[_Change]:
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return [_Change(i1, i2, j1, j2)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def _group(changes: List[_Change]) -> List[List[_Change]]:
    groups: List[List[_Change]] = []
    for change in changes:
        if groups and change.i1 - groups[-1][-1].i2 <= MERGE_GAP:
            groups[-1].append(change)
        else:
            groups.append([change])
    return groups


def build_hunks(original: str, proposed: str) -> Dict[str, Any]:
    """Return ``{"hunks": [...], "additions": n, "deletions": n, "truncated": bool}``.

    Every hunk is ``{"id", "oldStart", "newStart", "removed", "added", "lines"}``
    with ``lines`` entries of ``{"type": "context"|"removed"|"added", "text",
    "oldNo", "newNo"}`` (line numbers are 1-based; absent on the other side).
    """
    old, new = _split(original), _split(proposed)
    if original == "":
        old = []
    if proposed == "":
        new = []
    groups = _group(_changes(old, new))
    hunks: List[Dict[str, Any]] = []
    total_added = total_removed = 0
    truncated = False

    for index, group in enumerate(groups):
        first, last = group[0], group[-1]
        start_old = max(0, first.i1 - CONTEXT_LINES)
        end_old = min(len(old), last.i2 + CONTEXT_LINES)
        lines: List[Dict[str, Any]] = []
        removed = added = 0
        old_pos = start_old
        for change in group:
            while old_pos < change.i1:
                lines.append({"type": "context", "text": _clip(old[old_pos]),
                              "oldNo": old_pos + 1, "newNo": old_pos - change.i1 + change.j1 + 1})
                old_pos += 1
            for offset, text in enumerate(old[change.i1:change.i2]):
                lines.append({"type": "removed", "text": _clip(text), "oldNo": change.i1 + offset + 1})
                removed += 1
            for offset, text in enumerate(new[change.j1:change.j2]):
                lines.append({"type": "added", "text": _clip(text), "newNo": change.j1 + offset + 1})
                added += 1
            old_pos = change.i2
        while old_pos < end_old:
            delta = last.j2 - last.i2
            lines.append({"type": "context", "text": _clip(old[old_pos]),
                          "oldNo": old_pos + 1, "newNo": old_pos + delta + 1})
            old_pos += 1
        total_added += added
        total_removed += removed
        if index < MAX_HUNKS:
            hunks.append({
                "id": f"h{index + 1}",
                "oldStart": start_old + 1,
                "newStart": first.j1 - (first.i1 - start_old) + 1,
                "removed": removed,
                "added": added,
                "lines": _cap_runs(lines),
            })
        else:
            truncated = True

    return {"hunks": hunks, "additions": total_added, "deletions": total_removed,
            "truncated": truncated, "hunkCount": len(groups)}


def _cap_runs(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cap very long runs of removed/added lines so one huge edit cannot flood the chat."""
    out: List[Dict[str, Any]] = []
    run_type: Optional[str] = None
    run_len = 0
    skipped = 0
    for line in lines + [{"type": "end"}]:
        kind = line["type"]
        if kind == run_type and kind in ("removed", "added"):
            run_len += 1
            if run_len > MAX_LINES_PER_BLOCK:
                skipped += 1
                continue
        else:
            if skipped:
                out.append({"type": "note", "text": f"… {skipped} more {run_type} line(s) not shown"})
            run_type, run_len, skipped = kind, 1, 0
        if kind != "end":
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Reversal
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _find_all(haystack: List[str], needle: List[str]) -> List[int]:
    if not needle:
        return []
    size = len(needle)
    return [i for i in range(len(haystack) - size + 1) if haystack[i:i + size] == needle]


def reverse_hunks(original: str, proposed: str, current: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Undo the ``original -> proposed`` change on top of *current*.

    Returns ``(new_text, conflicts)``.  ``new_text`` is ``None`` when any edit
    cannot be located unambiguously in *current* (the user changed or moved that
    part of the file); nothing is partially reverted in that case.
    """
    if _normalize(current) == _normalize(proposed):
        return _normalize(original), []
    old, new = _split(original), _split(proposed)
    cur = _split(current)
    changes = _changes(old, new)
    plans: List[Tuple[int, int, List[str]]] = []  # (start, end, replacement) in `cur`
    conflicts: List[Dict[str, Any]] = []
    for number, change in enumerate(changes, 1):
        added = new[change.j1:change.j2]
        removed = old[change.i1:change.i2]
        found: List[int] = []
        used = (0, 0)
        for ctx in (3, 2, 1, 0):
            before = new[max(0, change.j1 - ctx):change.j1]
            after = new[change.j2:change.j2 + ctx]
            needle = before + added + after
            if not needle:
                continue
            found = _find_all(cur, needle)
            if len(found) == 1:
                used = (len(before), len(after))
                break
            if len(found) > 1 and ctx == 0:
                break
        if len(found) != 1:
            conflicts.append({
                "change": number,
                "line": change.j1 + 1,
                "reason": "the edited lines were changed again or moved" if not found
                else "the edited lines appear more than once in the file",
            })
            continue
        start = found[0] + used[0]
        plans.append((start, start + len(added), removed))
    if conflicts:
        return None, conflicts
    # Overlapping plans mean two edits resolved to the same current lines.
    plans.sort()
    for a, b in zip(plans, plans[1:]):
        if a[1] > b[0]:
            return None, [{"change": 0, "line": a[0] + 1, "reason": "edits overlap in the current file"}]
    for start, end, replacement in reversed(plans):
        cur[start:end] = replacement
    return "\n".join(cur), []


__all__ = ["build_hunks", "reverse_hunks"]
