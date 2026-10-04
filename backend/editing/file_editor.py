"""Authoritative, safety-first store for proposed project file changes.

This module is the **single** mechanism through which the agent may modify a
project file.  The enforced lifecycle is::

    PROPOSED -> APPROVED -> APPLIED
         |          |
         |          +-> CONFLICTED   (file changed on disk since proposal)
         |          +-> FAILED       (write/verify error)
         +-> REJECTED / CANCELLED

Nothing here writes to a project file except :meth:`ChangeStore.apply`, and
``apply`` refuses to run unless the change is in ``APPROVED`` state and the
on-disk content still matches the hash captured at proposal time.  Every write
is atomic (temp file + ``os.replace``) and is verified after the fact.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from editing import resolve_project_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Change lifecycle states
# ---------------------------------------------------------------------------
PROPOSED = "PROPOSED"
APPROVED = "APPROVED"
REJECTED = "REJECTED"
APPLIED = "APPLIED"
FAILED = "FAILED"
CONFLICTED = "CONFLICTED"
CANCELLED = "CANCELLED"

#: States in which a change can no longer be applied.
TERMINAL_STATES = frozenset({APPLIED, REJECTED, CANCELLED, FAILED, CONFLICTED})

#: Supported operations. ``delete`` is the most destructive and is gated by the
#: same explicit-approval requirement as every other operation.
OP_CREATE = "create"
OP_MODIFY = "modify"
OP_DELETE = "delete"
ALLOWED_OPS = frozenset({OP_CREATE, OP_MODIFY, OP_DELETE})

#: A proposal that has not been acted on within this window is considered stale
#: and can no longer be approved.
PROPOSAL_TTL = timedelta(hours=24)

#: Store/backup directory names created inside the project.
STORE_DIRNAME = ".llm_training_agent"
CHANGES_DIRNAME = "pending_changes"
BACKUP_DIRNAME = ".llm_training_agent_backups"

#: Maximum size of a single proposal payload (matches core.security).
MAX_CONTENT_CHARS = 2_000_000


class ChangeStoreError(Exception):
    """Domain error carrying a stable machine-readable ``code``.

    The API layer maps ``code`` onto an HTTP status; the message is always
    written for a human and never contains absolute paths or secrets.
    """

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _read_bytes(path: Path) -> Optional[bytes]:
    """Read *path*, returning ``None`` when it is not an existing regular file."""
    try:
        if not path.exists() or not path.is_file():
            return None
        return path.read_bytes()
    except OSError as exc:
        raise ChangeStoreError(
            "FILE_UNREADABLE",
            f"The file could not be read: {exc.strerror or 'unknown error'}.",
            409,
        ) from exc


def _has_bom(data: Optional[bytes]) -> bool:
    """Whether *data* starts with a UTF-8 byte-order mark."""
    return bool(data) and data[:3] == b"\xef\xbb\xbf"


def _decode(data: Optional[bytes]) -> str:
    """Decode *data* as text, tolerating a BOM and undecodable bytes.

    A UTF-8 BOM is consumed here; :func:`_encode` re-adds it on write so the
    original file encoding survives an approved change.
    """
    if data is None:
        return ""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _encode(text: str, bom: bool = False) -> bytes:
    """Encode *text* as UTF-8, optionally restoring a leading BOM."""
    data = text.encode("utf-8")
    return (b"\xef\xbb\xbf" + data) if bom else data


def _atomic_write(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically, creating parent directories as needed.

    A temporary file in the same directory is written, flushed and fsynced
    before being moved into place with ``os.replace``.  A crash or a full disk
    therefore leaves either the old content or the new content, never a
    half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ChangeStoreError(
            "WRITE_FAILED",
            f"The file could not be written: {exc.strerror or 'unknown error'}.",
            500,
        ) from exc


def _detect_newline(text: str) -> str:
    """Return the dominant newline sequence of *text* (``\\r\\n`` or ``\\n``)."""
    return "\r\n" if "\r\n" in text else "\n"


def _apply_newline(text: str, newline: str) -> str:
    """Normalize every line ending in *text* to *newline*.

    Preserving the original file's line endings avoids an unrelated,
    whole-file diff when the proposal only changes a couple of lines.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    return unified if newline == "\n" else unified.replace("\n", newline)


def diff_stats(original: str, proposed: str) -> Tuple[int, int]:
    """Return ``(additions, deletions)`` between two texts."""
    import difflib

    added = deleted = 0
    matcher = difflib.SequenceMatcher(None, original.splitlines(), proposed.splitlines())
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            deleted += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, deleted


def unified_diff(original: str, proposed: str, file_path: str) -> str:
    """Build a unified diff between *original* and *proposed* for *file_path*."""
    import difflib

    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )


class ChangeStore:
    """Persist and govern proposed changes for a single project root.

    Proposals are stored as JSON under
    ``<project>/.llm_training_agent/pending_changes/<id>.json`` so they survive
    backend restarts.  The store is scoped to exactly one project root, which is
    what keeps concurrent workspaces from crossing over: a proposal created for
    project A is simply not reachable through a store built for project B.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self._store_dir = self.project_root / STORE_DIRNAME / CHANGES_DIRNAME
        self._backup_dir = self.project_root / BACKUP_DIRNAME
        self._memory: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def _change_file(self, change_id: str) -> Path:
        return self._store_dir / f"{change_id}.json"

    def _load_from_disk(self, change_id: str) -> Optional[Dict[str, Any]]:
        path = self._change_file(change_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # corrupt/truncated record
            logger.warning("Could not read pending change %s: %s", change_id, exc)
            return None
        if isinstance(data, dict):
            self._memory[change_id] = data
            return data
        return None

    def _persist(self, record: Dict[str, Any]) -> None:
        change_id = record["changeId"]
        self._memory[change_id] = record
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            self._change_file(change_id).write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - best effort persistence
            logger.warning("Could not persist change %s: %s", change_id, exc)

    def _all_records(self) -> List[Dict[str, Any]]:
        if self._store_dir.exists():
            for path in sorted(self._store_dir.glob("*.json")):
                if path.stem not in self._memory:
                    self._load_from_disk(path.stem)
        return list(self._memory.values())

    # ------------------------------------------------------------------
    # path safety
    # ------------------------------------------------------------------
    def project_file(self, file_path: str) -> Path:
        """Resolve *file_path* inside the project, rejecting traversal/escape.

        Delegates to :func:`core.security.resolve_workspace_child`, so absolute
        paths, ``..`` traversal and symlink escapes are rejected exactly as
        they are for every other backend route.
        """
        return resolve_project_file(self.project_root, file_path)

    @staticmethod
    def _relativize(root: Path, path: Path) -> str:
        """Return *path* relative to *root* in POSIX form (name as a fallback)."""
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            return path.name


    # ------------------------------------------------------------------
    # lifecycle: propose
    # ------------------------------------------------------------------
    def propose(
        self,
        file_path: str,
        proposed_content: Optional[str],
        operation: str = OP_MODIFY,
        reason: Optional[str] = None,
        source: str = "agent",
    ) -> Dict[str, Any]:
        """Record a proposed change **without touching the file**.

        The current on-disk bytes and their SHA-256 hash are captured so that a
        later :meth:`apply` can detect an intervening external modification
        instead of blindly overwriting the user's newer work.
        """
        if operation not in ALLOWED_OPS:
            raise ChangeStoreError(
                "INVALID_OPERATION",
                f"Unsupported operation '{operation}'. Allowed: {', '.join(sorted(ALLOWED_OPS))}.",
                422,
            )
        if operation == OP_DELETE and proposed_content is not None:
            raise ChangeStoreError(
                "INVALID_OPERATION", "A delete proposal must not carry new content.", 422
            )
        if operation != OP_DELETE and proposed_content is None:
            raise ChangeStoreError(
                "INVALID_OPERATION", f"A '{operation}' proposal requires new content.", 422
            )
        if proposed_content is not None and len(proposed_content) > MAX_CONTENT_CHARS:
            raise ChangeStoreError(
                "CONTENT_TOO_LARGE",
                f"The proposed content exceeds the {MAX_CONTENT_CHARS} character limit.",
                422,
            )

        target = self.project_file(file_path)
        exists = target.exists() and target.is_file()

        if operation == OP_CREATE and exists:
            raise ChangeStoreError(
                "FILE_EXISTS", "The file already exists; propose a modify instead.", 409
            )
        if operation in (OP_MODIFY, OP_DELETE) and not exists:
            raise ChangeStoreError("FILE_NOT_FOUND", "The file to change was not found.", 404)
        if exists and target.is_symlink() and operation == OP_DELETE:
            # resolve_project_file already rejects escaping symlinks; a link
            # that stays inside the project is tolerated for reads, but we
            # never follow one for an irreversible delete.
            raise ChangeStoreError(
                "UNSUPPORTED_TARGET", "Refusing to delete through a symbolic link.", 409
            )

        current_bytes = _read_bytes(target)
        original_text = _decode(current_bytes) if exists else ""
        current_hash = _hash_bytes(current_bytes) if current_bytes is not None else None
        newline = _detect_newline(original_text) if exists else "\n"
        bom = _has_bom(current_bytes)

        new_text = "" if operation == OP_DELETE else _apply_newline(proposed_content or "", newline)
        relative = self._relativize(self.project_root, target)
        additions, deletions = diff_stats(original_text, new_text)

        record: Dict[str, Any] = {
            "changeId": str(uuid.uuid4()),
            "projectRoot": str(self.project_root),
            "filePath": relative,
            "operation": operation,
            "status": PROPOSED,
            "originalContent": original_text,
            "proposedContent": new_text,
            "baseHash": current_hash,
            "baseExisted": exists,
            "newline": newline,
            "bom": bom,
            "diff": unified_diff(original_text, new_text, relative),
            "additions": additions,
            "deletions": deletions,
            "reason": reason,
            "source": source,
            "createdAt": _iso(_now()),
            "updatedAt": _iso(_now()),
        }
        self._persist(record)
        return record


    # ------------------------------------------------------------------
    # lifecycle: retrieval
    # ------------------------------------------------------------------
    def get(self, change_id: str) -> Optional[Dict[str, Any]]:
        """Return a stored change by id (memory first, then disk)."""
        if change_id in self._memory:
            return self._memory[change_id]
        return self._load_from_disk(change_id)

    def require(self, change_id: str) -> Dict[str, Any]:
        record = self.get(change_id)
        if record is None:
            raise ChangeStoreError("CHANGE_NOT_FOUND", "The change proposal was not found.", 404)
        return record

    def list(self, include_terminal: bool = True) -> List[Dict[str, Any]]:
        """Return change metadata (never file contents), newest first."""
        records = self._all_records()
        if not include_terminal:
            records = [r for r in records if r.get("status") not in TERMINAL_STATES]
        records.sort(key=lambda r: r.get("createdAt", ""), reverse=True)
        return [
            {k: v for k, v in r.items() if k not in ("originalContent", "proposedContent")}
            for r in records
        ]

    def is_expired(self, record: Dict[str, Any]) -> bool:
        """Whether *record* is older than :data:`PROPOSAL_TTL`."""
        try:
            created = datetime.fromisoformat(record.get("createdAt", ""))
        except (TypeError, ValueError):
            return True
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return _now() - created > PROPOSAL_TTL

    # ------------------------------------------------------------------
    # lifecycle: approve / reject / cancel
    # ------------------------------------------------------------------
    def approve(self, change_id: str) -> Dict[str, Any]:
        """Mark a proposal as user-approved.

        This is the *only* transition into a state from which :meth:`apply` will
        write to disk.  Re-approving or approving a non-proposal is refused so a
        stale UI cannot silently re-arm a change the user already declined.
        """
        record = self.require(change_id)
        status = record.get("status")
        if status == APPROVED:
            raise ChangeStoreError("ALREADY_APPROVED", "This change was already approved.", 409)
        if status != PROPOSED:
            raise ChangeStoreError(
                "INVALID_STATE", f"Only a PROPOSED change can be approved (current: {status}).", 409
            )
        if self.is_expired(record):
            record["status"] = CANCELLED
            record["updatedAt"] = _iso(_now())
            self._persist(record)
            raise ChangeStoreError(
                "PROPOSAL_EXPIRED", "This proposal has expired; ask the agent to regenerate it.", 409
            )
        record["status"] = APPROVED
        record["approvedAt"] = _iso(_now())
        record["updatedAt"] = _iso(_now())
        self._persist(record)
        return record

    def reject(self, change_id: str) -> Dict[str, Any]:
        """Reject a proposal. The target file is never touched."""
        record = self.require(change_id)
        status = record.get("status")
        if status == APPLIED:
            raise ChangeStoreError(
                "ALREADY_APPLIED", "This change was already applied; use rollback instead.", 409
            )
        if status in TERMINAL_STATES:
            raise ChangeStoreError("INVALID_STATE", f"Cannot reject a change in state {status}.", 409)
        record["status"] = REJECTED
        record["updatedAt"] = _iso(_now())
        self._persist(record)
        return record

    def cancel(self, change_id: str) -> Dict[str, Any]:
        """Cancel a change (used after a rollback, or to retire a proposal)."""
        record = self.require(change_id)
        if record.get("status") in (APPLIED, REJECTED):
            raise ChangeStoreError(
                "INVALID_STATE", f"Cannot cancel a change in state {record.get('status')}.", 409
            )
        record["status"] = CANCELLED
        record["updatedAt"] = _iso(_now())
        self._persist(record)
        return record


    # ------------------------------------------------------------------
    # lifecycle: apply
    # ------------------------------------------------------------------
    def check_conflict(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return conflict details when the target no longer matches the proposal.

        Comparing the current on-disk hash against the hash captured at proposal
        time is what prevents the agent from silently discarding edits the user
        made (in their editor, in git, or in another tool) while the proposal
        was waiting for review.
        """
        target = self.project_file(record["filePath"])
        current = _read_bytes(target)
        current_hash = _hash_bytes(current) if current is not None else None
        if current_hash != record.get("baseHash"):
            return {
                "reason": "The file changed on disk after this proposal was created.",
                "expectedHash": record.get("baseHash"),
                "actualHash": current_hash,
            }
        return None

    def verify(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Confirm the target file matches the applied proposal exactly."""
        target = self.project_file(record["filePath"])
        operation = record.get("operation", OP_MODIFY)
        exists = target.exists() and target.is_file()

        if operation == OP_DELETE:
            return {
                "verified": not exists,
                "filePath": record["filePath"],
                "operation": operation,
                "exists": exists,
            }
        if not exists:
            return {
                "verified": False,
                "filePath": record["filePath"],
                "operation": operation,
                "exists": False,
                "reason": "The file is missing after the write.",
            }
        current = _read_bytes(target) or b""
        expected = _encode(record.get("proposedContent") or "", bool(record.get("bom")))
        return {
            "verified": current == expected,
            "filePath": record["filePath"],
            "operation": operation,
            "exists": True,
            "expectedHash": _hash_bytes(expected),
            "actualHash": _hash_bytes(current),
        }

    def apply(self, change_id: str) -> Dict[str, Any]:
        """Apply an **approved** change atomically, then verify the result.

        Refuses (409) when the change is not ``APPROVED``, when it was already
        applied, or when the target changed since the proposal was generated.
        A write or verification failure moves the change to ``FAILED`` and keeps
        the original content recoverable from the backup.
        """
        record = self.require(change_id)
        status = record.get("status")
        if status == APPLIED:
            raise ChangeStoreError("ALREADY_APPLIED", "This change was already applied.", 409)
        if status != APPROVED:
            raise ChangeStoreError(
                "NOT_APPROVED",
                f"This change must be approved before it can be applied (current: {status}).",
                409,
            )

        conflict = self.check_conflict(record)
        if conflict:
            record["status"] = CONFLICTED
            record["conflict"] = conflict
            record["updatedAt"] = _iso(_now())
            self._persist(record)
            raise ChangeStoreError(
                "FILE_CONFLICT",
                "The file changed since this proposal was created. "
                "Review the current content and regenerate the proposal.",
                409,
                conflict,
            )

        target = self.project_file(record["filePath"])
        operation = record.get("operation", OP_MODIFY)
        backup_path: Optional[Path] = None

        try:
            if operation == OP_DELETE:
                current = _read_bytes(target)
                if current is not None:
                    backup_path = self._backup(record, current)
                    try:
                        target.unlink()
                    except OSError as exc:
                        raise ChangeStoreError(
                            "WRITE_FAILED",
                            f"The file could not be deleted: {exc.strerror or 'unknown error'}.",
                            500,
                        ) from exc
            else:
                if record.get("baseExisted"):
                    current = _read_bytes(target)
                    if current is not None:
                        backup_path = self._backup(record, current)
                _atomic_write(
                    target, _encode(record.get("proposedContent") or "", bool(record.get("bom")))
                )

            verification = self.verify(record)
            if not verification["verified"]:
                raise ChangeStoreError(
                    "VERIFICATION_FAILED",
                    "The file was written but its content could not be verified.",
                    500,
                    verification,
                )
        except ChangeStoreError as exc:
            record["status"] = FAILED
            record["error"] = {"code": exc.code, "message": exc.message}
            record["updatedAt"] = _iso(_now())
            self._persist(record)
            raise

        record["status"] = APPLIED
        record["appliedAt"] = _iso(_now())
        record["updatedAt"] = _iso(_now())
        if backup_path is not None:
            record["backupPath"] = str(backup_path)
        self._persist(record)
        return record


    # ------------------------------------------------------------------
    # lifecycle: rollback and cleanup
    # ------------------------------------------------------------------
    def rollback(self, change_id: str) -> Dict[str, Any]:
        """Restore the pre-apply content from the stored backup."""
        record = self.require(change_id)
        if record.get("status") != APPLIED:
            raise ChangeStoreError("NOT_APPLIED", "Only an applied change can be rolled back.", 409)
        backup_str = record.get("backupPath")
        if not backup_str:
            raise ChangeStoreError("NO_BACKUP", "No backup is available for this change.", 409)

        backup_path = Path(backup_str).resolve()
        backup_root = self._backup_dir.resolve()
        if backup_root not in backup_path.parents or not backup_path.is_file():
            raise ChangeStoreError("BACKUP_NOT_FOUND", "The backup for this change was not found.", 404)

        target = self.project_file(record["filePath"])
        if record.get("baseExisted"):
            _atomic_write(target, backup_path.read_bytes())
        else:
            # The change created this file, so rolling back removes it again.
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                raise ChangeStoreError(
                    "WRITE_FAILED",
                    f"The file could not be removed: {exc.strerror or 'unknown error'}.",
                    500,
                ) from exc

        record["status"] = CANCELLED
        record["rolledBackAt"] = _iso(_now())
        record["updatedAt"] = _iso(_now())
        self._persist(record)
        return record

    def _backup(self, record: Dict[str, Any], data: bytes) -> Path:
        """Persist the pre-change bytes so the operation stays recoverable."""
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        safe = record["filePath"].replace("/", "_").replace("\\", "_")
        path = self._backup_dir / f"{safe}.{record['changeId']}.bak"
        _atomic_write(path, data)
        return path

    def remove(self, change_id: str) -> None:
        """Delete a stored proposal record (housekeeping only)."""
        self._memory.pop(change_id, None)
        try:
            self._change_file(change_id).unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Could not delete pending change %s: %s", change_id, exc)
