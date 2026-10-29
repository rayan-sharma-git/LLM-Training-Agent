"""Cleaning jobs: a durable record of what was cleaned, where, and how to merge it.

A *job* is written to ``<project>/.llm-training-agent/cleaning_jobs/<id>.json``
the moment cleaning finishes.  It lists every source file (with the hash it had
when cleaning started), the cleaned copy, the method and the output mode.  Because
it lives in the project - not in a chat or in process memory - a user can say
"merge the cleaned data into my files" after restarting VS Code, from a
different chat, or days later.

Output modes
------------
``separate``
    Cleaned copies stay in ``.llm-training-agent/cleaned/<id>/`` until the user
    asks to merge them.
``in_place``
    The originals are replaced as soon as cleaning finishes (after backing them
    up), and the temporary folder is removed.

Merging never overwrites a file that changed after cleaning started, always
keeps a backup of the original, verifies what it wrote, and removes the
temporary folder only when every file merged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cleaning import dataset_io
from cleaning.dataset_cleaner import DatasetCleaner
from editing.file_editor import _atomic_write as atomic_write
from models.schemas import DatasetCleaningResult, ProjectContext

logger = logging.getLogger(__name__)

STORE_DIR = ".llm-training-agent"
JOBS_DIR = "cleaning_jobs"
CLEANED_DIR = "cleaned"
BACKUP_ROOT = ".llm_training_agent_backups"

METHODS = ("ai", "script")
OUTPUT_MODES = ("separate", "in_place")

PENDING = "completed"      # cleaned, waiting to be merged (separate mode)
MERGED = "merged"
PARTIAL = "partially_merged"
DISCARDED = "discarded"
RESTORED = "restored"


class CleaningJobError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CleaningJobs:
    """Jobs of one project root."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        self.jobs_dir = self.root / STORE_DIR / JOBS_DIR
        self.cleaned_root = self.root / STORE_DIR / CLEANED_DIR

    # ------------------------------------------------------------------ storage
    def _path(self, job_id: str) -> Path:
        if not job_id or not all(ch.isalnum() or ch in "-_" for ch in job_id):
            raise CleaningJobError("INVALID_JOB", "Invalid cleaning job id.")
        return self.jobs_dir / f"{job_id}.json"

    def _save(self, job: Dict[str, Any]) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(job["jobId"])
        atomic_write(tmp, json.dumps(job, indent=2, ensure_ascii=False).encode("utf-8"))

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(job_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        if self.jobs_dir.is_dir():
            for path in self.jobs_dir.glob("*.json"):
                job = self.get(path.stem)
                if job and (status is None or job.get("status") == status):
                    jobs.append(job)
        jobs.sort(key=lambda j: j.get("createdAt", ""), reverse=True)
        return jobs

    def pending_merges(self) -> List[Dict[str, Any]]:
        """Jobs whose cleaned copies still wait to be merged (survives restarts)."""
        return [j for j in self.list() if j.get("status") in (PENDING, PARTIAL)]

    # ------------------------------------------------------------------ cleaning
    async def run(
        self,
        context: ProjectContext,
        cleaner: DatasetCleaner,
        method: str,
        output_mode: str,
    ) -> Dict[str, Any]:
        """Clean every dataset file and record the job (and merge it in ``in_place`` mode)."""
        if method not in METHODS:
            raise CleaningJobError("INVALID_METHOD", "Cleaning method must be 'ai' or 'script'.")
        if output_mode not in OUTPUT_MODES:
            raise CleaningJobError("INVALID_OUTPUT_MODE", "Output mode must be 'separate' or 'in_place'.")

        job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        files, skipped = cleaner.discover(context)
        store = (self.root / STORE_DIR).resolve()
        files = [f for f in files if store not in f.resolve().parents]  # never re-clean our own output
        if not files:
            raise CleaningJobError("NO_DATASETS", "No dataset files were found in this project.")
        hashes = {str(p): _sha256(p) for p in files}

        out_rel = f"{STORE_DIR}/{CLEANED_DIR}/{job_id}"
        context = context.model_copy(update={
            "dataset_paths": [str(f.resolve().relative_to(self.root).as_posix()) for f in files]
        })
        result: DatasetCleaningResult = await cleaner.clean_project(
            context, output_dir=out_rel, use_llm=(method == "ai"), write_output=True
        )

        entries: List[Dict[str, Any]] = []
        for summary in result.files:
            source = Path(summary.source_file)
            if summary.output_path is None:
                continue
            entries.append({
                "relativePath": summary.relative_path,
                "sourceHash": hashes.get(str(source)) or hashes.get(str(source.resolve()), ""),
                "cleanedPath": Path(summary.output_path).resolve().relative_to(self.root).as_posix(),
                "format": summary.format,
                "recordsIn": summary.records_in,
                "recordsOut": summary.records_out,
                "recordsPreserved": summary.records_preserved,
                "failureSummary": summary.failure_summary,
                "status": "cleaned",
            })

        job: Dict[str, Any] = {
            "jobId": job_id,
            "createdAt": _now(),
            "projectRoot": str(self.root),
            "method": method,
            "outputMode": output_mode,
            "status": PENDING,
            "outputDirectory": out_rel,
            "files": entries,
            "skippedFiles": skipped,
            "summary": result.summary_text,
            "failureSummary": result.failure_summary,
            "warnings": result.warnings[:20],
            "recordsPreserved": result.records_preserved,
        }
        self._save(job)
        if output_mode == "in_place":
            report = self.merge(job_id)
            job = self.get(job_id) or job
            job["mergeReport"] = report
        return job

    # ------------------------------------------------------------------ merge
    def merge(self, job_id: str) -> Dict[str, Any]:
        """Replace the original files with their cleaned copies, then tidy up.

        Returns ``{"merged": [...], "skipped": [{"file", "reason"}], "removedTemp": bool, ...}``.
        """
        job = self.get(job_id)
        if job is None:
            raise CleaningJobError("JOB_NOT_FOUND", "That cleaning job was not found.")
        if job.get("status") not in (PENDING, PARTIAL):
            raise CleaningJobError("NOT_PENDING", f"This cleaning job is already {job.get('status')}.")

        backup_dir = self.root / BACKUP_ROOT / "cleaning" / job_id
        merged: List[str] = []
        skipped: List[Dict[str, str]] = []

        for entry in job["files"]:
            if entry.get("status") == "merged":
                merged.append(entry["relativePath"])
                continue
            reason = self._merge_one(job, entry, backup_dir)
            if reason is None:
                entry["status"] = "merged"
                merged.append(entry["relativePath"])
            else:
                entry["status"] = "skipped"
                entry["skipReason"] = reason
                skipped.append({"file": entry["relativePath"], "reason": reason})

        removed = False
        if not skipped:
            job["status"] = MERGED
            removed = self._remove_temp(job)
        else:
            job["status"] = PARTIAL if merged else PENDING
        job["mergedAt"] = _now()
        job["backupDirectory"] = str(backup_dir.relative_to(self.root).as_posix()) if backup_dir.exists() else None
        self._save(job)
        return {
            "jobId": job_id,
            "merged": merged,
            "skipped": skipped,
            "removedTemp": removed,
            "backupDirectory": job["backupDirectory"],
            "status": job["status"],
        }

    def _merge_one(self, job: Dict[str, Any], entry: Dict[str, Any], backup_dir: Path) -> Optional[str]:
        """Merge one file; return ``None`` on success or the reason it was skipped."""
        original = (self.root / entry["relativePath"]).resolve()
        cleaned = (self.root / entry["cleanedPath"]).resolve()
        try:
            original.relative_to(self.root)
            cleaned.relative_to(self.root / STORE_DIR / CLEANED_DIR)
        except ValueError:
            return "path is outside the project"
        if not original.is_file():
            return "the original file no longer exists"
        if not cleaned.is_file():
            return "the cleaned copy is missing"
        if entry.get("sourceHash") and _sha256(original) != entry["sourceHash"]:
            return "the original changed after it was cleaned (merging would overwrite your newer edits)"
        records, errors = dataset_io.read_records(cleaned)
        if errors or len(records) != entry.get("recordsOut", len(records)):
            return "the cleaned copy could not be read back correctly"
        try:
            backup = backup_dir / entry["relativePath"]
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, backup)
            data = cleaned.read_bytes()
            atomic_write(original, data)
            if _sha256(original) != _sha256(cleaned):
                atomic_write(original, backup.read_bytes())
                return "the merged file failed verification and was restored from the backup"
        except OSError as error:
            return f"the file could not be written ({error.strerror or 'I/O error'})"
        entry["mergedHash"] = _sha256(original)
        return None

    def _remove_temp(self, job: Dict[str, Any]) -> bool:
        """Delete this job's cleaned folder (only inside ``.llm-training-agent/cleaned``)."""
        target = (self.root / job["outputDirectory"]).resolve()
        try:
            target.relative_to(self.cleaned_root.resolve())
        except ValueError:
            return False
        if target == self.cleaned_root.resolve():
            return False
        try:
            if target.exists():
                shutil.rmtree(target)
            for leftover in (self.cleaned_root / "cleaning_manifest.json",):
                if leftover.exists() and not any(self.cleaned_root.glob("*/")):
                    leftover.unlink()
            if self.cleaned_root.is_dir() and not any(self.cleaned_root.iterdir()):
                self.cleaned_root.rmdir()
            return not target.exists()
        except OSError as error:
            logger.warning("Could not remove temporary cleaned folder: %s", error)
            return False

    # ------------------------------------------------------------------ restore / discard
    def restore(self, job_id: str) -> Dict[str, Any]:
        """Put the backed-up originals back (only for files not edited since the merge)."""
        job = self.get(job_id)
        if job is None:
            raise CleaningJobError("JOB_NOT_FOUND", "That cleaning job was not found.")
        if job.get("status") not in (MERGED, PARTIAL):
            raise CleaningJobError("NOT_MERGED", "Only a merged cleaning job can be restored.")
        backup_dir = self.root / BACKUP_ROOT / "cleaning" / job_id
        restored: List[str] = []
        skipped: List[Dict[str, str]] = []
        for entry in job["files"]:
            if entry.get("status") != "merged":
                continue
            original = (self.root / entry["relativePath"]).resolve()
            backup = backup_dir / entry["relativePath"]
            if not backup.is_file():
                skipped.append({"file": entry["relativePath"], "reason": "its backup is missing"})
            elif not original.is_file() or _sha256(original) != entry.get("mergedHash"):
                skipped.append({"file": entry["relativePath"],
                                "reason": "it was edited after the merge (restoring would lose those edits)"})
            else:
                atomic_write(original, backup.read_bytes())
                entry["status"] = "restored"
                restored.append(entry["relativePath"])
        job["status"] = RESTORED if not skipped and restored else job["status"]
        job["restoredAt"] = _now()
        self._save(job)
        return {"jobId": job_id, "restored": restored, "skipped": skipped}

    def discard(self, job_id: str) -> bool:
        """Drop a pending job and its cleaned copies without touching the originals."""
        job = self.get(job_id)
        if job is None or job.get("status") not in (PENDING, PARTIAL):
            return False
        self._remove_temp(job)
        job["status"] = DISCARDED
        self._save(job)
        return True


__all__ = ["CleaningJobs", "CleaningJobError", "METHODS", "OUTPUT_MODES", "PENDING", "MERGED", "PARTIAL"]
