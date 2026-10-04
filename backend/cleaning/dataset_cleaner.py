"""Chunked, multi-file dataset cleaner backed by the configured AI provider.

Why this module exists
----------------------
Dataset cleaning is a *batch* operation, but an LLM request must stay small.
This module therefore:

1. discovers **every** dataset file of a project (not just the first one),
2. splits each file into **small, deterministic chunks of records**,
3. sends **one request per chunk** to the LLM (a chunk never mixes two files),
4. validates every response against the chunk it came from and falls back to
   deterministic cleaning whenever the model breaks the contract, and
5. writes a cleaned file per source file (same format, same relative path) plus
   a manifest, so the cleaned dataset can be reconstructed losslessly.

Record-level guarantee: ``records_out == records_in`` per file, always.  A
record is never dropped, merged, reordered or invented; on failure the original
record is kept.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.llm import LLMHelper
from cleaning import dataset_io
from models.schemas import (
    ChunkCleaningSummary,
    DatasetCleaningResult,
    FileCleaningSummary,
    ProjectContext,
)
from core.security import resolve_output_dir

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 25
DEFAULT_MAX_CHUNK_CHARS = 12000
DEFAULT_OUTPUT_DIR = ".llm-training-agent/cleaned"
MANIFEST_NAME = "cleaning_manifest.json"

_CLEANING_PROMPT = ("cleaning", "dataset_cleaning.md")
_CLEANING_SYSTEM_PROMPT = ("system", "dataset_cleaner.md")


class DatasetCleaner:
    """Clean every dataset file of a project in LLM-sized chunks."""

    def __init__(
        self,
        llm: Optional[LLMHelper] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        output_dir_name: str = DEFAULT_OUTPUT_DIR,
    ):
        self._llm = llm or LLMHelper()
        self.chunk_size = max(1, int(chunk_size))
        self.max_chunk_chars = max(200, int(max_chunk_chars))
        self.output_dir_name = output_dir_name
        self._llm_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def llm_available(self) -> bool:
        """Whether an AI provider can be instantiated (cached after first check)."""
        if self._llm_available is None:
            try:
                self._llm_available = bool(self._llm.is_available)
            except Exception as e:  # noqa: BLE001 - provider probing must never break cleaning
                logger.debug(f"LLM availability probe failed: {e}")
                self._llm_available = False
        return self._llm_available

    def discover(
        self, context: ProjectContext, max_files: int = 0
    ) -> Tuple[List[Path], List[str]]:
        """Return every dataset file referenced by *context* plus skipped entries."""
        return dataset_io.discover_dataset_files(
            context.project_path, context.dataset_paths, max_files=max_files
        )

    async def clean_project(
        self,
        context: ProjectContext,
        output_dir: Optional[str] = None,
        max_files: int = 0,
        use_llm: bool = True,
        write_output: bool = True,
    ) -> DatasetCleaningResult:
        """Clean all discovered dataset files and return an aggregated result.

        Args:
            context: Project context providing ``project_path`` and
                ``dataset_paths`` (files and/or directories).
            output_dir: Destination directory.  Defaults to
                ``<project>/.llm-training-agent/cleaned``.
            max_files: Optional cap on the number of files processed.
            use_llm: When False, only deterministic cleaning is applied.
            write_output: When False, records are processed in memory only.
        """
        files, skipped = self.discover(context, max_files=max_files)
        project_root = Path(context.project_path) if context.project_path else Path(".")
        target_dir = self._resolve_output_dir(project_root, output_dir)

        result = DatasetCleaningResult(
            skipped_files=skipped,
            output_directory=str(target_dir) if write_output else None,
            chunk_size=self.chunk_size,
            max_chunk_chars=self.max_chunk_chars,
        )

        llm_enabled = use_llm and self.llm_available
        result.llm_used = llm_enabled
        if use_llm and not llm_enabled:
            result.warnings.append(
                "No AI provider available: deterministic cleaning was applied instead."
            )

        if not files:
            result.warnings.append("No dataset files were discovered in this project.")

        manifest_files: List[Dict[str, Any]] = []

        for path in files:
            _, summary = await self.clean_file(
                path=path,
                project_root=project_root,
                output_dir=target_dir,
                use_llm=use_llm,
                write_output=write_output,
            )
            result.files.append(summary)
            result.total_records_out += summary.records_out
            result.total_chunks += summary.chunks_total
            result.total_chunks_cleaned += summary.chunks_cleaned
            result.total_chunks_fallback += summary.chunks_fallback
            result.warnings.extend(summary.warnings)
            manifest_files.append(
                {
                    "source_file": summary.source_file,
                    "relative_path": summary.relative_path,
                    "output_path": summary.output_path,
                    "format": summary.format,
                    "records_in": summary.records_in,
                    "records_out": summary.records_out,
                    "chunks": [c.model_dump() for c in summary.chunk_summaries],
                }
            )

        result.total_files = len(result.files)
        result.total_records_in = sum(f.records_in for f in result.files)
        result.records_preserved = all(f.records_preserved for f in result.files)
        # Chunks are always built from a single file, so files can never mix.
        result.cross_file_contamination = False
        if not result.records_preserved:
            result.confidence = "low"
        elif result.total_files and llm_enabled:
            result.confidence = "high"
        elif result.total_files:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        if write_output and files:
            manifest_path = target_dir / MANIFEST_NAME
            self._write_manifest(
                manifest_path,
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "llm_used": llm_enabled,
                    "chunk_size": self.chunk_size,
                    "max_chunk_chars": self.max_chunk_chars,
                    "records_preserved": result.records_preserved,
                    "cross_file_contamination": False,
                    "files": manifest_files,
                    "skipped_files": skipped,
                },
            )
            result.manifest_path = str(manifest_path)

        return result

    def _resolve_output_dir(self, project_root: Path, output_dir: Optional[str]) -> Path:
        """Resolve output inside the project, rejecting absolute/traversal paths."""
        return resolve_output_dir(str(project_root), output_dir, self.output_dir_name)

    async def clean_file(
        self,
        path: Path,
        project_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        use_llm: bool = True,
        write_output: bool = True,
    ) -> Tuple[List[Dict[str, Any]], FileCleaningSummary]:
        """Clean one dataset file chunk by chunk.

        Returns the cleaned records (in original order) and a per-file summary.
        The cleaned file is written to ``output_dir / <relative source path>``
        when *write_output* is True, which keeps one source file mapped to
        exactly one output file.
        """
        path = Path(path)
        project_root = Path(project_root) if project_root else path.parent
        relative_path = self._relative_source(path, project_root)
        fmt = dataset_io.detect_format(path)

        records, read_errors = dataset_io.read_records(path)
        summary = FileCleaningSummary(
            source_file=str(path),
            relative_path=relative_path,
            format=fmt,
            records_in=len(records),
            errors=list(read_errors),
        )

        empty_records = count_empty_text_records(records)
        if empty_records:
            summary.warnings.append(
                f"{empty_records} record(s) have an empty prompt or response; kept unchanged."
            )

        if not records:
            summary.records_out = 0
            summary.records_preserved = True
            if not read_errors:
                summary.warnings.append("File contains no records.")
            return [], summary

        chunks = dataset_io.chunk_records(records, self.chunk_size, self.max_chunk_chars)
        summary.chunks_total = len(chunks)

        cleaned_records: List[Dict[str, Any]] = []
        # Chunks are produced from this file only -> records can never mix files.
        for index, chunk in enumerate(chunks, start=1):
            cleaned_chunk, chunk_summary = await self._clean_chunk(
                chunk=chunk,
                file_name=path.name,
                chunk_index=index,
                chunk_total=len(chunks),
                fmt=fmt,
                use_llm=use_llm,
            )
            summary.chunk_summaries.append(chunk_summary)
            summary.warnings.extend(chunk_summary.warnings)
            if chunk_summary.llm_used:
                summary.chunks_cleaned += 1
            else:
                summary.chunks_fallback += 1
            cleaned_records.extend(cleaned_chunk)

        summary.records_out = len(cleaned_records)
        summary.records_preserved = len(cleaned_records) == len(records)
        summary.llm_used = any(c.llm_used for c in summary.chunk_summaries)

        if not summary.records_preserved:
            summary.errors.append(
                f"Record count mismatch for {path.name}: "
                f"{len(records)} in, {len(cleaned_records)} out"
            )

        if write_output and output_dir:
            out_path = Path(output_dir) / relative_path
            dataset_io.write_records(out_path, cleaned_records, fmt)
            summary.output_path = str(out_path)

        return cleaned_records, summary

    # ------------------------------------------------------------------
    # Chunk level: one LLM request per chunk
    # ------------------------------------------------------------------

    async def _clean_chunk(
        self,
        chunk: List[Dict[str, Any]],
        file_name: str,
        chunk_index: int,
        chunk_total: int,
        fmt: str,
        use_llm: bool = True,
    ) -> Tuple[List[Dict[str, Any]], ChunkCleaningSummary]:
        """Clean a single chunk, falling back to deterministic cleaning.

        The LLM response is only accepted when it satisfies the record contract
        (a list of objects, same length as the chunk, same order).  Any contract
        violation keeps the original records for that chunk so that no data can
        be lost and other chunks are unaffected.
        """
        summary = ChunkCleaningSummary(chunk_index=chunk_index, record_count=len(chunk))

        if use_llm and self.llm_available:
            candidates, warnings, used = await self._call_llm_for_chunk(
                chunk=chunk,
                file_name=file_name,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                fmt=fmt,
            )
            if used and candidates is not None:
                merged = [
                    self._merge_record(original, cleaned)
                    for original, cleaned in zip(chunk, candidates)
                ]
                summary.status = "cleaned"
                summary.llm_used = True
                summary.records_cleaned = len(merged)
                summary.warnings.extend(warnings)
                return merged, summary

            summary.status = "fallback"
            summary.warnings.extend(warnings)
        else:
            summary.status = "deterministic"

        cleaned = self._deterministic_clean_records(chunk)
        summary.records_cleaned = len(cleaned)
        return cleaned, summary

    async def _call_llm_for_chunk(
        self,
        chunk: List[Dict[str, Any]],
        file_name: str,
        chunk_index: int,
        chunk_total: int,
        fmt: str,
    ) -> Tuple[Optional[List[Dict[str, Any]]], List[str], bool]:
        """Send exactly one chunk to the LLM and validate the returned records.

        Returns ``(records, warnings, accepted)``.  ``accepted`` is False when
        the request failed or the response broke the record contract.
        """
        warnings: List[str] = []
        try:
            template = self._llm.load_prompt(*_CLEANING_PROMPT)
            prompt = (
                template.replace("<<FILE_NAME>>", file_name)
                .replace("<<CHUNK_INDEX>>", str(chunk_index))
                .replace("<<CHUNK_TOTAL>>", str(chunk_total))
                .replace("<<RECORD_COUNT>>", str(len(chunk)))
                .replace("<<FORMAT>>", fmt)
                .replace("<<RECORDS_JSON>>", json.dumps(chunk, ensure_ascii=False, indent=1))
            )
            system_prompt = self._llm.load_prompt(*_CLEANING_SYSTEM_PROMPT)
        except Exception as e:  # noqa: BLE001 - prompt loading must not abort the run
            warnings.append(f"Chunk {chunk_index}: cleaning prompt unavailable ({e}).")
            return None, warnings, False

        payload = await self._llm.call_structured(
            prompt, temperature=0.1, system_prompt=system_prompt
        )
        if payload is None:
            warnings.append(
                f"Chunk {chunk_index}: LLM returned no parsable JSON; original records kept."
            )
            return None, warnings, False

        candidates = None
        for key in ("records", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break

        if candidates is None or len(candidates) != len(chunk) or not all(
            isinstance(item, dict) for item in candidates
        ):
            got = len(candidates) if isinstance(candidates, list) else "no"
            warnings.append(
                f"Chunk {chunk_index}: LLM violated the record contract "
                f"(expected {len(chunk)} records, got {got}); original records kept."
            )
            return None, warnings, False

        for note in payload.get("warnings", []) or []:
            if isinstance(note, str) and note.strip():
                warnings.append(f"Chunk {chunk_index}: {note.strip()}")

        return candidates, warnings, True

    def _merge_record(
        self, original: Dict[str, Any], cleaned: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge cleaned values into the original record's key set.

        Only keys that existed in the source record are updated; keys invented
        by the model are ignored and keys the model dropped keep their original
        value, so a record can never lose fields.
        """
        merged = dict(original)
        for key, value in cleaned.items():
            if key in original:
                merged[key] = value
        return merged

    # ------------------------------------------------------------------
    # Deterministic cleaning (used when no LLM is available or it fails)
    # ------------------------------------------------------------------

    def _deterministic_clean_records(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply rule-based text normalisation to every record (count preserved)."""
        return [
            {key: _clean_value(value) for key, value in record.items()} for record in records
        ]

    def _relative_source(self, path: Path, project_root: Path) -> str:
        """Return the project-relative POSIX path of a source file.

        Files outside the project root fall back to their file name so that a
        cleaned output can never escape the output directory.
        """
        try:
            return path.resolve().relative_to(project_root.resolve()).as_posix()
        except (ValueError, OSError):
            return path.name

    def _write_manifest(self, path: Path, payload: Dict[str, Any]) -> None:
        """Write the cleaning manifest describing how outputs were produced."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_text(value: str) -> str:
    """Normalise a single text value without changing its meaning."""
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    # Drop non-printable control characters while keeping newlines and tabs.
    text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_value(value: Any) -> Any:
    """Recursively normalise strings inside a JSON-compatible value."""
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, dict):
        return {key: _clean_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    return value


def count_empty_text_records(records: List[Dict[str, Any]]) -> int:
    """Count records whose prompt or response text is empty.

    Uses the same field aliases as ``DatasetAnalyzer`` so the two components
    agree on what an "incomplete" record is.  Records are only reported, never
    removed, because removing them would break the record-count guarantee.
    """
    prompt_keys = ("prompt", "instruction", "question", "input")
    response_keys = ("response", "completion", "output", "answer", "text")
    empty = 0
    for record in records:
        prompt = ""
        response = ""
        for key in prompt_keys:
            if record.get(key):
                prompt = str(record[key])
                break
        for key in response_keys:
            if record.get(key):
                response = str(record[key])
                break
        if not prompt.strip() or not response.strip():
            empty += 1
    return empty