"""Chunked, multi-file dataset cleaner backed by the configured AI provider.

How it works
------------
Dataset cleaning is a *batch* operation, but an LLM request must stay small, so:

1. every dataset file of a project is discovered,
2. each file is split into small, ordered chunks of records,
3. in AI mode each chunk is sent in its own request (a chunk never mixes files),
4. every response is checked against the chunk it came from; if the model breaks
   the contract the chunk falls back to plain script cleaning, and
5. the cleaned file replaces the original (atomically), or - when the caller
   asks for it - a copy is saved under ``.llm-training-agent/cleaned`` so it can
   be reviewed and merged into the originals later.

Scripts (no AI, free) normalise whitespace and control characters and drop exact
duplicates and empty / meaningless records. The AI additionally fixes typos and
broken sentences and may flag a record that makes no sense at all for removal.

Nothing else is written: no manifests, no logs.

Count guarantee per file: ``records_out + duplicates_removed + junk_removed
== records_in``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
#: Written by older versions; never created any more, removed when found.
LEGACY_MANIFEST_NAME = "cleaning_manifest.json"

ProgressCallback = Callable[[Dict[str, Any]], None]

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
        remove_duplicates: bool = True,
        in_place: bool = True,
        on_progress: Optional[ProgressCallback] = None,
    ) -> DatasetCleaningResult:
        """Clean all discovered dataset files and return an aggregated result.

        Args:
            context: Project context providing ``project_path`` and
                ``dataset_paths`` (files and/or directories).
            output_dir: Destination for copies when *in_place* is False.
                Defaults to ``<project>/.llm-training-agent/cleaned``.
            max_files: Optional cap on the number of files processed.
            use_llm: When False, only script cleaning is applied (free).
            write_output: When False, records are processed in memory only.
            remove_duplicates: Drop exact duplicate records (first copy kept).
                A script step, so it costs nothing even in AI mode.
            in_place: Rewrite the original files (default) instead of saving
                copies into *output_dir*.
            on_progress: Called with ``{file_index, file_total, file_name,
                chunk_index, chunk_total}`` before every chunk.
        """
        files, skipped = self.discover(context, max_files=max_files)
        project_root = Path(context.project_path) if context.project_path else Path(".")
        target_dir = None if in_place else self._resolve_output_dir(project_root, output_dir)

        result = DatasetCleaningResult(
            skipped_files=skipped,
            output_directory=str(target_dir) if (write_output and target_dir) else None,
            in_place=in_place,
            chunk_size=self.chunk_size,
            max_chunk_chars=self.max_chunk_chars,
        )

        llm_enabled = use_llm and self.llm_available
        result.llm_used = False
        if use_llm and not llm_enabled:
            result.warnings.append(
                "No AI provider available: scripts were used instead."
            )

        if not files:
            result.warnings.append("No dataset files were discovered in this project.")

        for index, path in enumerate(files, start=1):
            _, summary = await self.clean_file(
                path=path,
                project_root=project_root,
                output_dir=target_dir,
                use_llm=use_llm,
                write_output=write_output,
                remove_duplicates=remove_duplicates,
                in_place=in_place,
                on_progress=(
                    (lambda info, i=index: on_progress({**info, "file_index": i, "file_total": len(files)}))
                    if on_progress
                    else None
                ),
            )
            result.files.append(summary)
            result.total_records_out += summary.records_out
            result.total_duplicates_removed += summary.duplicates_removed
            result.total_junk_removed += summary.junk_removed
            result.total_files_changed += 1 if summary.changed else 0
            result.total_chunks += summary.chunks_total
            result.total_chunks_cleaned += summary.chunks_cleaned
            result.total_chunks_fallback += summary.chunks_fallback
            result.warnings.extend(summary.warnings)

        result.total_files = len(result.files)
        result.total_records_in = sum(f.records_in for f in result.files)
        result.records_preserved = all(f.records_preserved for f in result.files)
        result.llm_used = result.total_chunks_cleaned > 0
        # Chunks are always built from a single file, so files can never mix.
        result.cross_file_contamination = False
        if llm_enabled and result.total_chunks and not result.total_chunks_cleaned:
            result.warnings.append(
                "The AI model did not return a usable answer for any chunk, so only "
                "scripts were applied. Check the provider and key in Settings."
            )
        if not result.records_preserved:
            result.confidence = "low"
        elif result.total_files and result.llm_used:
            result.confidence = "high"
        elif result.total_files:
            result.confidence = "medium"
        else:
            result.confidence = "low"
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
        remove_duplicates: bool = True,
        in_place: bool = True,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Tuple[List[Dict[str, Any]], FileCleaningSummary]:
        """Clean one dataset file chunk by chunk.

        Returns the cleaned records (in original order) and a per-file summary.
        With *in_place* the cleaned records replace the source file; otherwise
        they are written to ``output_dir / <relative source path>``.
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

        if not records:
            summary.records_out = 0
            summary.records_preserved = True
            if not read_errors:
                summary.warnings.append(f"{path.name}: the file contains no records.")
            return [], summary

        chunks = dataset_io.chunk_records(records, self.chunk_size, self.max_chunk_chars)
        summary.chunks_total = len(chunks)

        cleaned_records: List[Dict[str, Any]] = []
        removed_by_model = 0
        # Chunks are produced from this file only -> records can never mix files.
        for index, chunk in enumerate(chunks, start=1):
            if on_progress:
                on_progress(
                    {"file_name": path.name, "chunk_index": index, "chunk_total": len(chunks)}
                )
            cleaned_chunk, chunk_summary = await self._clean_chunk(
                chunk=chunk,
                file_name=path.name,
                chunk_index=index,
                chunk_total=len(chunks),
                fmt=fmt,
                use_llm=use_llm,
            )
            removed_by_model += len(chunk) - len(cleaned_chunk)
            summary.chunk_summaries.append(chunk_summary)
            summary.warnings.extend(chunk_summary.warnings)
            if chunk_summary.llm_used:
                summary.chunks_cleaned += 1
            else:
                summary.chunks_fallback += 1
            cleaned_records.extend(cleaned_chunk)

        # Script steps that run in both modes.
        before = len(cleaned_records)
        cleaned_records = [r for r in cleaned_records if not is_junk_record(r)]
        junk = before - len(cleaned_records)
        summary.junk_removed = junk + removed_by_model

        if remove_duplicates:
            cleaned_records, removed = remove_duplicate_records(cleaned_records)
            summary.duplicates_removed = removed

        summary.records_out = len(cleaned_records)
        summary.records_preserved = (
            len(cleaned_records) + summary.duplicates_removed + summary.junk_removed
            == len(records)
        )
        summary.llm_used = any(c.llm_used for c in summary.chunk_summaries)

        if not summary.records_preserved:
            summary.errors.append(
                f"Record count mismatch for {path.name}: "
                f"{len(records)} in, {len(cleaned_records)} kept, "
                f"{summary.duplicates_removed + summary.junk_removed} removed"
            )
            return cleaned_records, summary  # never write a file whose count is off

        if not cleaned_records:
            summary.warnings.append(
                f"{path.name}: every record was empty or a duplicate, so the file was left unchanged."
            )
            summary.records_out = len(records)
            summary.duplicates_removed = 0
            summary.junk_removed = 0
            return records, summary

        if write_output:
            if in_place:
                summary.changed = _write_in_place(path, cleaned_records, fmt)
            else:
                out_path = Path(output_dir) / relative_path
                _write_copy(path, out_path, cleaned_records, fmt)
                summary.changed = True
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
                    if not _is_flagged_for_removal(cleaned)
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

        flagged = sum(1 for item in candidates if _is_flagged_for_removal(item))
        if flagged * 2 > len(chunk):
            warnings.append(
                f"Chunk {chunk_index}: the model wanted to remove {flagged} of {len(chunk)} "
                "records, which is too many to trust; scripts were used for this chunk."
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
        value. A value the model blanked out is not accepted either, so the AI
        can fix text but never erase it.
        """
        merged = dict(original)
        for key, value in cleaned.items():
            if key not in original or key == REMOVE_FLAG:
                continue
            if _is_blank(value) and not _is_blank(original[key]):
                merged[key] = _clean_value(original[key])
                continue
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


def remove_duplicate_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop exact duplicate records, keeping the first copy and the order.

    Two records are duplicates when they hold the same keys and values (key
    order is ignored; surrounding whitespace is ignored).  Works across chunk borders,
    which the per-chunk LLM requests cannot do.
    """
    seen = set()
    kept: List[Dict[str, Any]] = []
    for record in records:
        # Compare the script-cleaned form, so copies that differ only by stray
        # whitespace or control characters count as duplicates in AI mode too
        # (the model's output is not guaranteed to be whitespace-normalised).
        try:
            key = json.dumps(_clean_value(record), sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            key = repr(record)
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept, len(records) - len(kept)


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


REMOVE_FLAG = "_remove"


def _is_flagged_for_removal(item: Any) -> bool:
    return isinstance(item, dict) and item.get(REMOVE_FLAG) is True


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


_PROMPT_KEYS = ("prompt", "instruction", "question", "input")
_RESPONSE_KEYS = ("response", "completion", "output", "answer", "text")


def _has_content(text: str) -> bool:
    """True when *text* contains at least one letter or digit."""
    return bool(re.search(r"\w", text))


def _leaf_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaf_strings(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _leaf_strings(item)]
    return []


def is_junk_record(record: Dict[str, Any]) -> bool:
    """True for a record a script can safely call meaningless.

    That is: the prompt or the response field (when the record has one) is
    empty or holds no letter or digit at all, or the whole record holds no
    text and no number. Records with another layout (for example chat
    ``messages``) are only dropped when they are completely empty.
    """
    for keys in (_PROMPT_KEYS, _RESPONSE_KEYS):
        for key in keys:
            if key in record and isinstance(record[key], str) and not _has_content(record[key]):
                return True
    leaves = _leaf_strings(record)
    has_number = any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in record.values())
    return not has_number and not any(_has_content(leaf) for leaf in leaves)


def _json_wrapper(path: Path) -> Optional[Dict[str, Any]]:
    """The ``{"data": [...]}`` object of a .json source, so it keeps its shape."""
    if path.suffix.lower() != ".json":
        return None
    try:
        data = json.loads(dataset_io.read_text(path))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data
    return None


def _write_in_place(path: Path, records: List[Dict[str, Any]], fmt: str) -> bool:
    """Replace *path* with *records*; True when its content changed."""
    wrapper = _json_wrapper(path)
    if wrapper is None:
        return dataset_io.write_records_atomic(path, records, fmt)
    tmp = path.with_name(f".{path.name}.tmp-write")
    try:
        tmp.write_text(json.dumps({**wrapper, "data": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        if path.read_bytes() == tmp.read_bytes():
            return False
        os.replace(tmp, path)
        return True
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_copy(source: Path, target: Path, records: List[Dict[str, Any]], fmt: str) -> None:
    wrapper = _json_wrapper(source)
    if wrapper is None:
        dataset_io.write_records(target, records, fmt)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**wrapper, "data": records}, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# Separate-folder copies: list, merge into the originals, discard
# ----------------------------------------------------------------------


def cleaned_copies_dir(project_root: str) -> Path:
    return resolve_output_dir(project_root, None, DEFAULT_OUTPUT_DIR)


def list_cleaned_copies(project_root: str) -> List[str]:
    """Project-relative paths of cleaned copies waiting to be merged."""
    folder = cleaned_copies_dir(project_root)
    if not folder.is_dir():
        return []
    return sorted(
        f.relative_to(folder).as_posix()
        for f in folder.rglob("*")
        if f.is_file() and dataset_io.is_dataset_file(f) and f.name != LEGACY_MANIFEST_NAME
    )


def _remove_internal_folder(folder: Path) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    parent = folder.parent
    try:
        if parent.name == dataset_io.INTERNAL_DIR_NAME and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


def merge_cleaned_copies(project_root: str) -> List[str]:
    """Replace the originals with the cleaned copies, then delete the copies folder."""
    root = Path(project_root).resolve()
    folder = cleaned_copies_dir(project_root)
    merged: List[str] = []
    for rel in list_cleaned_copies(project_root):
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(folder / rel, target)
        merged.append(rel)
    _remove_internal_folder(folder)
    return merged


def discard_cleaned_copies(project_root: str) -> int:
    """Delete the cleaned copies without touching the originals."""
    folder = cleaned_copies_dir(project_root)
    count = len(list_cleaned_copies(project_root))
    _remove_internal_folder(folder)
    return count
