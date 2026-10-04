"""Dataset file discovery, reading, writing and chunking utilities.

This module centralises every filesystem concern of the dataset pipeline so
that both :class:`analyzers.dataset_analyzer.DatasetAnalyzer` and
:class:`cleaning.dataset_cleaner.DatasetCleaner` use exactly the same parsing
rules and never diverge.

Design notes
------------
* Reading never raises for a single bad file: problems are collected into an
  ``errors`` list so that a multi-file run keeps going.
* Chunking is deterministic (sequential, order preserving) so that a chunk can
  always be mapped back to its exact position in the source file.  This is what
  makes a lossless reconstruction of a cleaned file possible.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from core.security import resolve_dataset_path as _resolve_workspace_dataset_path

logger = logging.getLogger(__name__)

# Formats the pipeline can read and write.
SUPPORTED_SUFFIXES: Tuple[str, ...] = (".jsonl", ".json", ".csv", ".tsv", ".txt")

# Recognised dataset formats that need an optional dependency to parse.
UNSUPPORTED_SUFFIXES: Tuple[str, ...] = (".parquet",)

#: Encoding used to read files that belong to the user's project. ``utf-8-sig``
#: behaves exactly like ``utf-8`` except that it transparently strips a leading
#: UTF-8 BOM. Windows editors (Notepad, VS Code), Excel, and PowerShell's
#: ``Set-Content -Encoding UTF8`` all write that BOM by default, and a stray BOM
#: makes ``json.loads`` fail outright ("Unexpected UTF-8 BOM") and makes the first
#: CSV/TSV column name unmatchable. A user's real dataset must never be rejected
#: over a byte-order marker that carries no data meaning.
PROJECT_FILE_ENCODING = "utf-8-sig"


def read_text(path: Path) -> str:
    """Read a project text file, tolerating a UTF-8 BOM.

    This is the single reader used for every user-supplied text file (datasets,
    configs, prompt templates), so no code path can regress to strict ``utf-8``
    and reject a BOM-prefixed file.
    """
    return path.read_text(encoding=PROJECT_FILE_ENCODING)


# Guard against loading enormous files into memory in a single shot.
MAX_FILE_BYTES = 200 * 1024 * 1024


def is_dataset_file(path: Path) -> bool:
    """Return True when *path* is a dataset format handled by this module."""
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def detect_format(path: Path) -> str:
    """Return the canonical format name for *path* (``jsonl``, ``json`` ...)."""
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "jsonl"


def read_records(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read a dataset file and return ``(records, errors)``.

    Supported formats: JSONL, JSON (array or ``{"data": [...]}``), CSV, TSV and
    plain text (alternating prompt/response lines).  Errors are reported instead
    of raised so that a batch run can continue across files.
    """
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    suffix = path.suffix.lower()

    try:
        if suffix == ".jsonl":
            for i, line in enumerate(read_text(path).splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i + 1}: {e}")
                    continue
                if isinstance(item, dict):
                    records.append(item)
                else:
                    errors.append(
                        f"Line {i + 1}: expected a JSON object, got {type(item).__name__}"
                    )
        elif suffix == ".csv":
            with open(path, newline="", encoding=PROJECT_FILE_ENCODING) as f:
                for row in csv.DictReader(f):
                    records.append(dict(row))
        elif suffix == ".tsv":
            with open(path, newline="", encoding=PROJECT_FILE_ENCODING) as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    records.append(dict(row))
        elif suffix == ".json":
            data = json.loads(read_text(path))
            if isinstance(data, list):
                records = [item for item in data if isinstance(item, dict)]
                if len(records) != len(data):
                    errors.append("Some JSON array entries were not objects and were skipped")
            elif isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    records = [item for item in data["data"] if isinstance(item, dict)]
                else:
                    records = [data]
        elif suffix == ".txt":
            # Simple text format: alternating prompt/response lines.
            lines = read_text(path).splitlines()
            for i in range(0, len(lines) - 1, 2):
                records.append({"prompt": lines[i], "response": lines[i + 1]})
        elif suffix in UNSUPPORTED_SUFFIXES:
            errors.append(
                f"'{suffix}' parsing requires an optional dependency that is not installed"
            )
        else:
            errors.append(f"Unsupported dataset format: '{suffix}'")
    except Exception as e:  # noqa: BLE001 - reported, never raised, by design
        errors.append(f"Failed to read {path.name}: {e}")

    return records, errors


def _fieldnames(records: List[Dict[str, Any]]) -> List[str]:
    """Return the union of record keys in first-seen order (CSV/TSV headers)."""
    names: List[str] = []
    for record in records:
        for key in record.keys():
            if key not in names:
                names.append(key)
    return names


def write_records(path: Path, records: List[Dict[str, Any]], fmt: str | None = None) -> None:
    """Write *records* to *path* using the file format (or explicit *fmt*).

    The destination directory is created when missing.  CSV/TSV headers use the
    union of record keys in first-seen order, so no field is silently dropped.
    """
    fmt = (fmt or detect_format(path)).lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    elif fmt == "json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt in ("csv", "tsv"):
        delimiter = "\t" if fmt == "tsv" else ","
        fieldnames = _fieldnames(records)
        if not fieldnames:
            path.write_text("", encoding="utf-8")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for record in records:
                writer.writerow({key: record.get(key, "") for key in fieldnames})
    elif fmt == "txt":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                prompt = record.get("prompt", record.get("instruction", ""))
                response = record.get("response", record.get("completion", ""))
                f.write(f"{prompt}\n{response}\n")
    else:
        raise ValueError(f"Unsupported output format: '{fmt}'")


def resolve_dataset_path(project_path: str | Path, dataset_path: str) -> Path:
    """Resolve a dataset path strictly inside the project root."""
    return _resolve_workspace_dataset_path(str(project_path), dataset_path)


def discover_dataset_files(
    project_path: str | Path,
    dataset_paths: Iterable[str],
    max_files: int = 0,
) -> Tuple[List[Path], List[str]]:
    """Expand *dataset_paths* into a deterministic list of concrete files.

    Each entry may be a file or a directory; directories are expanded
    recursively using :data:`SUPPORTED_SUFFIXES`.  Returned paths are absolute,
    deduplicated and sorted.  Entries that cannot be used are returned in the
    ``skipped`` list together with the reason.
    """
    root = Path(project_path) if project_path else Path(".")
    discovered: Dict[str, Path] = {}
    skipped: List[str] = []

    for entry in dataset_paths or []:
        resolved = resolve_dataset_path(root, str(entry))
        if resolved.is_dir():
            for suffix in SUPPORTED_SUFFIXES:
                for found in sorted(resolved.rglob(f"*{suffix}")):
                    if found.is_file():
                        discovered.setdefault(str(found.resolve()), found)
        elif resolved.is_file():
            if is_dataset_file(resolved):
                discovered.setdefault(str(resolved.resolve()), resolved)
            else:
                skipped.append(f"{entry}: unsupported dataset format")
        else:
            skipped.append(f"{entry}: path does not exist")

    files: List[Path] = []
    for key in sorted(discovered.keys()):
        path = discovered[key]
        try:
            size = path.stat().st_size
        except OSError as e:
            skipped.append(f"{path.name}: cannot stat file ({e})")
            continue
        if size > MAX_FILE_BYTES:
            skipped.append(f"{path.name}: larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
            continue
        if max_files and len(files) >= max_files:
            skipped.append(f"{path.name}: max_files limit reached")
            continue
        files.append(path)

    return files, skipped


def chunk_records(
    records: List[Dict[str, Any]],
    chunk_size: int = 25,
    max_chunk_chars: int = 12000,
) -> List[List[Dict[str, Any]]]:
    """Split *records* into sequential, order-preserving chunks.

    A chunk is closed as soon as either limit would be exceeded:

    * ``chunk_size``      - maximum number of records per chunk.
    * ``max_chunk_chars`` - maximum serialised size (characters) per chunk.

    Chunking is performed per file by the caller, so a chunk never contains
    records originating from two different files.  A single record larger than
    *max_chunk_chars* still gets its own chunk (never dropped, never split).
    """
    chunk_size = max(1, int(chunk_size))
    max_chunk_chars = max(200, int(max_chunk_chars))

    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0

    for record in records:
        record_chars = len(json.dumps(record, ensure_ascii=False))
        would_overflow = bool(current) and (
            len(current) >= chunk_size or (current_chars + record_chars) > max_chunk_chars
        )
        if would_overflow:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(record)
        current_chars += record_chars

    if current:
        chunks.append(current)

    return chunks