"""Dataset quality analyzer — reads real dataset files and computes actual statistics."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from models.schemas import ProjectContext, DatasetAnalysisResult
from ai.llm import LLMHelper
from cleaning.dataset_io import read_records

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyzes dataset quality by reading actual dataset files.

    Supports multiple dataset files per project: statistics are aggregated
    across every discovered dataset file. Per-file reading is capped to keep
    memory usage bounded for large datasets (with an explicit truncation
    warning in the findings). All metrics are computed deterministically.
    """

    # Analysis is read-only and bounded: never read more records than this
    # per file, and never aggregate more than this in total.
    MAX_RECORDS_PER_FILE = 50_000
    MAX_TOTAL_RECORDS = 100_000
    MAX_FILES = 20

    def __init__(self):
        self._llm = LLMHelper()

    async def analyze(
        self,
        context: ProjectContext,
        dataset_path: Optional[str] = None,
    ) -> DatasetAnalysisResult:
        """Analyze the project's dataset files for quality, duplicates, and consistency."""
        # Resolve the dataset file(s) to analyze.
        if dataset_path:
            dataset_files: List[Path] = [Path(dataset_path) if Path(dataset_path).is_absolute() else Path(context.project_path) / dataset_path]
        else:
            entries = context.dataset_paths or []
            if not entries:
                raise ValueError("No dataset path provided")
            dataset_files = []
            for entry in entries:
                p = Path(entry) if Path(entry).is_absolute() else Path(context.project_path) / entry
                if p.is_dir():
                    # Expand directories deterministically (same rules as the cleaning pipeline)
                    found = sorted(
                        child for child in p.rglob("*")
                        if child.is_file() and child.suffix.lower() in (".jsonl", ".json", ".csv", ".tsv", ".txt")
                    )
                    dataset_files.extend(found)
                else:
                    dataset_files.append(p)

        existing_files = [f for f in dataset_files if f.exists()]
        missing_files = [str(f) for f in dataset_files if not f.exists()]
        if not existing_files:
            raise FileNotFoundError(f"Dataset file(s) not found: {missing_files or [str(p) for p in dataset_files]}")

        skipped_oversized: List[str] = []
        if len(existing_files) > self.MAX_FILES:
            skipped_oversized = [str(f) for f in existing_files[self.MAX_FILES:]]
            existing_files = existing_files[: self.MAX_FILES]

        logger.info(f"Analyzing {len(existing_files)} dataset file(s): {[f.name for f in existing_files]}")

        # Step 1: Read and parse every dataset file (bounded, deterministic).
        records: List[Dict[str, Any]] = []
        parse_errors: List[str] = []
        empty_files: List[str] = []
        truncated_files: List[str] = []

        for path in existing_files:
            file_records, file_errors = self._read_dataset(path)
            parse_errors.extend(f"{path.name}: {err}" for err in file_errors)
            if not file_records:
                empty_files.append(path.name)
                continue
            if len(file_records) > self.MAX_RECORDS_PER_FILE:
                file_records = file_records[: self.MAX_RECORDS_PER_FILE]
                truncated_files.append(path.name)
            remaining = self.MAX_TOTAL_RECORDS - len(records)
            if remaining <= 0:
                break
            if len(file_records) > remaining:
                truncated_files.append(path.name)
                file_records = file_records[:remaining]
            records.extend(file_records)

        sample_count = len(records)
        if sample_count == 0:
            # Nothing could be read — report this explicitly. No statistics
            # are invented: every score is 0 and the confidence is minimal.
            findings = ["No readable records were found in any dataset file."]
            if empty_files:
                findings.append(f"Empty or unparseable file(s): {', '.join(empty_files)}")
            for err in parse_errors[:5]:
                findings.append(f"  - {err}")
            return DatasetAnalysisResult(
                dataset_name=", ".join(f.name for f in existing_files),
                sample_count=0,
                token_count=0,
                average_prompt_length=0.0,
                average_response_length=0.0,
                duplicate_percentage=0.0,
                near_duplicate_percentage=0.0,
                missing_field_percentage=0.0,
                formatting_consistency_score=0.0,
                language_consistency_score=0.0,
                instruction_consistency_score=0.0,
                response_consistency_score=0.0,
                quality_score=0.0,
                findings=findings,
                warnings=["Dataset is empty or could not be parsed — quality metrics are unavailable"],
                recommendations=["Provide a valid dataset file (JSONL, JSON, CSV, TSV or TXT)"],
                confidence="very_low",
            )

        # Step 2: Compute deterministic statistics
        # Measured from the *resolved* prompt/response text, not from a fixed list
        # of field names: counting only "text"/"prompt"/"completion" reported ~0
        # tokens for instruction/input/output datasets.
        extracted = [self._extract_text(r) for r in records]
        total_chars = sum(
            len(t["prompt"]) + len(t["response"]) for t in extracted
        )
        token_count = int(total_chars / 4)  # Rough estimate: ~4 chars per token

        prompt_lengths = [len(t["prompt"]) for t in extracted]
        response_lengths = [len(t["response"]) for t in extracted]

        avg_prompt = float(sum(prompt_lengths) / len(prompt_lengths)) if prompt_lengths else 0.0
        avg_response = float(sum(response_lengths) / len(response_lengths)) if response_lengths else 0.0

        # Step 3: Detect duplicates (exact match on instruction+response hash)
        duplicate_percentage, near_duplicate_percentage = self._detect_duplicates(records)

        # Step 4: Check for missing fields
        missing_field_percentage = self._detect_missing_fields(records)

        # Step 5: Consistency scores (deterministic heuristics)
        formatting_score = self._check_formatting_consistency(records)
        language_score = self._check_language_consistency(records)
        instruction_score = self._check_instruction_consistency(records)
        response_score = self._check_response_consistency(records)

        # Step 6: Quality score (weighted combination of deterministic metrics)
        quality_score = self._compute_quality_score(
            duplicate_percentage,
            near_duplicate_percentage,
            missing_field_percentage,
            formatting_score,
            language_score,
            instruction_score,
            response_score,
            sample_count,
        )

        # Step 7: Generate findings, warnings, recommendations
        findings, warnings, recommendations = self._generate_findings(
            sample_count, duplicate_percentage, near_duplicate_percentage,
            missing_field_percentage, formatting_score, quality_score, parse_errors
        )

        # Step 8: Optionally call LLM for quality assessment
        confidence = "high" if sample_count > 100 else "medium" if sample_count > 10 else "low"

        dataset_name = (
            existing_files[0].name
            if len(existing_files) == 1
            else f"{len(existing_files)} files ({', '.join(f.name for f in existing_files[:3])}{'...' if len(existing_files) > 3 else ''})"
        )
        if truncated_files:
            warnings_extra = [f"Analysis capped at {self.MAX_RECORDS_PER_FILE} records per file / {self.MAX_TOTAL_RECORDS} total — truncated: {', '.join(sorted(set(truncated_files)))}"]
        else:
            warnings_extra = []
        if missing_files:
            warnings_extra.append(f"Dataset path(s) not found and skipped: {', '.join(missing_files)}")
        if empty_files and sample_count > 0:
            warnings_extra.append(f"File(s) with no readable records: {', '.join(empty_files)}")
        if skipped_oversized:
            warnings_extra.append(f"File limit ({self.MAX_FILES}) reached — skipped: {', '.join(Path(p).name for p in skipped_oversized)}")

        # If LLM is available, enhance the analysis (a deterministic sample, not
        # the whole dataset, is sent to the LLM).
        if self._llm.is_available:
            llm_result = await self._llm_enhance(existing_files[0], records[:5], {
                "sample_count": sample_count,
                "duplicate_percentage": duplicate_percentage,
                "quality_score": quality_score,
                "formatting_consistency_score": formatting_score,
                "language_consistency_score": language_score,
            })
            if llm_result:
                confidence = "high"
                # Merge LLM findings with deterministic findings
                findings.extend(llm_result.get("findings", []))
                warnings.extend(llm_result.get("warnings", []))

        warnings.extend(warnings_extra)

        return DatasetAnalysisResult(
            dataset_name=dataset_name,
            sample_count=sample_count,
            token_count=token_count,
            average_prompt_length=avg_prompt,
            average_response_length=avg_response,
            duplicate_percentage=duplicate_percentage,
            near_duplicate_percentage=near_duplicate_percentage,
            missing_field_percentage=missing_field_percentage,
            formatting_consistency_score=formatting_score,
            language_consistency_score=language_score,
            instruction_consistency_score=instruction_score,
            response_consistency_score=response_score,
            quality_score=quality_score,
            findings=findings,
            warnings=warnings,
            recommendations=recommendations,
            confidence=confidence,
        )

    def _read_dataset(self, path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
        """Read a dataset file (JSONL, JSON, CSV, TSV or TXT) and return records.

        Delegates to the shared dataset I/O layer so that analysis and cleaning
        always interpret a dataset file in exactly the same way.
        """
        return read_records(path)

    def _extract_text(self, record: Dict[str, Any]) -> Dict[str, str]:
        """Extract prompt/response text from a record, handling different field names.

        Alpaca/ShareGPT-style records split the prompt across ``instruction`` and
        ``input`` (``question`` + ``context`` is the other common pair). Returning
        only the first of the two made every record sharing one instruction look
        identical, which inflated duplicate rates and collapsed token counts. Both
        halves are joined so each record is described by all of its own text.
        """
        prompt = record.get("prompt") or record.get("question") or record.get("instruction") or ""
        # Only add the companion field when the primary one did not already carry
        # it: "prompt" is usually self-contained, while "instruction" is not.
        if not record.get("prompt"):
            extra = record.get("input") if record.get("instruction") else record.get("context")
            if extra:
                prompt = f"{prompt}\n{extra}" if prompt else str(extra)
        response = record.get("response") or record.get("completion") or record.get("output") or record.get("answer") or record.get("text") or ""
        return {"prompt": str(prompt), "response": str(response)}

    def _detect_duplicates(self, records: List[Dict[str, Any]]) -> tuple[float, float]:
        """Detect exact and near-duplicate records using hashing.

        Exact duplicates are records whose prompt *and* response are identical.
        Near-duplicates share a prompt but differ in the response, which is the
        case that actually needs a human decision (contradictory labels).

        The two checks keep separate hash sets. Sharing one set made a prompt
        hash collide with an unrelated full-text hash and made every exact
        duplicate also count as a near-duplicate, so the two percentages could
        not be compared.
        """
        seen_full: Set[str] = set()
        seen_prompts: Set[str] = set()
        duplicate_count = 0
        near_duplicate_count = 0

        for r in records:
            text = self._extract_text(r)

            # Exact duplicate
            h = hashlib.md5((text["prompt"] + "|" + text["response"]).encode("utf-8")).hexdigest()
            if h in seen_full:
                duplicate_count += 1
            else:
                seen_full.add(h)

            # Near-duplicate: same prompt seen before but a different response.
            if text["prompt"]:
                prompt_hash = hashlib.md5(text["prompt"].encode("utf-8")).hexdigest()
                if prompt_hash in seen_prompts:
                    if h not in seen_full:
                        near_duplicate_count += 1
                else:
                    seen_prompts.add(prompt_hash)

        total = len(records) if records else 1
        return (duplicate_count / total * 100, near_duplicate_count / total * 100)

    def _detect_missing_fields(self, records: List[Dict[str, Any]]) -> float:
        """Calculate percentage of records with missing or empty fields."""
        if not records:
            return 0.0
        missing_count = 0
        for r in records:
            text = self._extract_text(r)
            if not text["prompt"].strip() or not text["response"].strip():
                missing_count += 1
        return missing_count / len(records) * 100

    def _check_formatting_consistency(self, records: List[Dict[str, Any]]) -> float:
        """Check if responses follow consistent formatting."""
        if not records:
            return 0.0
        scores = []
        for r in records:
            text = self._extract_text(r)
            response = text["response"]
            # Check for consistent structure (e.g., starts with capital, ends with period)
            score = 1.0
            if response and not response[0].isupper():
                score -= 0.3
            if response and response.rstrip()[-1:] not in (".", "!", "?"):
                score -= 0.2
            scores.append(max(0.0, score))
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _check_language_consistency(self, records: List[Dict[str, Any]]) -> float:
        """Check if all responses use the same primary language."""
        if not records:
            return 0.0
        # Simple heuristic: detect common non-English characters
        non_english_count = 0
        for r in records:
            text = self._extract_text(r)
            combined = text["prompt"] + text["response"]
            # Count non-ASCII characters (rough proxy for non-English)
            non_ascii = sum(1 for c in combined if ord(c) > 127)
            if non_ascii > len(combined) * 0.1:
                non_english_count += 1
        return float(1.0 - (non_english_count / len(records)))

    def _check_instruction_consistency(self, records: List[Dict[str, Any]]) -> float:
        """Check if instructions are consistent in style."""
        if not records:
            return 0.0
        # Check if instructions have common structure (e.g., "You are a...")
        consistent_count = 0
        for r in records:
            text = self._extract_text(r)
            prompt = text["prompt"]
            if prompt and len(prompt) > 10:
                consistent_count += 1
        return float(consistent_count / len(records)) if records else 0.0

    def _check_response_consistency(self, records: List[Dict[str, Any]]) -> float:
        """Check if responses are consistently formatted."""
        if not records:
            return 0.0
        scores = []
        for r in records:
            text = self._extract_text(r)
            response = text["response"]
            score = 1.0
            if not response.strip():
                score = 0.0
            elif len(response) < 20:
                score = 0.5
            scores.append(score)
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _compute_quality_score(
        self, dup_pct, near_dup_pct, missing_pct,
        fmt_score, lang_score, inst_score, resp_score, sample_count
    ) -> float:
        """Compute overall quality score (0-1)."""
        # Penalize duplicates
        dup_penalty = min(1.0, dup_pct / 10.0)  # 10% duplicates = full penalty
        near_dup_penalty = min(0.5, near_dup_pct / 20.0)
        missing_penalty = min(0.5, missing_pct / 5.0)

        # Average of consistency scores
        consistency = (fmt_score + lang_score + inst_score + resp_score) / 4.0

        # Small datasets get proportionally less confidence in the score, but
        # the score is dampened rather than zeroed so that a small but clean
        # dataset is never reported as "quality 0.0".
        size_factor = min(1.0, sample_count / 1000.0) if sample_count < 1000 else 1.0
        size_dampening = 0.7 + 0.3 * size_factor  # 0.70..1.00 multiplier

        score = (consistency * size_dampening) - dup_penalty - near_dup_penalty - missing_penalty
        return max(0.0, min(1.0, score))

    def _generate_findings(
        self, sample_count, dup_pct, near_dup_pct, missing_pct,
        fmt_score, quality_score, parse_errors
    ) -> tuple[List[str], List[str], List[str]]:
        """Generate deterministic findings, warnings, and recommendations."""
        findings = [f"Dataset contains {sample_count} samples."]

        warnings = []
        recommendations = []

        if dup_pct > 0:
            findings.append(f"{dup_pct:.1f}% exact duplicate samples detected.")
            recommendations.append("Remove duplicate samples to improve training efficiency.")

        if near_dup_pct > 5:
            warnings.append(f"{near_dup_pct:.1f}% near-duplicate samples (same prompt, different response).")
            recommendations.append("Consolidate near-duplicates to reduce data noise.")

        if missing_pct > 0:
            warnings.append(f"{missing_pct:.1f}% of samples have missing or empty fields.")
            recommendations.append("Review and fill in missing fields or remove incomplete samples.")

        if fmt_score < 0.8:
            warnings.append(f"Formatting consistency is low ({fmt_score:.0%}).")
            recommendations.append("Apply consistent formatting (capitalization, punctuation) across all responses.")

        if parse_errors:
            warnings.append(f"{len(parse_errors)} parsing errors encountered.")
            for err in parse_errors[:3]:
                warnings.append(f"  - {err}")

        if sample_count < 50:
            warnings.append(f"Small dataset ({sample_count} samples). Consider collecting more data.")

        if quality_score < 0.5:
            recommendations.append("Dataset quality is low. Consider collecting higher-quality training data.")
        elif quality_score > 0.8:
            findings.append("Dataset quality is good.")

        return findings, warnings, recommendations

    async def _llm_enhance(self, path: Path, sample_records: List[Dict], stats: Dict[str, Any]) -> Optional[Dict]:
        """Use LLM to enhance dataset analysis with quality assessment."""
        try:
            sample_text = "\n\n".join(
                json.dumps(r, ensure_ascii=False) for r in sample_records[:5]
            )
            prompt = f"""Analyze the following dataset samples and quality statistics.
Provide a JSON response with: quality_score (0-1), findings (list), warnings (list), recommendations (list), confidence (very_high|high|medium|low|very_low).

Dataset: {path.name}
Statistics:
- Sample count: {stats['sample_count']}
- Duplicate percentage: {stats['duplicate_percentage']:.1f}%
- Quality score (preliminary): {stats['quality_score']:.2f}
- Formatting consistency: {stats['formatting_consistency_score']:.2f}
- Language consistency: {stats['language_consistency_score']:.2f}

Sample records (first 5):
{sample_text}

Base your analysis ONLY on the provided data. Do not fabricate statistics."""
            return await self._llm.call_structured(prompt, temperature=0.3)
        except Exception as e:
            logger.debug(f"LLM enhancement failed: {e}")
            return None
