"""Use-case detector — infers the task from real project evidence.

The detector answers "what is this project actually trying to do?" without
asking the user and without guessing from the project *name*.  Evidence is
collected from, in order of reliability:

  1. the **shape of the dataset** (field names + a bounded sample of values) —
     this is the strongest signal, because it is measured, not assumed;
  2. the **output distribution** of those values (is it a closed label set? a
     long passage? a short fact? a code block?);
  3. **prompt templates** actually present in the project;
  4. **project structure** (training/eval/inference scripts, config files);
  5. the **declared model** (an embedding model implies a retrieval use case).

Every emitted signal carries a ``provenance`` and the evidence string that
produced it, so the report can separate a measured fact from a keyword
heuristic.  When signals disagree or are too weak, the confidence drops and
``uncertainties`` states exactly what could not be resolved — the detector
never manufactures a confident answer from thin evidence.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models.schemas import ProjectContext, UseCaseAnalysisResult, UseCaseSignal
from cleaning import dataset_io
from cleaning.dataset_io import read_records

logger = logging.getLogger(__name__)


class UseCaseDetector:
    """Infers the project's use case from dataset shape, prompts and structure."""

    #: Bounded sample size — enough to characterise the data, cheap for large sets.
    SAMPLE_SIZE = 200
    #: Bounded number of dataset files inspected for field-shape evidence.
    MAX_FILES = 5
    #: A closed label set at or below this size looks like classification.
    LABEL_SET_MAX = 25
    #: Character count separating a short answer from a long generated passage.
    LONG_OUTPUT_CHARS = 1200

    # --- field-name signals (measured: these fields actually exist) ----------
    _FIELD_SIGNALS: List[Tuple[Tuple[str, ...], str, float]] = [
        (("question", "answer"), "question_answering", 0.9),
        (("qa_pairs",), "question_answering", 0.8),
        (("context", "answer"), "question_answering", 0.7),
        (("article", "highlights"), "summarization", 0.9),
        (("summary",), "summarization", 0.8),
        (("text", "summary"), "summarization", 0.8),
        (("document", "summary"), "summarization", 0.7),
        (("label",), "classification", 0.85),
        (("labels",), "classification", 0.8),
        (("category",), "classification", 0.8),
        (("sentiment",), "classification", 0.75),
        (("intent",), "classification", 0.7),
        (("sql",), "structured_generation", 0.8),
        (("output_schema",), "structured_generation", 0.7),
        (("messages",), "chatbot", 0.85),
        (("conversation",), "chatbot", 0.85),
        (("chat",), "chatbot", 0.7),
        (("code",), "code_generation", 0.7),
        (("function",), "code_generation", 0.6),
        (("embedding",), "embedding_search", 0.8),
        (("query", "result"), "retrieval", 0.6),
        (("ticket",), "customer_support", 0.7),
    ]

    # --- output-value signals (measured: computed from the sampled records) ---
    _CODE_SYNTAX = re.compile(r"```|^\s*(def |class |import |from \w+ import|function |const |SELECT )", re.M | re.I)
    _JSON_LIKE = re.compile(r"^\s*[\[{].*[\]}]\s*$", re.S)
    _URL = re.compile(r"https?://", re.I)

    def analyze(
        self,
        context: ProjectContext,
        dataset_result: Optional[Any] = None,
    ) -> UseCaseAnalysisResult:
        """Infer the use case from the project's real dataset, prompts and files."""
        signals: List[UseCaseSignal] = []
        uncertainties: List[str] = []
        notes: List[str] = []

        records, field_evidence, field_uncertainties = self._inspect_dataset(context)
        signals.extend(field_evidence)
        uncertainties.extend(field_uncertainties)

        shape_evidence, shape_uncertainties, notes = self._inspect_output_shape(records, notes)
        signals.extend(shape_evidence)
        uncertainties.extend(shape_uncertainties)

        signals.extend(self._inspect_prompts(context))
        signals.extend(self._inspect_structure(context))

        if not records and not context.prompt_templates:
            uncertainties.append(
                "No readable dataset and no prompt template were found, so the use case "
                "could not be inferred from project content."
            )

        # When the DatasetAnalyzer did not run, fall back to the records this
        # detector read itself, so the approach recommendation is never based on
        # a "0 samples" assumption that was never actually measured.
        sample_count = int(getattr(dataset_result, "sample_count", 0) or 0) if dataset_result else 0
        if not sample_count and records:
            sample_count = len(records)
        return self._conclude(signals, uncertainties, notes, context, sample_count)

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------
    def _inspect_dataset(
        self, context: ProjectContext
    ) -> Tuple[List[Dict[str, Any]], List[UseCaseSignal], List[str]]:
        """Read a bounded sample of records and derive field-shape evidence."""
        signals: List[UseCaseSignal] = []
        uncertainties: List[str] = []
        records: List[Dict[str, Any]] = []

        try:
            candidates = [
                Path(p) if Path(p).is_absolute() else Path(context.project_path) / p
                for p in (context.dataset_paths or [])
            ]
        except (TypeError, ValueError, OSError):
            return records, signals, ["Dataset paths could not be resolved."]

        files = [p for p in candidates if p.is_file()][: self.MAX_FILES]
        if not files:
            if candidates:
                uncertainties.append(
                    "Dataset paths were declared but no readable dataset file was found."
                )
            return records, signals, uncertainties

        for path in files:
            try:
                file_records, errors = read_records(path)
            except Exception as error:  # pragma: no cover - defensive
                logger.warning(f"Use-case detector could not read {path.name}: {error}")
                uncertainties.append(f"Could not read dataset file {path.name}: {error}")
                continue
            if not file_records:
                uncertainties.append(f"Dataset file {path.name} contained no readable records.")
                continue
            if errors:
                uncertainties.append(f"{path.name}: {len(errors)} record(s) could not be parsed.")
            records.extend(file_records[: self.SAMPLE_SIZE])

        if not records:
            return records, signals, uncertainties

        keys = {str(k).lower() for record in records for k in record.keys()}
        signals.append(UseCaseSignal(
            task="_dataset_schema",
            weight=0.0,
            provenance="measured",
            evidence=f"Dataset fields present: {', '.join(sorted(keys))}",
        ))

        # Multi-turn chat formats nest the turns; include them in the key space.
        nested_keys = set()
        for record in records:
            for value in record.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    nested_keys = {str(k).lower() for item in value for k in item.keys()}
                    break
            if nested_keys:
                break
        if nested_keys:
            keys |= nested_keys
            signals.append(UseCaseSignal(
                task="_dataset_schema", weight=0.0, provenance="measured",
                evidence=f"Nested turn objects contain field(s): {', '.join(sorted(nested_keys))}",
            ))

        for expected, task, weight in self._FIELD_SIGNALS:
            if all(field in keys for field in expected):
                signals.append(UseCaseSignal(
                    task=task,
                    weight=weight,
                    provenance="measured",
                    evidence=f"Dataset contains field(s): {', '.join(expected)}",
                ))

        return records, signals, uncertainties

    def _inspect_output_shape(
        self, records: List[Dict[str, Any]], notes: List[str]
    ) -> Tuple[List[UseCaseSignal], List[str], List[str]]:
        """Derive task signals from the *measured* shape of the output values."""
        signals: List[UseCaseSignal] = []
        uncertainties: List[str] = []

        outputs = [text for text in (self._output_text(r) for r in records) if text]
        if not outputs:
            uncertainties.append("No output/response values were found in the sampled records.")
            return signals, uncertainties, notes

        total = len(outputs)
        lengths = sorted(len(text) for text in outputs)
        distinct = {text.strip().lower() for text in outputs}
        notes.append(
            f"Measured output values: {total} sampled, {len(distinct)} distinct, "
            f"median length {lengths[total // 2]} characters."
        )

        code_like = sum(1 for text in outputs if self._CODE_SYNTAX.search(text))
        json_like = sum(1 for text in outputs if self._JSON_LIKE.match(text.strip()))
        short = sum(1 for text in outputs if len(text) <= 80)
        long_form = sum(1 for text in outputs if len(text) > self.LONG_OUTPUT_CHARS)

        if code_like / total >= 0.5:
            signals.append(UseCaseSignal(
                task="code_generation", weight=0.85, provenance="measured",
                evidence=f"{code_like}/{total} sampled outputs contain code syntax or code fences",
            ))
        if json_like / total >= 0.5:
            signals.append(UseCaseSignal(
                task="structured_generation", weight=0.8, provenance="measured",
                evidence=f"{json_like}/{total} sampled outputs are JSON/object literals",
            ))
        if len(distinct) <= self.LABEL_SET_MAX and total >= 10:
            signals.append(UseCaseSignal(
                task="classification", weight=0.9, provenance="calculated",
                evidence=(
                    f"Outputs collapse to {len(distinct)} distinct values across {total} "
                    "samples — a closed label set"
                ),
            ))
        if long_form / total >= 0.5:
            signals.append(UseCaseSignal(
                task="summarization", weight=0.7, provenance="measured",
                evidence=(
                    f"{long_form}/{total} sampled outputs exceed "
                    f"{self.LONG_OUTPUT_CHARS} characters"
                ),
            ))
        elif short / total >= 0.8 and len(distinct) > self.LABEL_SET_MAX:
            signals.append(UseCaseSignal(
                task="question_answering", weight=0.6, provenance="measured",
                evidence=(
                    f"{short}/{total} sampled outputs are short (≤80 characters) and highly varied"
                ),
            ))
        if long_form and any(self._URL.search(text) for text in outputs):
            signals.append(UseCaseSignal(
                task="retrieval", weight=0.35, provenance="heuristic",
                evidence="Some sampled outputs cite URLs alongside long-form content",
            ))
        return signals, uncertainties, notes

    def _inspect_prompts(self, context: ProjectContext) -> List[UseCaseSignal]:
        """Derive weak signals from prompt templates that actually exist."""
        signals: List[UseCaseSignal] = []
        for entry in (context.prompt_templates or [])[:3]:
            path = Path(entry) if Path(entry).is_absolute() else Path(context.project_path) / entry
            try:
                text = path.read_text(encoding=dataset_io.PROJECT_FILE_ENCODING, errors="ignore")[:4000].lower()
            except OSError:
                continue
            if not text.strip():
                continue
            if "summar" in text:
                signals.append(UseCaseSignal(
                    task="summarization", weight=0.35, provenance="measured",
                    evidence=f"Prompt template {path.name} asks for summarization",
                ))
            if "classif" in text or "categor" in text or "label" in text:
                signals.append(UseCaseSignal(
                    task="classification", weight=0.35, provenance="measured",
                    evidence=f"Prompt template {path.name} asks for a label/category",
                ))
            if "json" in text or "schema" in text:
                signals.append(UseCaseSignal(
                    task="structured_generation", weight=0.3, provenance="measured",
                    evidence=f"Prompt template {path.name} specifies a structured output format",
                ))
        return signals

    def _inspect_structure(self, context: ProjectContext) -> List[UseCaseSignal]:
        """Derive signals from the project's scripts and declared model."""
        signals: List[UseCaseSignal] = []

        if "embed" in (context.base_model or "").lower():
            signals.append(UseCaseSignal(
                task="embedding_search", weight=0.9, provenance="measured",
                evidence=f"Declared base model is an embedding model: {context.base_model}",
            ))

        inference = [Path(p).name for p in (context.inference_scripts or [])]
        if any("rag" in name.lower() or "retriev" in name.lower() for name in inference):
            signals.append(UseCaseSignal(
                task="retrieval", weight=0.5, provenance="measured",
                evidence=f"Inference script suggests retrieval: {', '.join(inference)}",
            ))
        if context.evaluation_scripts:
            signals.append(UseCaseSignal(
                task="_evaluation_present", weight=0.0, provenance="measured",
                evidence=(
                    "Evaluation code present: "
                    + ", ".join(Path(p).name for p in context.evaluation_scripts)
                ),
            ))
        if context.detected_framework:
            signals.append(UseCaseSignal(
                task="_framework", weight=0.0, provenance="measured",
                evidence=f"Detected training framework: {context.detected_framework}",
            ))
        return signals

    def _conclude(
        self,
        signals: List[UseCaseSignal],
        uncertainties: List[str],
        notes: List[str],
        context: ProjectContext,
        sample_count: int,
    ) -> UseCaseAnalysisResult:
        """Score the collected evidence and produce an honest conclusion."""
        # Saturating per-signal contribution: corroboration helps, but a second
        # weak match can never outweigh one strong measured signal.
        scores: Dict[str, float] = {}
        evidence_by_task: Dict[str, List[UseCaseSignal]] = {}
        for signal in signals:
            evidence_by_task.setdefault(signal.task, []).append(signal)
            if signal.weight <= 0 or signal.task.startswith("_"):
                continue
            seen = sum(1 for s in evidence_by_task[signal.task] if s.weight > 0)
            scores[signal.task] = scores.get(signal.task, 0.0) + signal.weight / (0.5 + 0.5 * seen)

        if context.detected_framework and any(
            key in (context.detected_framework or "").lower()
            for key in ("axolotl", "peft", "llamafactory", "unsloth")
        ):
            scores["instruction_tuning"] = scores.get("instruction_tuning", 0.0) + 0.25

        if sample_count == 0:
            uncertainties.append(
                "No readable records were measured, so no use case could be derived from data."
            )

        if not scores:
            return UseCaseAnalysisResult(
                primary_task="unknown",
                confidence="very_low",
                confidence_score=0.0,
                evidence=[s for s in signals if s.task == "_dataset_schema"] or signals[:3],
                reasoning=(
                    "No dataset field names, output shapes or prompt templates indicated "
                    "a specific task."
                ),
                uncertainties=uncertainties or [
                    "Insufficient project content to infer a use case."
                ],
                recommended_approach="prompting",
                approach_rationale=(
                    "With no evidence about the task or data, the lowest-risk first step is "
                    "to establish the task and data with prompting before paying for training."
                ),
            )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        primary_task, top_score = ranked[0]
        secondary = [task for task, _ in ranked[1:3]]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        # Confidence = evidence strength, discounted by how close the runner-up is.
        margin = (top_score - runner_up) / top_score if top_score else 0.0
        confidence_score = round(min(1.0, top_score) * (0.6 + 0.4 * min(1.0, margin)), 3)
        confidence = self._confidence_label(confidence_score)

        if len(ranked) > 1 and margin < 0.25:
            uncertainties.append(
                f"Evidence for '{primary_task}' is close to the evidence for "
                f"'{ranked[1][0]}' — the use case is ambiguous."
            )

        top_evidence = evidence_by_task.get(primary_task, [])
        if not [e for e in top_evidence if e.provenance == "measured"]:
            uncertainties.append(
                f"The '{primary_task}' label rests on heuristic/value evidence rather than "
                "a directly measured project fact."
            )

        reasoning = (
            f"The strongest project evidence points to '{primary_task}'. "
            + " ".join(e.evidence for e in top_evidence[:3])
            + (f" Supporting observations: {' '.join(notes)}" if notes else "")
        )

        approach, rationale = self._recommend_approach(primary_task, sample_count, context)

        return UseCaseAnalysisResult(
            primary_task=primary_task,
            secondary_tasks=secondary,
            confidence=confidence,
            confidence_score=confidence_score,
            evidence=top_evidence,
            reasoning=reasoning.strip(),
            uncertainties=sorted(set(uncertainties)),
            recommended_approach=approach,
            approach_rationale=rationale,
        )

    def _recommend_approach(
        self, task: str, samples: int, context: ProjectContext
    ) -> Tuple[str, str]:
        """Recommend prompting / RAG / fine-tuning from the measured evidence."""
        has_rag_code = any(
            "rag" in Path(p).name.lower() or "retriev" in Path(p).name.lower()
            for p in (context.inference_scripts or [])
        )

        if task in ("embedding_search", "retrieval"):
            return "rag", (
                f"The project is retrieval-shaped (evidence for '{task}' was found"
                + (", and retrieval/vector code is present" if has_rag_code else "")
                + "), so supplying private knowledge is the goal. RAG fits directly; "
                "fine-tuning would teach style, not facts."
            )
        if samples == 0:
            return "prompting", (
                "No usable training data was measured, so fine-tuning is not yet viable. "
                "Fix the prompt and confirm the task first."
            )
        if samples < 100:
            return "combination", (
                f"Only {samples} usable record(s) were measured — below the range where "
                "fine-tuning reliably beats a well-specified prompt. Start with prompting, "
                "and add RAG if private documents hold the missing knowledge."
            )
        if task == "classification":
            return "lora", (
                f"{samples} records with a closed label set is the classic LoRA case: "
                "consistent labelling behaviour is what needs teaching, and LoRA avoids "
                "overwriting general capability with a small dataset."
            )
        if task in (
            "code_generation", "structured_generation", "summarization",
            "chatbot", "instruction_tuning", "question_answering",
        ):
            return "lora", (
                f"{samples} records for a '{task}' task support behavioural fine-tuning. "
                "LoRA/QLoRA teaches the target format and domain style while keeping the "
                "base model's general ability intact."
            )
        return "combination", (
            f"'{task}' with {samples} records: prompting carries the instruction, RAG "
            "covers private or changing documents, and LoRA covers consistent style. Start "
            "with prompting and add layers only where evaluation shows a gap."
        )

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        if score >= 0.25:
            return "low"
        return "very_low"

    @staticmethod
    def _output_text(record: Dict[str, Any]) -> str:
        """Extract the target/output text (same field precedence the
        DatasetAnalyzer uses, so both agree on what a response is)."""
        for key in ("response", "completion", "output", "answer", "text", "summary", "label"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
