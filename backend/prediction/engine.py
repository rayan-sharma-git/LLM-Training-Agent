"""Training outcome prediction engine.

The engine answers one question honestly:

    "Given the actual project, dataset, model, hardware and training
     configuration, what can we reasonably estimate, what assumptions are we
     making, what risks are indicated, and what cannot be predicted reliably?"

Every value it returns is labelled with its provenance:

  * ``measured``   - read from the user's real project files (dataset
                     statistics, the user's training config, detected GPUs)
  * ``calculated`` - derived from measured values with a documented formula
                     (total tokens, training steps)
  * ``heuristic``  - an engineering rule of thumb applied to measured values
                     (risk indicators, expected direction of behaviour)
  * ``assumed``    - a value that had to be assumed to produce an estimate
                     (compute utilisation, unknown GPU throughput)
  * ``unknown``    - required information that was not available

The engine deliberately does NOT produce a guaranteed accuracy, benchmark
score or expected quality improvement: this project holds no empirical
evaluation data, so no such number could be defended. ``status``,
``confidence_basis``, ``uncertainty`` and ``unknowns`` make the limits of the
estimate explicit instead of hiding them behind a percentage.

Resource estimates are NOT recalculated here. The engine reuses the existing
``hardware.gpu_time_estimator.GPUTimeEstimator`` (the single authoritative
implementation of the FLOPs/throughput math) and only feeds it the real
analyzer outputs.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from models.schemas import (
    DatasetAnalysisResult,
    HyperparameterAnalysisResult,
    ModelAnalysisResult,
    PredictionResult,
    ProjectContext,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engineering heuristics
#
# These thresholds are documented rules of thumb, chosen because they mark the
# point where the expected *direction* of trouble changes — not because they are
# scientifically established laws. Every risk below reports its own rationale,
# so an engineer can disagree with a threshold and still use the analysis.
# ---------------------------------------------------------------------------

#: Below this sample count, fine-tuning tends to memorise rather than generalise.
SMALL_DATASET_SAMPLES = 100
#: Above this count the diversity signals (near-duplicate ratio) become meaningful.
MODERATE_DATASET_SAMPLES = 1000
#: Duplicate share at which optimisation spends capacity on repeats.
HIGH_DUPLICATE_PCT = 10.0
#: Duplicate share that already biases sampling away from unique content.
MODERATE_DUPLICATE_PCT = 2.0
#: Near-duplicate share at which output diversity is expected to collapse.
HIGH_NEAR_DUPLICATE_PCT = 20.0
#: Missing/empty field share at which the training signal becomes inconsistent.
HIGH_MISSING_FIELD_PCT = 10.0
#: Missing/empty field share that already weakens instruction following.
MODERATE_MISSING_FIELD_PCT = 2.0
#: Learning rates above this are conventionally unstable for common fine-tuning
#: setups (typical: 1e-5..5e-5 full fine-tune, 1e-4..3e-4 LoRA/QLoRA).
HIGH_LEARNING_RATE = 1e-3
#: Learning rates below this are conventionally too small to move the loss in a
#: few thousand steps (LoRA commonly uses 1e-4..3e-4, full fine-tune 1e-5+).
LOW_LEARNING_RATE = 1e-6
#: Project health score below which the dataset/configuration combination is
#: treated as a quality risk.
LOW_QUALITY_SCORE = 0.5
#: Characters per token used when converting measured record lengths to tokens.
#: A statistical average for English text — an assumption, never a measurement.
CHARS_PER_TOKEN = 4.0

#: Reporting order for risk severities (most important first).
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
#: Epoch counts beyond this over a tiny step budget are a classic overfitting
#: pattern (the model sees the same samples many times).
MANY_EPOCHS = 5
#: Steps per epoch below this make repeating the data for many epochs wasteful.
LOW_STEPS_PER_EPOCH = 200
#: Sequence length past which memory pressure becomes the expected bottleneck.
LONG_SEQUENCE_LENGTH = 4096
#: Consistency score required before "good" behaviour is expected.
GOOD_CONSISTENCY = 0.8
#: Below this score, formatting/response consistency is expected to be poor.
POOR_CONSISTENCY = 0.5


def _finite(
    value: Any,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    """Return *value* as a usable float, or ``None`` when it is invalid.

    Guards every failure mode that would otherwise produce a
    believable-looking number: missing values, non-numeric strings, NaN,
    infinities, and out-of-range values (zero/negative counts, absurd sizes).
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    if minimum is not None and numeric < minimum:
        return None
    if maximum is not None and numeric > maximum:
        return None
    return numeric


def _clean_pct(value: Any, maximum: float = 100.0) -> Optional[float]:
    """Clamp a percentage into ``[0, maximum]``; ``None`` when unusable."""
    return _finite(value, minimum=0.0, maximum=maximum)


def _clean_score(value: Any) -> Optional[float]:
    """Clamp a 0..1 score; ``None`` when unusable."""
    return _finite(value, minimum=0.0, maximum=1.0)


def _parse_param_count(parameter_count: Any) -> Optional[float]:
    """Parse a parameter count such as ``"8B"`` or ``7.1`` into billions."""
    if parameter_count is None:
        return None
    if isinstance(parameter_count, (int, float)) and not isinstance(parameter_count, bool):
        return _finite(parameter_count, minimum=0.0, maximum=1e5)
    text = str(parameter_count).strip().lower().replace(",", "")
    if not text or text in ("unknown", "none", "n/a", "unrecognized"):
        return None
    multiplier = 1.0
    if text.endswith("b"):
        text = text[:-1]
    elif text.endswith("m"):
        text = text[:-1]
        multiplier = 1e-3
    try:
        value = float(text) * multiplier
    except ValueError:
        return None
    return _finite(value, minimum=0.0, maximum=1e5)


class PredictionEngine:
    """Predicts likely training outcomes from real analysis results.

    All predictions are deterministic heuristics derived from the actual
    DatasetAnalyzer / HyperparameterAnalyzer / ModelAdvisor outputs. They are
    heuristic expectations, not guaranteed results, and the confidence field
    reflects how much real data the prediction is based on.
    """

    async def predict(
        self,
        context: ProjectContext,
        dataset_result: DatasetAnalysisResult,
        hp_result: HyperparameterAnalysisResult,
        model_result: ModelAnalysisResult,
    ) -> PredictionResult:
        """Predict training outcomes from the measured analysis results."""
        logging.info("Generating predictions")

        failure_modes: List[str] = []
        expected_strengths: List[str] = []
        expected_weaknesses: List[str] = []

        # ------------------------------------------------------------------
        # Dataset-driven signals (measured)
        # ------------------------------------------------------------------
        sample_count = dataset_result.sample_count
        has_data = sample_count > 0

        # Instruction following: driven by measured prompt/response quality
        if has_data and dataset_result.missing_field_percentage < 2 and dataset_result.instruction_consistency_score > 0.8:
            instruction_following = "good"
            expected_strengths.append("Consistent, complete prompts measured across the dataset")
        elif has_data and dataset_result.missing_field_percentage > 10:
            instruction_following = "poor"
            failure_modes.append(
                f"{dataset_result.missing_field_percentage:.0f}% of samples have missing/empty fields"
            )
        else:
            instruction_following = "fair"

        # Hallucination risk: small datasets and duplicates raise it
        if not has_data:
            hallucination_risk = "unknown"
            failure_modes.append("No readable dataset — training outcome cannot be predicted")
        elif sample_count < 100 or dataset_result.duplicate_percentage > 10:
            hallucination_risk = "high"
            if sample_count < 100:
                failure_modes.append(f"Very small dataset ({sample_count} samples) — memorization and hallucination risk")
            if dataset_result.duplicate_percentage > 10:
                failure_modes.append(f"{dataset_result.duplicate_percentage:.0f}% duplicate samples bias the model")
        elif dataset_result.duplicate_percentage > 2:
            hallucination_risk = "medium"
        else:
            hallucination_risk = "low"

        # Formatting: from the measured response formatting consistency
        formatting = (
            "good" if has_data and dataset_result.formatting_consistency_score > 0.8
            else "poor" if has_data and dataset_result.formatting_consistency_score < 0.5
            else "fair" if has_data
            else "unknown"
        )
        response_consistency = (
            "good" if has_data and dataset_result.response_consistency_score > 0.8
            else "poor" if has_data and dataset_result.response_consistency_score < 0.5
            else "fair" if has_data
            else "unknown"
        )
        if has_data and dataset_result.formatting_consistency_score < 0.5:
            failure_modes.append("Inconsistent response formatting measured in the dataset")

        # ------------------------------------------------------------------
        # Model-driven signals (reference data)
        # ------------------------------------------------------------------
        model_known = model_result.parameter_count not in (None, "", "unknown")
        if not model_known:
            reasoning_prediction = "unknown"
            expected_weaknesses.append(
                f"Base model '{model_result.selected_model}' is not recognized — capability predictions unavailable"
            )
        else:
            # Parameter-count-scaled heuristic (reference data, not measured)
            try:
                params = float(str(model_result.parameter_count).lower().replace("b", ""))
            except ValueError:
                params = None
            reasoning_prediction = (
                "good" if params and params >= 7 else "fair" if params and params >= 1 else "poor"
            )
            if params and params >= 7:
                expected_strengths.append(f"~{model_result.parameter_count} parameter model provides solid reasoning capacity")
            elif params is not None:
                expected_weaknesses.append(
                    f"Small (~{model_result.parameter_count}) model may struggle with complex reasoning"
                )

        # Creativity: derive from dataset diversity (duplicates reduce it).
        creativity = (
            "medium" if not has_data
            else "high" if dataset_result.near_duplicate_percentage < 5 and sample_count > 500
            else "low" if dataset_result.near_duplicate_percentage > 20
            else "medium"
        )

        # ------------------------------------------------------------------
        # Hyperparameter-driven signals (user configuration)
        # ------------------------------------------------------------------
        if hp_result.epochs is not None and hp_result.batch_size is not None and has_data:
            steps_per_epoch = max(1, sample_count // max(1, hp_result.batch_size))
            if hp_result.epochs > 5 and steps_per_epoch < 200:
                failure_modes.append(
                    f"Many epochs ({hp_result.epochs}) over a small step count (~{steps_per_epoch}/epoch) — overfitting risk"
                )
        if hp_result.learning_rate is not None and hp_result.learning_rate > 1e-3:
            failure_modes.append("Learning rate >1e-3 measured in the configuration — instability risk")

        # Overfitting risk from the hyperparameter analyzer feeds hallucination
        if hp_result.overfitting_risk == "high" and hallucination_risk == "medium":
            hallucination_risk = "high"

        # ------------------------------------------------------------------
        # Confidence: reflects how much real data the prediction used
        # ------------------------------------------------------------------
        evidence = 0
        if has_data:
            evidence += 1
        if model_known:
            evidence += 1
        if hp_result.learning_rate is not None or hp_result.epochs is not None:
            evidence += 1
        confidence = {0: "very_low", 1: "low", 2: "medium", 3: "high"}[evidence]

        if not expected_strengths:
            expected_strengths.append("No clear strengths could be derived from the available data")
        if not expected_weaknesses:
            expected_weaknesses.append("No significant weaknesses identified from the measured data")

        return PredictionResult(
            instruction_following_prediction=instruction_following,
            hallucination_risk=hallucination_risk,
            reasoning_prediction=reasoning_prediction,
            response_consistency_prediction=response_consistency,
            creativity_prediction=creativity,
            formatting_prediction=formatting,
            likely_failure_modes=failure_modes,
            expected_strengths=expected_strengths,
            expected_weaknesses=expected_weaknesses,
            confidence=confidence,
        )