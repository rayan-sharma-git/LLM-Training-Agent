"""Shared mapping helpers for analyzer results.

Both the HTTP routes (``api/routes.py``) and the WebSocket pipeline
(``api/websocket.py``) must turn analyzer output into typed models before
handing them to the Prediction/Recommendation/Report layers. This module is the
single authoritative place for that conversion, so the two entry points cannot
drift apart.

Every fallback here is an *explicit unknown* (zero/unknown/``low`` confidence),
never a plausible-looking demo value: a failed analyzer must be visible
downstream instead of silently looking like a successful analysis.
"""
from __future__ import annotations

from typing import Any, Dict

from models.schemas import (
    CostEstimate,
    DatasetAnalysisResult,
    EngineeringReport,
    HyperparameterAnalysisResult,
    ModelAnalysisResult,
    PromptAnalysisResult,
    PredictionResult,
)


def safe_model(data: Any, model_cls: type, **defaults) -> Any:
    """Safely construct a Pydantic model from an analyzer result.

    Accepts either an already-constructed model (returned unchanged) or a dict
    produced by ``model_dump()``. If *data* is an error/partial dict that cannot
    satisfy the model's required fields, the explicit *defaults* are used.
    """
    if data is None:
        return model_cls(**defaults)
    if isinstance(data, model_cls):
        return data
    if not isinstance(data, dict):
        return model_cls(**defaults)
    try:
        return model_cls(**{k: v for k, v in data.items() if k in model_cls.model_fields})
    except Exception:
        return model_cls(**defaults)


def default_dataset_result() -> DatasetAnalysisResult:
    """Explicit "dataset could not be analysed" state (all metrics zero)."""
    return DatasetAnalysisResult(
        dataset_name="unknown", sample_count=0, token_count=0,
        average_prompt_length=0, average_response_length=0,
        duplicate_percentage=0, near_duplicate_percentage=0,
        missing_field_percentage=0, formatting_consistency_score=0,
        language_consistency_score=0, instruction_consistency_score=0,
        response_consistency_score=0, quality_score=0, confidence="low",
    )


def default_prompt_result() -> PromptAnalysisResult:
    """Explicit "no prompts found" state (neutral scores, very low confidence)."""
    return PromptAnalysisResult(
        template_name="unknown", ambiguity_score=0.5, clarity_score=0.5,
        formatting_score=0.5, instruction_quality_score=0.5,
        consistency_score=0.5, confidence="low",
    )


def default_hp_result() -> HyperparameterAnalysisResult:
    """Explicit "no training configuration found" state (no invented values)."""
    return HyperparameterAnalysisResult(efficiency_score=0.0, confidence="low")


def default_model_result() -> ModelAnalysisResult:
    """Explicit "unrecognized model" state."""
    return ModelAnalysisResult(
        selected_model="unknown", parameter_count="unknown",
        context_length=0, estimated_vram="unknown", confidence="low",
    )


def default_cost_result() -> CostEstimate:
    """Explicit "cost could not be estimated" state."""
    return CostEstimate(
        estimated_training_time="unknown", estimated_gpu_hours=0.0,
        estimated_vram_usage="unknown", estimated_checkpoint_size="unknown",
        estimated_storage_requirement="unknown", confidence="low",
    )


def default_prediction_result() -> PredictionResult:
    """Explicit "prediction unavailable" state used when the engine fails."""
    return PredictionResult(
        status="not_estimated",
        confidence="very_low",
        confidence_basis="The prediction engine did not run successfully.",
        uncertainty="No prediction is available for this project.",
        unknowns=["Prediction engine did not complete"],
    )


def default_engineering_report(project_name: str) -> EngineeringReport:
    """Explicit "report unavailable" state used when report generation fails."""
    return EngineeringReport(
        executive_summary=f"Analysis of {project_name} completed with errors.",
        project_health_score=0.0,
        training_readiness_score=0.0,
        prioritized_recommendations=[],
        action_plan=["Fix backend errors and re-run analysis."],
    )


def coerce_gpu_estimate(raw: Any) -> Dict[str, Any]:
    """Normalize a GPU time estimate (``EstimateResult``/model/dict) to a dict.

    Returns an empty dict when no usable estimate is available so callers can
    treat "no estimate" as missing data instead of zero seconds.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    to_dict = getattr(raw, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return {}
    if hasattr(raw, "model_dump"):
        try:
            return raw.model_dump()
        except Exception:
            return {}
    return {}
