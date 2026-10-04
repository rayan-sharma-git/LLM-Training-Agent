"""Recommendation generation engine."""
from __future__ import annotations

import logging
import uuid
from typing import List
from models.schemas import (
    ProjectContext,
    DatasetAnalysisResult,
    PromptAnalysisResult,
    HyperparameterAnalysisResult,
    ModelAnalysisResult,
    CostEstimate,
    PredictionResult,
    Recommendation,
)


class RecommendationEngine:
    """Generates prioritized recommendations from analysis results."""
    
    async def generate(self, context: ProjectContext, dataset_result: DatasetAnalysisResult, prompt_result: PromptAnalysisResult, hp_result: HyperparameterAnalysisResult, model_result: ModelAnalysisResult, cost_result: CostEstimate, prediction_result: PredictionResult) -> List[Recommendation]:
        """Generate recommendations."""
        logging.info("Generating recommendations")
        
        recommendations = []
        
        # --- Dataset recommendations (from measured dataset analysis) -------
        if dataset_result.sample_count == 0:
            recommendations.append(self._create_recommendation(
                category="dataset",
                title="Dataset could not be analyzed",
                description="No readable records were found in the project's dataset files.",
                reasoning="Training cannot proceed or be evaluated without readable data",
                evidence=f"sample_count={dataset_result.sample_count}; findings: {'; '.join(dataset_result.findings[:3]) or 'none'}",
                severity="critical",
                confidence="high",
                estimated_benefit="Unblocks all data-driven analysis",
                implementation_difficulty="easy",
                estimated_engineering_time="varies",
                affected_files=context.dataset_paths,
                suggested_actions=["Verify the dataset files exist and are valid JSONL/JSON/CSV/TSV", "Re-run analysis"],
            ))
        elif dataset_result.duplicate_percentage > 5.0:
            recommendations.append(self._create_recommendation(
                category="dataset",
                title="Remove duplicate samples",
                description=f"Dataset contains {dataset_result.duplicate_percentage}% duplicates",
                reasoning="Duplicates reduce training efficiency",
                evidence=f"Duplicate percentage: {dataset_result.duplicate_percentage}%",
                severity="high",
                confidence="high",
                estimated_benefit="10-20% faster training",
                implementation_difficulty="easy",
                estimated_engineering_time="1-2 hours",
                affected_files=context.dataset_paths,
                suggested_actions=["Run deduplication script", "Verify dataset integrity"],
            ))

        if dataset_result.missing_field_percentage > 2:
            recommendations.append(self._create_recommendation(
                category="dataset",
                title="Fix samples with missing fields",
                description=f"{dataset_result.missing_field_percentage:.1f}% of samples have missing or empty prompt/response fields",
                reasoning="Incomplete samples degrade instruction-following quality",
                evidence=f"Missing-field percentage: {dataset_result.missing_field_percentage:.1f}% (measured)",
                severity="high",
                confidence="high",
                estimated_benefit="Cleaner training signal",
                implementation_difficulty="easy",
                estimated_engineering_time="1-3 hours",
                affected_files=context.dataset_paths,
                suggested_actions=["Remove or complete incomplete records", "Add schema validation to the data pipeline"],
            ))
        
        # --- Prompt recommendations (from measured prompt analysis) ---------
        if prompt_result.confidence != "very_low" and (prompt_result.clarity_score < 0.6 or prompt_result.detected_issues):
            recommendations.append(self._create_recommendation(
                category="prompt",
                title="Improve prompt template quality",
                description=f"Prompt template '{prompt_result.template_name}' has measurable issues.",
                reasoning="Clear, unambiguous prompts measurably improve fine-tuned model behaviour",
                evidence=(
                    f"clarity={prompt_result.clarity_score:.2f}, ambiguity={prompt_result.ambiguity_score:.2f}; "
                    f"issues: {'; '.join(prompt_result.detected_issues[:3]) or 'none'}"
                ),
                severity="medium",
                confidence="medium",
                estimated_benefit="Better instruction adherence",
                implementation_difficulty="easy",
                estimated_engineering_time="30-60 minutes",
                affected_files=context.prompt_templates,
                suggested_actions=prompt_result.recommendations[:3] or ["Clarify instructions and add template variables"],
            ))
        if hp_result.learning_rate and hp_result.learning_rate > 1e-3:
            recommendations.append(self._create_recommendation(
                category="hyperparameter",
                title="Reduce learning rate",
                description=f"Learning rate {hp_result.learning_rate} may be too high",
                reasoning="High learning rates can cause training instability",
                evidence=f"Current learning rate: {hp_result.learning_rate}",
                severity="medium",
                confidence="medium",
                estimated_benefit="More stable training",
                implementation_difficulty="easy",
                estimated_engineering_time="30 minutes",
                affected_files=[],
                suggested_actions=["Reduce learning rate by 10x", "Add learning rate scheduler"],
            ))
        
        if hp_result.confidence in ("very_low", "low"):
            recommendations.append(self._create_recommendation(
                category="hyperparameter",
                title="Add a training configuration",
                description="No training hyperparameters (learning rate, batch size, epochs) were found in the project.",
                reasoning="Analysis and cost estimation are unreliable without the actual configuration",
                evidence="; ".join(hp_result.findings[:3]) or "all core hyperparameters missing",
                severity="medium",
                confidence="high",
                estimated_benefit="Enables config validation and accurate cost estimates",
                implementation_difficulty="easy",
                estimated_engineering_time="30 minutes",
                affected_files=context.configuration_files,
                suggested_actions=["Add a config.yaml with learning_rate, batch size and epochs"],
            ))

        # --- Model recommendations (from reference-data analysis) -----------
        if model_result.parameter_count == "unknown":
            recommendations.append(self._create_recommendation(
                category="model",
                title="Specify a recognized base model",
                description=f"The base model '{model_result.selected_model}' could not be resolved to known specifications.",
                reasoning="Model capability and VRAM guidance require a known model",
                evidence=f"selected_model={model_result.selected_model}; parameter_count=unknown",
                severity="medium",
                confidence="high",
                estimated_benefit="Data-based capability and hardware guidance",
                implementation_difficulty="easy",
                estimated_engineering_time="15 minutes",
                affected_files=context.configuration_files,
                suggested_actions=model_result.recommended_alternatives[:3] or ["Set model_name_or_path in the training config"],
            ))

        # --- Prediction-driven recommendation -------------------------------
        if prediction_result.hallucination_risk == "high":
            recommendations.append(self._create_recommendation(
                category="data",
                title="Mitigate hallucination risk before training",
                description="Measured dataset characteristics predict a high hallucination risk.",
                reasoning="Small datasets, duplicates or heavy overfitting settings cause hallucination",
                evidence="; ".join(prediction_result.likely_failure_modes[:3]) or "hallucination_risk=high (derived)",
                severity="high",
                confidence="medium",
                estimated_benefit="More factual fine-tuned model",
                implementation_difficulty="moderate",
                estimated_engineering_time="2-6 hours",
                affected_files=context.dataset_paths,
                suggested_actions=["Collect more training data", "Deduplicate the dataset", "Reduce epochs"],
            ))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        recommendations.sort(key=lambda r: severity_order.get(r.severity, 5))
        
        return recommendations
    
    def _create_recommendation(self, **kwargs) -> Recommendation:
        """Helper to create recommendation with UUID."""
        return Recommendation(recommendation_id=str(uuid.uuid4()), **kwargs)