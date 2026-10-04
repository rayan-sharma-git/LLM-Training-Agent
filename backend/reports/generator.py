"""Engineering report generator."""
from __future__ import annotations

import logging
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
    EngineeringReport,
)


class ReportGenerator:
    """Generates comprehensive engineering reports."""
    
    def generate(self, context: ProjectContext, dataset_result: DatasetAnalysisResult, prompt_result: PromptAnalysisResult, hp_result: HyperparameterAnalysisResult, model_result: ModelAnalysisResult, cost_result: CostEstimate, prediction_result: PredictionResult, recommendations: List[Recommendation], gpu_time_estimate=None, hardware_detection=None) -> EngineeringReport:
        """Generate engineering report from all analysis results."""
        logging.info("Generating report")
        
        # Build summaries
        dataset_summary = {
            "quality_score": dataset_result.quality_score,
            "sample_count": dataset_result.sample_count,
            "confidence": dataset_result.confidence,
        }
        
        prompt_summary = {
            "clarity_score": prompt_result.clarity_score,
            "complexity": prompt_result.prompt_complexity,
            "confidence": prompt_result.confidence,
        }
        
        hp_summary = {
            "efficiency_score": hp_result.efficiency_score,
            "overfitting_risk": hp_result.overfitting_risk,
            "confidence": hp_result.confidence,
        }
        
        model_summary = {
            "selected_model": model_result.selected_model,
            "context_length": model_result.context_length,
            "confidence": model_result.confidence,
        }
        
        prediction_summary = {
            "instruction_following": prediction_result.instruction_following_prediction,
            "hallucination_risk": prediction_result.hallucination_risk,
            "confidence": prediction_result.confidence,
        }
        
        cost_summary = {
            "training_time": cost_result.estimated_training_time,
            "gpu_hours": cost_result.estimated_gpu_hours,
            "confidence": cost_result.confidence,
        }

        # GPU estimate & hardware data for report
        gpu_estimate_model = None
        if gpu_time_estimate:
            from models.schemas import GPUTimeEstimate
            try:
                if isinstance(gpu_time_estimate, GPUTimeEstimate):
                    gpu_estimate_model = gpu_time_estimate
                elif isinstance(gpu_time_estimate, dict):
                    gpu_estimate_model = GPUTimeEstimate(**{k: v for k, v in gpu_time_estimate.items() if k in GPUTimeEstimate.model_fields})
            except Exception as e:
                logging.warning(f"Failed to create GPUTimeEstimate: {e}")

        hardware_model = None
        if hardware_detection:
            from models.schemas import HardwareInfo
            try:
                if isinstance(hardware_detection, HardwareInfo):
                    hardware_model = hardware_detection
                elif isinstance(hardware_detection, dict):
                    hardware_model = HardwareInfo(**{k: v for k, v in hardware_detection.items() if k in HardwareInfo.model_fields})
            except Exception as e:
                logging.warning(f"Failed to create HardwareInfo: {e}")

        # Compute scores
        project_health_score = self._compute_health_score(dataset_result, hp_result, model_result)
        training_readiness_score = self._compute_readiness_score(dataset_result, hp_result, model_result)
        
        # Executive summary
        executive_summary = self._generate_executive_summary(
            context, dataset_result, hp_result, model_result, recommendations
        )
        
        # Action plan
        action_plan = self._generate_action_plan(recommendations)
        
        return EngineeringReport(
            executive_summary=executive_summary,
            project_health_score=project_health_score,
            training_readiness_score=training_readiness_score,
            dataset_summary=dataset_summary,
            prompt_summary=prompt_summary,
            hyperparameter_summary=hp_summary,
            model_summary=model_summary,
            prediction_summary=prediction_summary,
            cost_summary=cost_summary,
            gpu_time_estimate=gpu_estimate_model,
            hardware_detection=hardware_model,
            prioritized_recommendations=recommendations,
            action_plan=action_plan,
        )
    
    def _compute_health_score(self, dataset_result: DatasetAnalysisResult, hp_result: HyperparameterAnalysisResult, model_result: ModelAnalysisResult) -> float:
        """Compute overall project health score from the measured analysis results.

        Every input is a real analyzer output. The model contribution is a
        documented weight based on how much verified data exists about the
        selected model (high confidence = reference database match), not a
        quality guarantee.
        """
        scores = [dataset_result.quality_score, hp_result.efficiency_score]
        # Model-confidence weight: high = recognized model (reference data),
        # medium = partially verified, low/very_low = unknown model.
        model_weight = {
            "high": 0.8, "very_high": 0.9, "medium": 0.5, "low": 0.2, "very_low": 0.1,
        }.get(model_result.confidence, 0.5)
        scores.append(model_weight)
        return sum(scores) / len(scores)

    def _compute_readiness_score(self, dataset_result: DatasetAnalysisResult, hp_result: HyperparameterAnalysisResult, model_result: ModelAnalysisResult) -> float:
        """Compute training readiness from the measured results (0-1, continuous).

        This is a documented heuristic weighting of the real analyzer outputs —
        never a fixed constant.
        """
        has_data = dataset_result.sample_count > 0
        # Data readiness: quality score, discounted when data is missing.
        data_component = dataset_result.quality_score if has_data else 0.0
        # Configuration readiness: efficiency score, discounted when the
        # configuration was not found (confidence very_low).
        config_component = (
            hp_result.efficiency_score * 0.5
            if hp_result.confidence in ("very_low", "low")
            else hp_result.efficiency_score
        )
        # Model readiness: recognized model with known specs.
        model_component = 1.0 if model_result.confidence == "high" else 0.5 if model_result.confidence == "medium" else 0.0

        return round(0.5 * data_component + 0.3 * config_component + 0.2 * model_component, 3)
    
    def _generate_executive_summary(self, context: ProjectContext, dataset_result: DatasetAnalysisResult, hp_result: HyperparameterAnalysisResult, model_result: ModelAnalysisResult, recommendations: List[Recommendation]) -> str:
        """Generate executive summary."""
        return (
            f"Project '{context.project_name}' analyzed successfully. "
            f"Dataset quality: {dataset_result.quality_score:.0%}. "
            f"Training configuration efficiency: {hp_result.efficiency_score:.0%}. "
            f"{len(recommendations)} recommendations generated."
        )
    
    def _generate_action_plan(self, recommendations: List[Recommendation]) -> List[str]:
        """Generate prioritized action plan."""
        if not recommendations:
            return ["No critical actions required. Proceed with training."]
        
        plan = []
        for i, rec in enumerate(recommendations[:5], 1):
            plan.append(f"{i}. {rec.title}: {rec.description}")
        return plan