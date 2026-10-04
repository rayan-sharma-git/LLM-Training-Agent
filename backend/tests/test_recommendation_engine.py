"""Tests for RecommendationEngine."""
import pytest
from recommendation.engine import RecommendationEngine
from models.schemas import (
    ProjectContext, DatasetAnalysisResult, PromptAnalysisResult,
    HyperparameterAnalysisResult, ModelAnalysisResult, CostEstimate, PredictionResult,
)


@pytest.fixture
def engine():
    return RecommendationEngine()


@pytest.fixture
def sample_context():
    return ProjectContext(project_name="test", project_path="/tmp")


@pytest.fixture
def dataset_result():
    return DatasetAnalysisResult(
        dataset_name="data.json", sample_count=1000, token_count=250000,
        average_prompt_length=150.0, average_response_length=300.0,
        duplicate_percentage=10.0, near_duplicate_percentage=5.0,
        missing_field_percentage=1.0, formatting_consistency_score=0.95,
        language_consistency_score=0.98, instruction_consistency_score=0.90,
        response_consistency_score=0.92, quality_score=0.88, confidence="high",
    )


@pytest.fixture
def prompt_result():
    return PromptAnalysisResult(
        template_name="default", ambiguity_score=0.2, clarity_score=0.85,
        formatting_score=0.90, instruction_quality_score=0.80,
        consistency_score=0.88, confidence="medium",
    )


@pytest.fixture
def hp_result():
    return HyperparameterAnalysisResult(
        learning_rate=5e-4, batch_size=8, epochs=3,
        efficiency_score=0.75, confidence="medium",
    )


@pytest.fixture
def model_result():
    return ModelAnalysisResult(
        selected_model="llama3.2", parameter_count="7B",
        context_length=4096, estimated_vram="8GB", confidence="medium",
    )


@pytest.fixture
def cost_result():
    return CostEstimate(
        estimated_training_time="2-4 hours", estimated_gpu_hours=3.0,
        estimated_vram_usage="8GB", estimated_checkpoint_size="14GB",
        estimated_storage_requirement="50GB", confidence="medium",
    )


@pytest.fixture
def prediction_result():
    return PredictionResult(
        instruction_following_prediction="good",
        hallucination_risk="low", confidence="medium",
    )


@pytest.mark.asyncio
async def test_recommendation_engine_with_high_duplicates(
    engine, sample_context, dataset_result, prompt_result, hp_result, model_result, cost_result, prediction_result
):
    recommendations = await engine.generate(
        sample_context, dataset_result, prompt_result, hp_result, model_result, cost_result, prediction_result
    )
    assert len(recommendations) > 0
    assert all(r.evidence for r in recommendations)
    assert all(r.confidence for r in recommendations)


@pytest.mark.asyncio
async def test_recommendation_engine_no_duplicates(
    engine, sample_context, prompt_result, hp_result, model_result, cost_result, prediction_result
):
    dataset_result = DatasetAnalysisResult(
        dataset_name="data.json", sample_count=1000, token_count=250000,
        average_prompt_length=150.0, average_response_length=300.0,
        duplicate_percentage=1.0, near_duplicate_percentage=0.0,
        missing_field_percentage=0.0, formatting_consistency_score=0.95,
        language_consistency_score=0.98, instruction_consistency_score=0.90,
        response_consistency_score=0.92, quality_score=0.95, confidence="high",
    )
    recommendations = await engine.generate(
        sample_context, dataset_result, prompt_result, hp_result, model_result, cost_result, prediction_result
    )
    assert isinstance(recommendations, list)