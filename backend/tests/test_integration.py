"""Integration tests for backend pipeline."""
import pytest
from scanner.scanner import ProjectScanner
from scanner.context_builder import ContextBuilder
from analyzers.dataset_analyzer import DatasetAnalyzer
from analyzers.prompt_analyzer import PromptAnalyzer
from analyzers.hyperparameter_analyzer import HyperparameterAnalyzer
from analyzers.model_advisor import ModelAdvisor
from analyzers.cost_estimator import CostEstimator
from prediction.engine import PredictionEngine
from recommendation.engine import RecommendationEngine
from reports.generator import ReportGenerator


@pytest.mark.asyncio
async def test_full_analysis_pipeline(tmp_path):
    """Test the complete analysis pipeline end-to-end."""
    # Create a mock project
    (tmp_path / "data.json").write_text('{"key": "value"}')
    (tmp_path / "train.py").write_text("print('train')")
    (tmp_path / "requirements.txt").write_text("transformers\ndatasets\npeft\n")

    # Step 1: Scan project
    scanner = ProjectScanner(str(tmp_path))
    scan_result = scanner.scan()
    assert scan_result["project_name"] == tmp_path.name

    # Step 2: Build context
    context_builder = ContextBuilder()
    context = context_builder.build(scan_result)
    assert context.project_name == tmp_path.name

    # Step 3: Run analyzers
    dataset_analyzer = DatasetAnalyzer()
    prompt_analyzer = PromptAnalyzer()
    hp_analyzer = HyperparameterAnalyzer()
    model_advisor = ModelAdvisor()
    cost_estimator = CostEstimator()

    dataset_result = await dataset_analyzer.analyze(context)
    prompt_result = await prompt_analyzer.analyze(context)
    hp_result = await hp_analyzer.analyze(context)
    model_result = await model_advisor.analyze(context)
    cost_result = await cost_estimator.analyze(context)

    # Step 4: Predict
    prediction_engine = PredictionEngine()
    prediction_result = await prediction_engine.predict(context, dataset_result, hp_result, model_result)

    # Step 5: Generate recommendations
    rec_engine = RecommendationEngine()
    recommendations = await rec_engine.generate(
        context, dataset_result, prompt_result, hp_result, model_result, cost_result, prediction_result
    )

    # Step 6: Generate report
    report_gen = ReportGenerator()
    report = report_gen.generate(
        context, dataset_result, prompt_result, hp_result, model_result,
        cost_result, prediction_result, recommendations
    )

    # Verify report
    assert report.project_health_score > 0
    assert report.training_readiness_score > 0
    assert len(report.action_plan) > 0
