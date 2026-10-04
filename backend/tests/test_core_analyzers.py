"""Tests that the five core analyzers perform REAL analysis of project data.

These tests verify that outputs are derived from the user's actual files and
configuration — and that no hardcoded demo/placeholder values are returned
when the required data is missing.
"""
import json
import pytest

from analyzers.dataset_analyzer import DatasetAnalyzer
from analyzers.hyperparameter_analyzer import HyperparameterAnalyzer
from analyzers.cost_estimator import CostEstimator
from analyzers.model_advisor import ModelAdvisor
from analyzers.prompt_analyzer import PromptAnalyzer
from prediction.engine import PredictionEngine
from reports.generator import ReportGenerator
from models.schemas import (
    ProjectContext,
    DatasetAnalysisResult,
    HyperparameterAnalysisResult,
    ModelAnalysisResult,
)


def make_records(n, prompt="Explain gravity in one sentence.", response="Gravity is the attraction between masses."):
    return [{"prompt": prompt, "response": response} for _ in range(n)]


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def _good_dataset_result():
    return DatasetAnalysisResult(
        dataset_name="data.jsonl", sample_count=5000, token_count=900_000,
        average_prompt_length=400.0, average_response_length=800.0,
        duplicate_percentage=0.0, near_duplicate_percentage=0.0,
        missing_field_percentage=0.0, formatting_consistency_score=0.95,
        language_consistency_score=0.98, instruction_consistency_score=0.9,
        response_consistency_score=0.92, quality_score=0.9, confidence="high",
    )


# ---------------------------------------------------------------------------
# Dataset Analyzer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dataset_analyzer_reads_real_records(tmp_path):
    data = tmp_path / "data.jsonl"
    records = [
        {"prompt": f"Question {i} about gravity.", "response": f"Answer {i}: gravity pulls masses together."}
        for i in range(30)
    ]
    records[5]["prompt"] = ""  # one missing field
    records[7] = dict(records[0])  # one exact duplicate
    write_jsonl(data, records)

    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), dataset_paths=["data.jsonl"])
    result = await DatasetAnalyzer().analyze(ctx)

    assert result.sample_count == 30
    assert result.dataset_name == "data.jsonl"
    # Measured from real data:
    assert result.duplicate_percentage == pytest.approx(1 / 30 * 100)
    assert result.missing_field_percentage == pytest.approx(1 / 30 * 100)
    assert result.average_prompt_length > 0
    assert result.average_response_length > 0
    # Existing quality formula damps very small datasets; check measured metrics
    assert result.formatting_consistency_score > 0.5
    assert 0.0 <= result.quality_score <= 1.0


@pytest.mark.asyncio
async def test_dataset_analyzer_aggregates_multiple_files(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl(a, make_records(10))
    write_jsonl(b, make_records(20))
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")

    ctx = ProjectContext(
        project_name="t", project_path=str(tmp_path),
        dataset_paths=["a.jsonl", "b.jsonl", "empty.jsonl"],
    )
    result = await DatasetAnalyzer().analyze(ctx)

    assert result.sample_count == 30
    assert "a.jsonl" in result.dataset_name and "b.jsonl" in result.dataset_name
    assert any("empty.jsonl" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_dataset_analyzer_empty_dataset_reports_unknown_state(tmp_path):
    data = tmp_path / "empty.jsonl"
    data.write_text("", encoding="utf-8")
    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), dataset_paths=["empty.jsonl"])
    result = await DatasetAnalyzer().analyze(ctx)

    assert result.sample_count == 0
    assert result.quality_score == 0.0
    assert result.confidence == "very_low"  # not fabricated "high" confidence
    assert result.findings  # explains why nothing could be analyzed


@pytest.mark.asyncio
async def test_dataset_analyzer_malformed_records_are_reported(tmp_path):
    data = tmp_path / "bad.jsonl"
    data.write_text(
        json.dumps(make_records(1)[0]) + "\n" + "{not valid json}\n" + json.dumps(make_records(1)[0]),
        encoding="utf-8",
    )
    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), dataset_paths=["bad.jsonl"])
    result = await DatasetAnalyzer().analyze(ctx)

    assert result.sample_count == 2
    assert any("Line 2" in w or "parsing" in w.lower() for w in result.warnings + result.findings)


@pytest.mark.asyncio
async def test_dataset_analyzer_missing_file_raises(tmp_path):
    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), dataset_paths=["missing.jsonl"])
    with pytest.raises(FileNotFoundError):
        await DatasetAnalyzer().analyze(ctx)


# ---------------------------------------------------------------------------
# Hyperparameter Analyzer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hyperparameter_analyzer_reads_real_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "learning_rate: 0.0005\nper_device_train_batch_size: 4\nnum_train_epochs: 12\n",
        encoding="utf-8",
    )
    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), configuration_files=["config.yaml"])
    result = await HyperparameterAnalyzer().analyze(ctx)

    assert result.learning_rate == pytest.approx(5e-4)
    assert result.batch_size == 4
    assert result.epochs == 12
    assert result.confidence == "high"
    assert any("project configuration" in f for f in result.findings)
    assert not any("learning rate: not found" in f for f in result.findings)


@pytest.mark.asyncio
async def test_hyperparameter_analyzer_does_not_fabricate_missing_values(tmp_path):
    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), configuration_files=[])
    result = await HyperparameterAnalyzer().analyze(ctx)

    # Values must stay unknown — never silently defaulted to 2e-4 / 8 / 3.
    assert result.learning_rate is None
    assert result.batch_size is None
    assert result.epochs is None
    assert result.confidence == "very_low"
    assert any("not found in the project configuration" in f for f in result.findings)


# ---------------------------------------------------------------------------
# Cost Estimator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cost_estimator_uses_real_measured_values(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "per_device_train_batch_size: 2\nnum_train_epochs: 2\nmax_seq_length: 1024\n",
        encoding="utf-8",
    )
    ctx = ProjectContext(
        project_name="t", project_path=str(tmp_path),
        base_model="meta-llama/llama-3-8b-instruct",
        configuration_files=["config.yaml"],
    )
    hp = HyperparameterAnalysisResult(
        learning_rate=5e-4, batch_size=2, epochs=2, sequence_length=1024, efficiency_score=0.8,
    )
    result = await CostEstimator().analyze(ctx, dataset_result=_good_dataset_result(), hp_result=hp)

    # Deterministic FLOPs formula: 6 * P * tokens / (TFLOPS * 1e12) / 3600 / 0.7
    tokens = 5000 * ((400.0 + 800.0) / 4.0) * 2
    expected_hours = (6 * 8e9 * tokens) / (163e12 * 3600.0) / 0.70
    assert result.estimated_gpu_hours == pytest.approx(expected_hours, rel=0.01)
    assert result.estimated_gpu_hours > 0
    assert result.confidence == "medium"  # no assumed defaults (config was found)
    assert not any("assumed default" in a for a in result.assumptions)


@pytest.mark.asyncio
async def test_cost_estimator_does_not_fabricate_when_data_missing(tmp_path):
    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), base_model=None)
    result = await CostEstimator().analyze(ctx, dataset_result=None, hp_result=None)

    assert result.estimated_gpu_hours == 0.0
    assert "unknown" in result.estimated_training_time
    assert result.confidence == "very_low"
    assert any("unknown" in a for a in result.assumptions)


@pytest.mark.asyncio
async def test_cost_estimator_labels_assumed_defaults(tmp_path):
    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), base_model="mistral-7b")
    result = await CostEstimator().analyze(ctx, dataset_result=_good_dataset_result(), hp_result=None)

    assert result.estimated_gpu_hours > 0
    assert result.confidence == "medium"
    assert any("assumed default" in a for a in result.assumptions)


@pytest.mark.asyncio
async def test_cost_estimator_uses_detected_gpu(tmp_path):
    ctx = ProjectContext(
        project_name="t", project_path=str(tmp_path), base_model="mistral-7b",
        hardware_information={"gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_gb": 24}]},
    )
    result = await CostEstimator().analyze(ctx, dataset_result=_good_dataset_result(), hp_result=None)
    assert any("RTX 4090" in a and "detected" in a for a in result.assumptions)


# ---------------------------------------------------------------------------
# Model Advisor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_advisor_reports_known_model_reference_data():
    ctx = ProjectContext(project_name="t", project_path="/tmp", base_model="meta-llama/llama-3-8b-instruct")
    result = await ModelAdvisor().analyze(ctx)
    assert result.parameter_count == "8B"
    assert result.confidence == "high"
    assert any("reference" in w for w in result.weaknesses)


@pytest.mark.asyncio
async def test_model_advisor_unknown_model_gets_unknown_not_medium():
    ctx = ProjectContext(project_name="t", project_path="/tmp", base_model="totally-made-up-model-99x")
    result = await ModelAdvisor().analyze(ctx)

    assert result.parameter_count == "unknown"
    assert result.reasoning_capability == "unknown"  # NOT fabricated "medium"
    assert result.coding_capability == "unknown"
    assert result.confidence == "low"
    assert any("not recognized" in w for w in result.weaknesses)


# ---------------------------------------------------------------------------
# Prompt Analyzer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_analyzer_reads_real_template_file(tmp_path):
    (tmp_path / "prompt.txt").write_text(
        "You are a helpful assistant.\n"
        "### Instruction:\nSummarize the {document} in one paragraph.\n"
        "### Response:\n",
        encoding="utf-8",
    )
    ctx = ProjectContext(project_name="t", project_path=str(tmp_path), prompt_templates=["prompt.txt"])
    result = await PromptAnalyzer().analyze(ctx)

    assert result.template_name == "prompt.txt"
    assert result.clarity_score > 0
    assert result.prompt_complexity in ("simple", "moderate", "complex")


@pytest.mark.asyncio
async def test_prompt_analyzer_accepts_inline_template_content():
    ctx = ProjectContext(project_name="t", project_path="/tmp")
    ctx.prompt_templates = ["You are a helpful assistant.\nSummarize {document} clearly and return JSON with keys title and summary."]
    result = await PromptAnalyzer().analyze(ctx)

    # Inline content must be analyzed — not misread as a missing file path.
    assert result.template_name == "inline template"
    assert result.clarity_score > 0
    assert result.confidence != "very_low"


@pytest.mark.asyncio
async def test_prompt_analyzer_no_templates_reports_state():
    ctx = ProjectContext(project_name="t", project_path="/tmp")
    result = await PromptAnalyzer().analyze(ctx)
    assert result.template_name == "none"
    assert result.confidence == "very_low"
    assert any("No prompt templates" in i for i in result.detected_issues)


# ---------------------------------------------------------------------------
# Prediction Engine (no more stub)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prediction_engine_derives_from_real_results():
    engine = PredictionEngine()
    ds = _good_dataset_result()
    hp = HyperparameterAnalysisResult(learning_rate=5e-4, batch_size=8, epochs=3, efficiency_score=0.85)
    model = ModelAnalysisResult(
        selected_model="llama-3-8b", parameter_count="8B", context_length=8192,
        estimated_vram="16GB", confidence="high",
    )
    result = await engine.predict(ProjectContext(project_name="t", project_path="/tmp"), ds, hp, model)

    assert result.instruction_following_prediction == "good"
    assert result.hallucination_risk == "low"
    assert result.reasoning_prediction == "good"
    assert result.confidence == "high"


@pytest.mark.asyncio
async def test_prediction_engine_not_always_good():
    engine = PredictionEngine()
    ds = _good_dataset_result()
    ds.sample_count = 10  # tiny dataset
    ds.duplicate_percentage = 40.0
    hp = HyperparameterAnalysisResult(learning_rate=5e-3, batch_size=2, epochs=20, efficiency_score=0.3)
    model = ModelAnalysisResult(
        selected_model="unknown-model", parameter_count="unknown", context_length=0,
        estimated_vram="unknown", confidence="low",
    )
    result = await engine.predict(ProjectContext(project_name="t", project_path="/tmp"), ds, hp, model)

    assert result.hallucination_risk == "high"
    assert result.reasoning_prediction == "unknown"
    assert result.likely_failure_modes  # real failure modes listed
    assert result.confidence in ("low", "medium")


# ---------------------------------------------------------------------------
# Report Generator (no hardcoded scores)
# ---------------------------------------------------------------------------

def test_report_scores_are_derived_not_constant():
    gen = ReportGenerator()
    ds = _good_dataset_result()
    hp = HyperparameterAnalysisResult(learning_rate=5e-4, batch_size=8, epochs=3, efficiency_score=0.9, confidence="high")
    model = ModelAnalysisResult(
        selected_model="llama-3-8b", parameter_count="8B", context_length=8192,
        estimated_vram="16GB", confidence="high",
    )

    good = gen._compute_readiness_score(ds, hp, model)
    assert good != 0.85  # old hardcoded value
    assert 0.0 <= good <= 1.0

    empty = gen._compute_readiness_score(
        DatasetAnalysisResult(
            dataset_name="none", sample_count=0, token_count=0,
            average_prompt_length=0, average_response_length=0,
            duplicate_percentage=0, near_duplicate_percentage=0,
            missing_field_percentage=0, formatting_consistency_score=0,
            language_consistency_score=0, instruction_consistency_score=0,
            response_consistency_score=0, quality_score=0, confidence="very_low",
        ),
        HyperparameterAnalysisResult(efficiency_score=0.5, confidence="very_low"),
        ModelAnalysisResult(selected_model="unknown", parameter_count="unknown",
                            context_length=0, estimated_vram="unknown", confidence="low"),
    )
    assert empty < good
    assert empty != 0.6  # old hardcoded value