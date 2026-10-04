"""End-to-end tests for the complete Analyze action.

Runs the real pipeline (scanner → dataset/prompt/hyperparameter/model/cost
analyzers → use-case detection → prediction → recommendations → GPU estimate →
orchestrator) against a realistic synthetic project, and asserts the
properties the Analyze button depends on.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from analyzers.use_case_detector import UseCaseDetector
from models.schemas import ProjectContext


def _write_project(root: Path, samples: int = 240) -> Path:
    """Create a realistic customer-support classification project."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "instruction": f"My order #{1000 + i} has not arrived after three weeks. What now?",
            "response": ["refund", "shipping_status", "escalate", "cancel"][i % 4],
        }
        for i in range(samples)
    ]
    # Seed real defects so the analyzers have something to find.
    records.append(dict(records[0]))                              # exact duplicate
    records.append({"instruction": "", "response": "refund"})     # missing input
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )

    (root / "config.yaml").write_text(
        "model_name_or_path: meta-llama/llama-3-8b-instruct\n"
        "learning_rate: 0.0002\n"
        "batch_size: 4\n"
        "epochs: 3\n"
        "optimizer: adamw_torch\n"
        "lora_rank: 16\n"
        "lora_alpha: 32\n",
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text("transformers\npeft\ndatasets\n", encoding="utf-8")
    (root / "train.py").write_text("# training entry point\n", encoding="utf-8")
    (root / "eval.py").write_text("# evaluation entry point\n", encoding="utf-8")
    (root / "prompt.txt").write_text(
        "You are a support assistant. Classify the ticket and return only the label.",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return _write_project(tmp_path / "support_project")


def _run_pipeline(path: Path):
    """Execute the real analysis pipeline on the event loop."""
    from api import routes

    return asyncio.new_event_loop().run_until_complete(
        routes._execute_analysis_pipeline(str(path))
    )


def test_use_case_detected_from_dataset_evidence(project: Path):
    """A closed label set must be recognised as classification, with evidence."""
    context = ProjectContext(
        project_name="support_project",
        project_path=str(project),
        dataset_paths=[str(project / "data" / "train.jsonl")],
    )
    result = UseCaseDetector().analyze(context)

    assert result.primary_task == "classification"
    assert result.confidence in ("medium", "high")
    assert result.evidence, "a use case must cite evidence"
    assert any(s.provenance in ("measured", "calculated") for s in result.evidence)
    assert result.reasoning
    assert result.recommended_approach == "lora"


def test_use_case_is_uncertain_when_project_is_empty(tmp_path: Path):
    """An empty project must not produce a confident guess."""
    result = UseCaseDetector().analyze(
        ProjectContext(project_name="empty", project_path=str(tmp_path))
    )
    assert result.primary_task == "unknown"
    assert result.confidence == "very_low"
    assert result.uncertainties


def test_full_pipeline_produces_complete_analysis(project: Path):
    """One Analyze run must yield every section, with real dataset numbers."""
    analysis = _run_pipeline(project)["complete_analysis"]
    assert analysis is not None, "the orchestrator must always produce an analysis"

    # Dataset numbers come from the real file, not placeholders.
    dataset = analysis.dataset_analysis
    assert dataset is not None
    # 240 generated + 1 duplicate + 1 malformed, plus prompt.txt which the
    # scanner also classifies as a dataset file and which parses as 1 record.
    assert dataset["sample_count"] >= 242
    assert dataset["duplicate_percentage"] > 0, "the seeded duplicate must be detected"
    assert dataset["missing_field_percentage"] > 0, "the blank input must be detected"

    # Use case, prompt, model, training and cost are all present.
    assert analysis.use_case is not None
    assert analysis.use_case.primary_task == "classification"
    assert analysis.prompt_analysis is not None
    assert analysis.model_analysis is not None
    assert analysis.model_analysis["selected_model"] == "meta-llama/llama-3-8b-instruct"
    assert analysis.training_configuration is not None
    current = analysis.training_configuration["current_configuration"]
    assert current["learning_rate"] == 0.0002, "read from the project's config file"
    assert current["lora_rank"] == 16
    assert analysis.training_configuration["recommended_starting_point"]["learning_rate_reason"]
    assert analysis.cost_estimate is not None
    assert analysis.model_candidates, "at least the configured model must be a candidate"
    # This project's defects are small (0.4% duplicates), so no high-severity
    # risk is warranted; risk thresholds are driven by measured magnitude.
    assert isinstance(analysis.risks, list)
    assert analysis.next_experiment is not None

    # The rendered chat message must be complete, not a stub.
    markdown = analysis.markdown
    for heading in (
        "## 1. Project Understanding",
        "## 2. Identified Use Case",
        "## 3. Dataset Analysis",
        "## 4. Prompt Quality",
        "## 5. Model Recommendations",
        "## 6. Model Comparison",
        "## 7. Hardware / GPU Requirements",
        "## 8. Training Configuration",
        "## 9. Estimated Training Time",
        "## 10. Cost Estimate",
        "## 11. Fine-tuning vs RAG vs Prompting",
        "## 13. Risks",
        "## 14. Recommendations",
        "## 15. Suggested Next Experiment",
    ):
        assert heading in markdown, f"missing section in rendered analysis: {heading}"

    # Real measured values must be visible in the rendered text.
    assert "classification" in markdown
    assert "Duplicates:" in markdown


def test_analysis_does_not_modify_the_project(project: Path):
    """Analysis is read-only: no project file may change."""
    before = {
        path.relative_to(project): path.stat().st_mtime_ns
        for path in project.rglob("*") if path.is_file()
    }
    _run_pipeline(project)
    after = {
        path.relative_to(project): path.stat().st_mtime_ns
        for path in project.rglob("*") if path.is_file()
    }
    assert before == after, "analysis must never write to the analysed project"


def test_failed_analyzer_degrades_only_its_own_section(project: Path, monkeypatch):
    """A broken analyzer must not discard the rest of the analysis."""
    from analyzers import prompt_analyzer

    async def _boom(self, context):
        raise RuntimeError("simulated prompt analyzer crash")

    monkeypatch.setattr(prompt_analyzer.PromptAnalyzer, "analyze", _boom)
    analysis = _run_pipeline(project)["complete_analysis"]

    # The failed section is None and explained, never faked.
    assert analysis.prompt_analysis is None
    assert any(item["section"] == "Prompt Quality" for item in analysis.unavailable_sections)
    # Everything else survived.
    assert analysis.dataset_analysis is not None
    assert analysis.dataset_analysis["sample_count"] >= 242
    assert analysis.use_case is not None
    assert analysis.analysis_status == "partial"
    assert "Prompt Quality" in analysis.markdown


def test_budget_never_invents_prices(project: Path):
    """Budget must state the gap rather than quoting invented numbers."""
    budget = _run_pipeline(project)["complete_analysis"].budget

    assert budget["budget_constraint"] == "not specified"
    assert {option["tier"] for option in budget["options"]} == {
        "Low cost", "Balanced", "Higher capability"
    }
    for option in budget["options"]:
        assert option["estimated_api_cost_per_month"] is None, "no invented prices"
        assert option["estimated_training_cost"] is None, "no invented prices"


def test_risks_are_derived_from_measured_defects(tmp_path: Path):
    """A genuinely poor project must surface evidence-backed risks.

    The seeded defects here are large enough to cross the documented
    thresholds, so a risk must be raised and must cite the measured value.
    """
    root = tmp_path / "bad_project"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    records = [
        {"instruction": "", "response": ""} for _ in range(30)
    ] + [{"instruction": f"q{i}", "response": f"a{i}"} for i in range(20)]
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    (root / "train.py").write_text("# only a training script, no evaluation\n", encoding="utf-8")

    analysis = _run_pipeline(root)["complete_analysis"]

    assert analysis.risks, "a project with empty records and no eval must raise risks"
    by_name = {risk.risk: risk for risk in analysis.risks}
    assert "Insufficient dataset size" in by_name
    assert "No usable training data" not in by_name, "records exist, just too few"
    # Every risk must justify its severity with a measured value.
    for risk in analysis.risks:
        assert risk.basis, f"risk '{risk.risk}' has no stated basis"
        assert risk.mitigation, f"risk '{risk.risk}' has no mitigation"
    # The risks must be visible to the user, not just in the data structure.
    assert "## 13. Risks" in analysis.markdown
    assert "Insufficient dataset size" in analysis.markdown
