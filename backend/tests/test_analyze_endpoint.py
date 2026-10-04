"""HTTP-level end-to-end verification of the Analyze action.

Traces the exact path the Analyze button takes:
  POST /api/v1/project/analyze
  -> project discovery -> dataset analysis -> use-case identification
  -> prompt analysis -> model recommendation -> hardware analysis
  -> training analysis -> GPU/time estimation -> cost estimation
  -> prediction -> risks -> recommendations -> final chat response

It asserts the response the chat window actually receives is a complete,
project-specific analysis rather than a single analyzer's result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A realistic summarization project."""
    data = tmp_path / "summarizer"
    (data / "data").mkdir(parents=True)
    rows = [
        {
            "article": f"Long policy document number {i} describing the process in detail. " * 8,
            "summary": f"Summary of policy document {i}.",
        }
        for i in range(150)
    ]
    (data / "data" / "docs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    (data / "config.yaml").write_text(
        "model_name_or_path: mistralai/mistral-7b-instruct\n"
        "learning_rate: 0.0001\nbatch_size: 2\nepochs: 2\n",
        encoding="utf-8",
    )
    (data / "train.py").write_text("# training\n", encoding="utf-8")
    (data / "eval.py").write_text("# evaluation\n", encoding="utf-8")
    (data / "requirements.txt").write_text("transformers\npeft\n", encoding="utf-8")
    return data


def test_analyze_endpoint_returns_complete_chat_analysis(project: Path, monkeypatch):
    """The one click must return the whole analysis, ready for the chat."""
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(project.parent))

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    response = client.post("/api/v1/project/analyze", json={"projectPath": str(project)})
    assert response.status_code == 200, response.text

    body = response.json()

    # The composed analysis is present and is the authoritative artifact.
    assert "analysis" in body, "Analyze must return the complete analysis"
    analysis = body["analysis"]
    assert analysis["markdown"], "the chat needs rendered output"
    assert analysis["project_name"] == "summarizer"

    # Not a single analyzer's result: every section is resolved.
    assert analysis["use_case"] is not None
    assert analysis["use_case"]["primary_task"] == "summarization"
    assert analysis["dataset_analysis"]["sample_count"] >= 150
    assert analysis["model_analysis"]["selected_model"] == "mistralai/mistral-7b-instruct"
    assert analysis["training_configuration"]["current_configuration"]["learning_rate"] == 0.0001
    assert analysis["cost_estimate"] is not None
    assert analysis["budget"]["budget_constraint"] == "not specified"
    assert isinstance(analysis["risks"], list)

    # The chat message answers the questions the user would otherwise ask.
    markdown = analysis["markdown"]
    for section in (
        "Identified Use Case",
        "Dataset Analysis",
        "Prompt Quality",
        "Model Recommendations",
        "Hardware / GPU Requirements",
        "Training Configuration",
        "Estimated Training Time",
        "Cost Estimate",
        "Fine-tuning vs RAG vs Prompting",
        "Risks",
        "Suggested Next Experiment",
    ):
        assert section in markdown, f"missing section: {section}"

    # No placeholders or pending states.
    assert "pending" not in markdown.lower()
    assert "TODO" not in markdown
    assert "placeholder" not in markdown.lower()


def test_training_time_estimate_uses_the_measured_dataset_size(project: Path):
    """The GPU estimate must be driven by measured data, not a zero-sample default.

    ``extract_training_config_from_context`` can only see a sample count if the
    scanner happened to record one, which it does not. Without the pipeline
    feeding the DatasetAnalyzer result in, the estimator silently reports
    0 seconds / 0 steps — a fake "instant" answer.
    """
    import asyncio

    from api import routes

    pipeline = asyncio.new_event_loop().run_until_complete(
        routes._execute_analysis_pipeline(str(project))
    )
    estimate = pipeline["gpu_time_estimate"]
    assert estimate is not None
    assert estimate["total_steps"] > 0, "0 steps means the dataset size never reached the estimator"
    assert estimate["estimated_seconds"] > 0, "0 seconds is a degenerate estimate, not a fast run"
    assert "unknown" not in str(estimate.get("estimated_time", "")).lower()


def test_analyze_endpoint_rejects_unauthorized_path(tmp_path: Path, monkeypatch):
    """The new analysis path must not weaken workspace authorization."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(allowed))

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    forbidden = tmp_path / "elsewhere"
    forbidden.mkdir()
    response = client.post("/api/v1/project/analyze", json={"projectPath": str(forbidden)})
    assert response.status_code in (400, 403)
    assert "analysis" not in response.json()


def _demo_project(root: Path) -> Path:
    """Build a small ticket-classification project for the manual demo."""
    (root / "data").mkdir(parents=True)
    rows = [
        {
            "instruction": f"My ticket #{i} is still unresolved after days. What should I do?",
            "response": ["refund", "escalate", "status", "cancel"][i % 4],
        }
        for i in range(150)
    ]
    rows.append({"instruction": "", "response": "refund"})
    (root / "data" / "tickets.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    (root / "config.yaml").write_text(
        "model_name_or_path: meta-llama/llama-3-8b-instruct\n"
        "learning_rate: 0.0002\nbatch_size: 4\nepochs: 3\nlora_rank: 16\n",
        encoding="utf-8",
    )
    (root / "train.py").write_text("# training\n", encoding="utf-8")
    (root / "eval.py").write_text("# evaluation\n", encoding="utf-8")
    (root / "requirements.txt").write_text("transformers\npeft\n", encoding="utf-8")
    return root


if __name__ == "__main__":
    """Write the exact analysis the chat window receives, for manual review.

    Written to a UTF-8 file rather than stdout: the analysis contains box-drawing
    and warning glyphs that a Windows console cannot encode.
    """
    import asyncio
    import tempfile

    from api.routes import _execute_analysis_pipeline

    root = _demo_project(Path(tempfile.mkdtemp()) / "demo")
    pipeline = asyncio.new_event_loop().run_until_complete(
        _execute_analysis_pipeline(str(root))
    )
    out = Path("analysis_preview.md")
    out.write_text(pipeline["complete_analysis"].markdown, encoding="utf-8")
    print(f"Wrote {out.resolve()}")
