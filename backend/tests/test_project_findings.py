"""Project analysis reads the real code: scripts, base model, method, settings, runtime, cost."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from analysis.code_inspector import inspect_project, parse_param_count
from analysis.project_findings import (
    HardwareProfile,
    build_project_findings,
    parse_hardware_statement,
    render_findings,
)
from api.routes import router

FINE_TUNE = '''
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EPOCHS = 3
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
args = TrainingArguments(output_dir="out", num_train_epochs=EPOCHS, per_device_train_batch_size=4,
                         gradient_accumulation_steps=4, learning_rate=2e-4, bf16=True)
trainer = SFTTrainer(model=model, args=args, max_seq_length=512)
trainer.train()
'''


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "tiny"
    (root / "data").mkdir(parents=True)
    rows = [{"instruction": f"Question {i}?", "output": f"Answer number {i} " * 6} for i in range(400)]
    (root / "data" / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (root / "fine_tune.py").write_text(FINE_TUNE, encoding="utf-8")
    return root


def test_inspector_finds_fine_tune_script_model_and_method(project: Path):
    result = inspect_project(str(project)).to_dict()
    assert [s["path"] for s in result["training_scripts"]] == ["fine_tune.py"]
    assert result["base_model"]["value"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert result["base_model"]["file"] == "fine_tune.py"
    assert result["parameter_count_billions"] == pytest.approx(1.1)
    assert result["method"] == "qlora"
    assert result["lora"]["r"]["value"] == 16
    args = result["training_args"]
    assert args["num_train_epochs"]["value"] == 3  # resolved through the EPOCHS constant
    assert args["per_device_train_batch_size"]["value"] == 4
    assert args["learning_rate"]["value"] == pytest.approx(2e-4)


def test_inspector_does_not_invent_a_model(tmp_path: Path):
    (tmp_path / "notes.py").write_text("print('hello')\n", encoding="utf-8")
    result = inspect_project(str(tmp_path)).to_dict()
    assert result["training_scripts"] == []
    assert not result["base_model"]
    assert result["method"] == "unknown"


def test_parse_param_count():
    assert parse_param_count("TinyLlama/TinyLlama-1.1B-Chat-v1.0") == pytest.approx(1.1)
    assert parse_param_count("meta-llama/Llama-3.1-8B") == pytest.approx(8.0)
    assert parse_param_count("my-custom-model") is None


def _findings(project: Path, profile: HardwareProfile, detected=None):
    from scanner.context_builder import ContextBuilder
    from scanner.scanner import ProjectScanner

    context = ContextBuilder().build(ProjectScanner(str(project)).scan())
    return build_project_findings(
        context, dataset_samples=400, detected_hardware=detected or {"gpus": []}, profile=profile,
    )


def test_cloud_gpu_with_rate_gives_a_cost_with_assumptions(project: Path):
    profile = HardwareProfile.from_dict({"mode": "cloud", "gpuName": "A100 80GB", "hourlyRateUsd": 2.0})
    findings, estimate, _ = _findings(project, profile)
    assert estimate is not None and estimate.estimated_seconds > 0
    cost = findings["cost"]
    assert cost["hourly_rate_usd"] == 2.0
    assert cost["estimate_usd"] == pytest.approx(estimate.estimated_seconds / 3600 * 2.0, abs=0.01)
    assert cost["low_usd"] <= cost["estimate_usd"] <= cost["high_usd"]
    assert any("you provided" in a for a in cost["assumptions"])
    text = "\n".join(render_findings(findings))
    assert "TinyLlama" in text and "fine_tune.py" in text and "QLoRA" in text


def test_cloud_gpu_without_rate_shows_no_price(project: Path):
    profile = HardwareProfile.from_dict({"mode": "cloud", "gpuName": "A100 80GB"})
    findings, estimate, _ = _findings(project, profile)
    assert estimate is not None
    assert findings["cost"] is None
    assert any("hourly rate" in m for m in findings["missing"])


def test_local_gpu_never_gets_a_rental_cost(project: Path):
    detected = {"gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "vram_mb": 12288}],
                "detection_method": "nvidia-smi"}
    # Even if a rate is lying around, a local GPU has no rental cost.
    profile = HardwareProfile.from_dict({"mode": "local", "hourlyRateUsd": 2.0})
    findings, estimate, _ = _findings(project, profile, detected)
    assert findings["hardware"]["mode"] == "local"
    assert findings["cost"] is None
    assert estimate is not None


def test_no_hardware_means_no_invented_runtime_or_price(project: Path):
    findings, estimate, _ = _findings(project, HardwareProfile())
    assert estimate is None
    assert findings["time"] is None and findings["cost"] is None
    assert findings["missing"]


def test_unknown_gpu_name_is_not_guessed(project: Path):
    profile = HardwareProfile.from_dict({"mode": "cloud", "gpuName": "ZX-9000", "hourlyRateUsd": 1.0})
    findings, estimate, _ = _findings(project, profile)
    assert estimate is None and findings["cost"] is None


def test_parse_hardware_statement():
    p = parse_hardware_statement("I train on a RunPod A100 at $1.89/hr")
    assert p.mode == "cloud" and p.hourly_rate_usd == 1.89 and "A100" in p.gpu_name
    assert p.provider == "Runpod"
    local = parse_hardware_statement("I'm using my own RTX 3060 locally")
    assert local.mode == "local" and local.hourly_rate_usd is None
    assert parse_hardware_statement("Please change epochs from 3 to 5") is None
    assert parse_hardware_statement("what does the A100 cost?") is None


def test_hardware_profile_validation_rejects_bad_values():
    p = HardwareProfile.from_dict({"mode": "weird", "hourlyRateUsd": -5, "gpuCount": 9999})
    assert p.mode == "auto" and p.hourly_rate_usd is None and p.gpu_count == 64
    assert HardwareProfile.from_dict("junk").is_set is False


def test_analyze_endpoint_reports_code_findings_and_cloud_cost(project: Path, monkeypatch):
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(project.parent))
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)
    response = client.post("/api/v1/project/analyze", json={
        "projectPath": str(project),
        "hardware": {"mode": "cloud", "gpuName": "A100 80GB", "hourlyRateUsd": 1.5},
    })
    assert response.status_code == 200, response.text
    analysis = response.json()["analysis"]
    md = analysis["markdown"]
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in md
    assert "fine_tune.py" in md
    assert analysis["project_findings"]["cost"]["hourly_rate_usd"] == 1.5
    assert "$1.50" in md
    assert analysis["training_time_estimate"]["estimated_seconds"] > 0


def test_analyze_endpoint_local_gpu_has_no_cloud_pricing(project: Path, monkeypatch):
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(project.parent))
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)
    response = client.post("/api/v1/project/analyze", json={
        "projectPath": str(project),
        "hardware": {"mode": "local", "gpuName": "RTX 3060"},
    })
    assert response.status_code == 200, response.text
    md = response.json()["analysis"]["markdown"]
    assert "own GPU" in md
    assert "per GPU-hour" not in md and "/hour" not in md
