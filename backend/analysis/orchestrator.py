"""End-to-end analysis orchestrator — one Analyze click, one complete answer.

This module is an *orchestration layer*, not a new source of truth.  It accepts
the real outputs of the existing pipeline (scanner, dataset/prompt/
hyperparameter/model/cost analyzers, prediction engine, recommendation engine,
GPU detector, GPU time estimator) and composes them into the single
authoritative analysis that the Analyze action presents.

Design rules enforced here:

* **No duplicate logic.**  Dataset statistics, model specs, cost and timing
  numbers are *taken* from the existing components, never recomputed.
* **No fabrication.**  A section that could not be produced is ``None`` and is
  listed in ``unavailable_sections`` with a reason.  Missing data is never
  replaced with a plausible default.
* **Provenance is explicit.**  Values are labelled measured / calculated /
  heuristic / assumed so the user can tell facts from estimates.
* **Partial failure is normal.**  Each section is built in isolation; one
  failing analyzer degrades that section only, and the run is reported as
  ``partial`` rather than discarded.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from models.schemas import (
    BudgetOption,
    CompleteAnalysis,
    ModelCandidate,
    ProjectContext,
    RiskItem,
    UseCaseAnalysisResult,
)
from analyzers.model_advisor import _MODEL_DATABASE

logger = logging.getLogger(__name__)

#: Disclaimer attached whenever model/pricing facts come from the bundled
#: reference database rather than a live lookup.
REFERENCE_DATA_NOTICE = (
    "Model specifications, context limits and hardware requirements below come from the "
    "agent's bundled reference database and the project's own configuration files — not "
    "from a live provider lookup. Treat pricing and availability as possibly outdated and "
    "verify against the provider's official documentation before committing budget."
)


def _pct(value: Any, digits: int = 1) -> str:
    """Format a value that is ALREADY a 0-100 percentage, tolerating None/strings."""
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "unknown"


def _ratio_pct(value: Any, digits: int = 1) -> str:
    """Format a 0-1 ratio (0.9) as a percentage ("90.0%"), tolerating None/strings.

    Scores such as ``quality_score`` and ``clarity_score`` are ratios in [0, 1],
    while ``duplicate_percentage`` is already in [0, 100]. Passing a ratio to
    :func:`_pct` rendered a perfect 1.0 as "1%", which understates every score
    in the report by a factor of 100. This helper keeps the two scales distinct.
    """
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "unknown"


def _num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unknown"


class AnalysisOrchestrator:
    """Composes real pipeline outputs into the complete analysis result."""

    def build(
        self,
        context: ProjectContext,
        results: Dict[str, Any],
        *,
        report: Any = None,
        prediction: Any = None,
        recommendations: Optional[List[Any]] = None,
        gpu_estimate: Any = None,
        hardware: Any = None,
        use_case: Optional[UseCaseAnalysisResult] = None,
        failures: Optional[List[Dict[str, Any]]] = None,
        analysis_status: str = "unknown",
    ) -> CompleteAnalysis:
        """Assemble the complete analysis from the pipeline's real outputs."""
        failures = failures or []
        unavailable: List[Dict[str, str]] = []

        dataset = self._dataset_section(results, unavailable)
        prompt = self._prompt_section(results, unavailable)
        model = self._model_section(results, unavailable)
        hardware_section = self._hardware_section(hardware, unavailable)
        training = self._training_section(results, model, unavailable)
        timing = self._timing_section(gpu_estimate, unavailable)
        cost = self._cost_section(results, unavailable)
        budget = self._budget_section(results, model, hardware_section, cost, context)

        analysis = CompleteAnalysis(
            project_name=context.project_name,
            generated_at=datetime.now(timezone.utc),
            analysis_status=analysis_status,
            partial_failures=failures,
            project_understanding=self._project_section(context),
            use_case=use_case,
            dataset_analysis=dataset,
            prompt_analysis=prompt,
            model_analysis=model,
            model_candidates=self._model_candidates(model, context, hardware_section),
            model_comparison=self._model_comparison(model, context, hardware_section),
            hardware=hardware_section,
            training_configuration=training,
            training_time_estimate=timing,
            cost_estimate=cost,
            budget=budget,
            expected_behavior=self._behavior_section(prediction, unavailable),
            approach_recommendation=self._approach_section(use_case, prompt, dataset, context),
            risks=self._risks(results, prediction, hardware_section, context, use_case),
            recommendations=[self._recommendation_dict(r) for r in (recommendations or [])],
            next_experiment=self._next_experiment(use_case, results, model, training),
            model_information_notice=REFERENCE_DATA_NOTICE,
            unavailable_sections=unavailable,
        )

        analysis.markdown = self.render_markdown(analysis)
        return analysis

    # ------------------------------------------------------------------
    # Section builders — each reads existing analyzer output, never recomputes
    # ------------------------------------------------------------------
    def _project_section(self, context: ProjectContext) -> Dict[str, Any]:
        """Project understanding, straight from the scanner's measured findings."""
        return {
            "project_name": context.project_name,
            "project_path": context.project_path,
            "detected_framework": context.detected_framework or "not detected",
            "framework_version": context.framework_version,
            "base_model": context.base_model or "not declared in the project",
            "tokenizer": context.tokenizer,
            "dataset_files": context.dataset_paths,
            "prompt_templates": context.prompt_templates,
            "configuration_files": context.configuration_files,
            "training_scripts": context.training_scripts,
            "evaluation_scripts": context.evaluation_scripts,
            "inference_scripts": context.inference_scripts,
            "file_count": (context.project_statistics or {}).get("file_count"),
            "note": (
                "Read-only inspection: no project file was modified during this analysis."
            ),
        }

    def _dataset_section(
        self, results: Dict[str, Any], unavailable: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        raw = results.get("dataset")
        if not isinstance(raw, dict) or raw.get("error"):
            unavailable.append({
                "section": "Dataset Analysis",
                "reason": (raw or {}).get("error", "the dataset analyzer did not return a result"),
            })
            return None
        return {
            "dataset_name": raw.get("dataset_name"),
            "sample_count": raw.get("sample_count"),
            "token_count": raw.get("token_count"),
            "average_prompt_length": raw.get("average_prompt_length"),
            "average_response_length": raw.get("average_response_length"),
            "duplicate_percentage": raw.get("duplicate_percentage"),
            "near_duplicate_percentage": raw.get("near_duplicate_percentage"),
            "missing_field_percentage": raw.get("missing_field_percentage"),
            "formatting_consistency_score": raw.get("formatting_consistency_score"),
            "language_consistency_score": raw.get("language_consistency_score"),
            "quality_score": raw.get("quality_score"),
            "findings": raw.get("findings", []),
            "warnings": raw.get("warnings", []),
            "recommendations": raw.get("recommendations", []),
            "confidence": raw.get("confidence"),
            "provenance": "measured",
        }

    def _prompt_section(
        self, results: Dict[str, Any], unavailable: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        raw = results.get("prompt")
        if not isinstance(raw, dict) or raw.get("error"):
            unavailable.append({
                "section": "Prompt Quality",
                "reason": (raw or {}).get("error", "the prompt analyzer did not return a result"),
            })
            return None
        return {
            "template_name": raw.get("template_name"),
            "clarity_score": raw.get("clarity_score"),
            "ambiguity_score": raw.get("ambiguity_score"),
            "formatting_score": raw.get("formatting_score"),
            "instruction_quality_score": raw.get("instruction_quality_score"),
            "consistency_score": raw.get("consistency_score"),
            "prompt_complexity": raw.get("prompt_complexity"),
            "detected_issues": raw.get("detected_issues", []),
            "recommendations": raw.get("recommendations", []),
            "confidence": raw.get("confidence"),
            "provenance": "measured",
        }

    def _model_section(
        self, results: Dict[str, Any], unavailable: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        raw = results.get("model")
        if not isinstance(raw, dict) or raw.get("error"):
            unavailable.append({
                "section": "Model Recommendations",
                "reason": (raw or {}).get("error", "the model advisor did not return a result"),
            })
            return None
        return {
            "selected_model": raw.get("selected_model"),
            "parameter_count": raw.get("parameter_count"),
            "context_length": raw.get("context_length"),
            "estimated_vram": raw.get("estimated_vram"),
            "reasoning_capability": raw.get("reasoning_capability"),
            "coding_capability": raw.get("coding_capability"),
            "instruction_following_capability": raw.get("instruction_following_capability"),
            "speed_score": raw.get("speed_score"),
            "memory_efficiency": raw.get("memory_efficiency"),
            "strengths": raw.get("strengths", []),
            "weaknesses": raw.get("weaknesses", []),
            "recommended_alternatives": raw.get("recommended_alternatives", []),
            "confidence": raw.get("confidence"),
            "provenance": "reference_data",
        }

    def _hardware_section(
        self, hardware: Any, unavailable: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Actual detected hardware, straight from the GPU detector."""
        if hardware is None:
            unavailable.append({
                "section": "Hardware / GPU Requirements",
                "reason": "GPU detection returned no result on this machine",
            })
            return None
        data = hardware.to_dict() if hasattr(hardware, "to_dict") else dict(hardware)
        gpus = data.get("gpus") or []
        vram_gb = [
            round((gpu.get("vram_mb") or 0) / 1024, 1)
            for gpu in gpus if gpu.get("vram_mb")
        ]
        return {
            "cuda_available": data.get("cuda_available"),
            "cuda_version": data.get("cuda_version"),
            "gpu_count": data.get("gpu_count", len(gpus)),
            "gpus": [
                {
                    "index": gpu.get("index"),
                    "name": gpu.get("name"),
                    "vram_gb": round((gpu.get("vram_mb") or 0) / 1024, 1) or None,
                    "driver_version": gpu.get("driver_version"),
                    "memory_utilization_percent": gpu.get("memory_utilization_percent"),
                }
                for gpu in gpus
            ],
            "max_vram_gb": max(vram_gb) if vram_gb else None,
            "total_vram_gb": round(sum(vram_gb), 1) if vram_gb else None,
            "detection_method": data.get("detection_method"),
            "error": data.get("error"),
            "provenance": "measured",
        }

    def _training_section(
        self,
        results: Dict[str, Any],
        model: Optional[Dict[str, Any]],
        unavailable: List[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        """Current configuration (from the user's files) + recommended starting point."""
        raw = results.get("hyperparameters")
        if not isinstance(raw, dict) or raw.get("error"):
            unavailable.append({
                "section": "Training Configuration",
                "reason": (raw or {}).get("error", "the hyperparameter analyzer returned no result"),
            })
            return None

        has_lora = raw.get("lora_rank") is not None
        params = _parse_params(model.get("parameter_count") if model else None)
        return {
            "current_configuration": {
                key: raw.get(key) for key in (
                    "learning_rate", "batch_size", "epochs", "optimizer", "scheduler",
                    "gradient_accumulation", "sequence_length", "weight_decay",
                    "warmup_ratio", "lora_rank", "lora_alpha", "lora_dropout",
                )
            },
            "provenance_of_current_values": "read from the project's config files, or missing",
            "efficiency_score": raw.get("efficiency_score"),
            "overfitting_risk": raw.get("overfitting_risk"),
            "underfitting_risk": raw.get("underfitting_risk"),
            "findings": raw.get("findings", []),
            "recommendations": raw.get("recommendations", []),
            "confidence": raw.get("confidence"),
            "recommended_starting_point": self._recommended_hyperparameters(
                has_lora=has_lora, params=params, model=model
            ),
        }

    def _recommended_hyperparameters(
        self,
        *,
        has_lora: bool,
        params: Optional[float],
        model: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Documented starting values, each labelled with why it is suggested."""
        method = "QLoRA" if ((params or 0) > 13 or (has_lora and (params or 0) > 7)) else "LoRA"
        context_length = int(model.get("context_length") or 0) if model else 0
        return {
            "method": method,
            "learning_rate": 2e-4 if has_lora else 1e-5,
            "learning_rate_reason": (
                "LoRA/QLoRA tolerates a ~10x higher learning rate than full fine-tuning; "
                "full fine-tuning starts near 1e-5 to avoid destroying pretrained weights."
            ),
            "epochs": 3,
            "epochs_reason": (
                "3 is the usual starting point; on a small dataset more epochs overfit "
                "quickly — validate on a held-out split before increasing."
            ),
            "batch_size": 4,
            "batch_size_reason": (
                "4 fits on most 12-24GB cards; raise it only if VRAM allows, or keep it "
                "and use gradient accumulation to reach the effective batch size."
            ),
            "gradient_accumulation": 4,
            "gradient_accumulation_reason": (
                "Reaches an effective batch size of ~16 without extra VRAM, which "
                "stabilises small-dataset training."
            ),
            "sequence_length": (
                min(2048, max(256, context_length // 4)) if context_length else 512
            ),
            "sequence_length_reason": (
                "Derived from the model's context window and capped at 2048. It must cover "
                "the measured average prompt + response length."
            ),
            "precision": "bf16 (fp16 if the GPU lacks bf16)",
            "precision_reason": "Half precision halves activation memory at negligible quality cost.",
            "optimizer": "AdamW",
            "optimizer_reason": "The conventional default for transformer fine-tuning.",
            "gradient_checkpointing": True,
            "gradient_checkpointing_reason": (
                "Trades ~30% extra compute for a large activation-memory reduction — the "
                "most effective lever when VRAM is the binding constraint."
            ),
            "quantization": "4-bit (QLoRA)" if method == "QLoRA" else "not required",
            "quantization_reason": (
                "4-bit base weights cut the memory needed to train a larger model on a "
                "consumer GPU; LoRA adapters stay at full precision."
            ),
            "provenance": (
                "heuristic — documented starting values, not measured for this project"
            ),
        }

    def _timing_section(
        self,
        gpu_estimate: Any,
        unavailable: List[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        """Training-time estimate, taken from the existing GPU time estimator."""
        if gpu_estimate is None:
            unavailable.append({
                "section": "Estimated Training Time",
                "reason": "no GPU could be detected, so wall-clock time cannot be estimated",
            })
            return None
        data = gpu_estimate.to_dict() if hasattr(gpu_estimate, "to_dict") else dict(gpu_estimate)
        return {
            "estimated_time": data.get("estimated_time"),
            "expected_range": data.get("range"),
            "estimated_seconds": data.get("estimated_seconds"),
            "total_steps": data.get("total_steps"),
            "throughput_samples_per_sec": data.get("throughput_samples_per_sec"),
            "throughput_steps_per_sec": data.get("throughput_steps_per_sec"),
            "vram_feasible": data.get("vram_feasible"),
            "vram_estimated_gb": data.get("vram_estimated_gb"),
            "vram_available_gb": data.get("vram_available_gb"),
            "confidence": data.get("confidence"),
            "mode": data.get("mode"),
            "assumptions": data.get("assumptions", []),
            "warnings": data.get("warnings", []),
            "uncertainty": (
                "This is a FLOPs-based projection, not a measured run. Real wall-clock time "
                "depends on data loading, kernel efficiency and dataloader workers. Run a "
                "short calibration benchmark for a measured baseline."
            ),
            "provenance": "calculated from measured model/dataset/GPU values with documented heuristics",
        }

    def _cost_section(
        self, results: Dict[str, Any], unavailable: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Resource cost, taken from the existing CostEstimator."""
        raw = results.get("cost")
        if not isinstance(raw, dict) or raw.get("error"):
            unavailable.append({
                "section": "Cost Estimate",
                "reason": (raw or {}).get("error", "the cost estimator returned no result"),
            })
            return None
        return {
            "estimated_training_time": raw.get("estimated_training_time"),
            "estimated_gpu_hours": raw.get("estimated_gpu_hours"),
            "estimated_vram_usage": raw.get("estimated_vram_usage"),
            "estimated_checkpoint_size": raw.get("estimated_checkpoint_size"),
            "estimated_storage_requirement": raw.get("estimated_storage_requirement"),
            "compatible_hardware": raw.get("compatible_hardware", []),
            "assumptions": raw.get("assumptions", []),
            "confidence": raw.get("confidence"),
            "note": (
                "These are self-hosted training resources (time, VRAM, disk). API and "
                "hosting prices are not included — see the Budget section."
            ),
            "provenance": "calculated from measured model/dataset size with documented formulas",
        }

    def _budget_section(
        self,
        results: Dict[str, Any],
        model: Optional[Dict[str, Any]],
        hardware: Optional[Dict[str, Any]],
        cost: Optional[Dict[str, Any]],
        context: ProjectContext,
    ) -> Dict[str, Any]:
        """Budget tiers.

        The agent does **not** invent prices.  No dollar figure is produced here:
        the bundled reference data has no verified price table, and a fabricated
        number is worse than an explicit gap.  Instead the user gets the
        *structure* of the decision (what dominates cost, what the levers are)
        plus a clear statement that prices must be filled from the provider's
        official pricing page.
        """
        detected = [gpu.get("name") for gpu in (hardware or {}).get("gpus", []) if gpu.get("name")]
        options = [
            BudgetOption(
                tier="Low cost",
                approach="Prompt engineering + RAG over the existing data; no training run",
                rationale=(
                    "Costs the least because it requires no GPU time. Best when the gap is "
                    "missing knowledge in private documents rather than missing behaviour."
                ),
                assumptions=[
                    "No training cost incurred",
                    "Inference cost scales with request volume and context length",
                ],
            ),
            BudgetOption(
                tier="Balanced",
                approach=(
                    f"LoRA fine-tune {model.get('selected_model') or 'the selected model'}"
                    if model else "LoRA fine-tune the selected model"
                ),
                rationale=(
                    "A single LoRA run on measured data is a few GPU-hours on existing "
                    "hardware, teaching format/consistent behaviour at a fraction of the "
                    "cost of a full fine-tune."
                ),
                assumptions=[
                    "Training cost = one-off GPU-hours on hardware you already own",
                    "If you rent a GPU, multiply GPU-hours by the rental rate for your region",
                ],
            ),
            BudgetOption(
                tier="Higher capability",
                approach=(
                    "Larger base model or hosted API for higher quality, latency and "
                    "concurrency at higher per-token cost"
                ),
                rationale=(
                    "Raises capability ceiling and removes the GPU requirement, converting a "
                    "capital cost into a per-token operating cost."
                ),
                assumptions=[
                    "Cost becomes usage-proportional rather than one-off",
                    "Includes data-residency and vendor-lock-in trade-offs",
                ],
            ),
        ]

        return {
            "budget_constraint": "not specified",
            "detected_local_hardware": detected or ["no GPU detected on this machine"],
            "options": [option.model_dump() for option in options],
            "cost_drivers": [
                "API cost — scales with requests/month x tokens/request x provider rate",
                "Training cost — GPU-hours x (rental rate, or your own hardware/electricity)",
                "Inference cost — per-token, and higher for longer contexts and output",
                "Hosting cost — if the tuned model is served rather than run locally",
                "Local hardware cost — amortised GPU purchase, plus power draw",
                "Storage cost — checkpoints, datasets and logs",
            ],
            "pricing_note": (
                "No per-token prices are quoted here: the agent has no verified, current "
                "price table and will not guess. Fill these from the provider's official "
                "pricing page before committing budget."
            ),
            "local_inference_note": (
                "Running the model locally (e.g. Ollama) removes the per-token API cost but "
                "not the total cost: the GPU, its power draw, and your time are still real "
                "costs. Local inference is a cost/latency/privacy trade-off, not free."
            ),
        }

    def _model_candidates(
        self,
        model: Optional[Dict[str, Any]],
        context: ProjectContext,
        hardware: Optional[Dict[str, Any]],
    ) -> List[ModelCandidate]:
        """Candidates built from the advisor's output and the shared reference DB.

        Specs always come from ``_MODEL_DATABASE`` — the same source the
        ModelAdvisor uses — so this section never introduces a second model
        catalogue.  Detected VRAM is stated alongside each candidate instead of
        being used to silently drop options.
        """
        if not model:
            return []
        max_vram = (hardware or {}).get("max_vram_gb")
        candidates: List[ModelCandidate] = []
        seen: set = set()

        def add(name: str, why: str, best_use: str) -> None:
            if not name or name in seen:
                return
            seen.add(name)
            info = self._lookup_model(name)
            if not info:
                return
            candidates.append(ModelCandidate(
                model=name,
                why_it_fits=why,
                expected_strengths=list(info.get("strengths", [])),
                expected_limitations=list(info.get("weaknesses", [])),
                resource_requirements=(
                    f"{info.get('parameter_count', '?')} parameters, "
                    f"{info.get('context_length', '?')} context, VRAM "
                    f"{info.get('estimated_vram', 'unknown')}"
                    + (f" (detected GPU has {max_vram} GB)" if max_vram else " (no GPU detected)")
                ),
                cost_considerations=(
                    "Open-weight model: no per-token API fee if run locally; a hosted "
                    "provider may still serve it at provider rates."
                ),
                best_use_case=best_use,
                provenance="reference_data",
            ))

        selected = str(model.get("selected_model") or "")
        add(selected, "It is the model this project already declares and configures.",
            "Continue the current design")
        for alt in model.get("recommended_alternatives", []):
            add(self._match_catalogue(str(alt)), str(alt),
                "Alternative suggested by the model advisor")

        if not candidates and selected:
            candidates.append(ModelCandidate(
                model=selected,
                why_it_fits="Declared by the project but absent from the bundled reference data.",
                expected_limitations=["Specifications could not be verified from reference data"],
                resource_requirements="unknown — model not in the reference database",
                cost_considerations="unknown — verify pricing and availability with the provider",
                best_use_case="As configured by the project",
                provenance="reference_data",
            ))
        return candidates

    @staticmethod
    def _match_catalogue(phrase: str) -> str:
        """Map a prose alternative onto a catalogue key (no new entries)."""
        lowered = phrase.lower()
        for key in _MODEL_DATABASE:
            family = key.split("/")[-1].split("-instruct")[0]
            if family and family in lowered:
                return key
        return ""

    @staticmethod
    def _lookup_model(name: str) -> Optional[Dict[str, Any]]:
        """Resolve a model name against the shared reference database."""
        if not name:
            return None
        if name in _MODEL_DATABASE:
            return _MODEL_DATABASE[name]
        lowered = name.lower()
        for key, info in _MODEL_DATABASE.items():
            if lowered in key.lower() or key.lower() in lowered:
                return info
        return None

    def _model_comparison(
        self,
        model: Optional[Dict[str, Any]],
        context: ProjectContext,
        hardware: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Trade-offs between candidates, explicitly labelled as tendencies."""
        if not model:
            return None
        return {
            "candidates": [c.model for c in self._model_candidates(model, context, hardware)],
            "expected_tendencies": {
                "instruction_following": (
                    "Larger instruction-tuned models generally follow format and constraint "
                    "instructions more reliably; a fine-tuned small model can match a larger "
                    "base model on this specific task because it is specialised."
                ),
                "reasoning": (
                    "Reasoning quality generally tracks model scale and training compute. A "
                    "task-specific fine-tune improves format adherence more reliably than "
                    "it improves reasoning."
                ),
                "factuality": (
                    "Fine-tuning shapes behaviour; it does not add knowledge and can raise "
                    "hallucination on small or inconsistent data. For factual grounding, RAG "
                    "is the correct tool."
                ),
                "formatting_consistency": (
                    "The behaviour fine-tuning most reliably improves, because it is learned "
                    "from the dataset's own formatting."
                ),
                "verbosity": (
                    "Models differ in default verbosity; fine-tuning on consistent data usually "
                    "narrows it toward the dataset's style."
                ),
                "latency_and_throughput": (
                    "Driven mainly by parameter count and quantisation: smaller and quantised "
                    "models are faster, with a measurable quality trade-off."
                ),
                "context_handling": (
                    "Fixed by the model's context window, not by fine-tuning. Inputs beyond "
                    "that window need RAG or truncation, not a bigger fine-tune."
                ),
                "domain_adaptation": (
                    "Fine-tuning helps where the domain's phrasing and output conventions "
                    "differ from the base model's pretraining distribution."
                ),
                "cost": (
                    "Self-hosted cost is dominated by training GPU-hours on hardware you "
                    "already own; hosted cost is dominated by tokens served."
                ),
            },
            "disclaimer": (
                "These are expected tendencies based on general model behaviour, not benchmarks "
                "measured on your task. Only an evaluation set can confirm them."
            ),
        }

    def _behavior_section(
        self, prediction: Any, unavailable: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Expected post-training behaviour, from the prediction engine."""
        if prediction is None:
            unavailable.append({
                "section": "Expected Model Behaviour",
                "reason": "the prediction engine returned no result",
            })
            return None
        data = prediction.model_dump() if hasattr(prediction, "model_dump") else dict(prediction)
        return {
            "instruction_following": data.get("instruction_following_prediction"),
            "hallucination_risk": data.get("hallucination_risk"),
            "reasoning": data.get("reasoning_prediction"),
            "response_consistency": data.get("response_consistency_prediction"),
            "creativity": data.get("creativity_prediction"),
            "formatting": data.get("formatting_prediction"),
            "expected_strengths": data.get("expected_strengths", []),
            "expected_weaknesses": data.get("expected_weaknesses", []),
            "likely_failure_modes": data.get("likely_failure_modes", []),
            "status": data.get("status"),
            "confidence": data.get("confidence"),
            "confidence_basis": data.get("confidence_basis"),
            "uncertainty": data.get("uncertainty"),
            "unknowns": data.get("unknowns", []),
            "evidence": data.get("evidence", []),
            "quality_note": data.get("quality_note"),
            "evaluation_required": data.get("evaluation_required", True),
            "provenance": "heuristic, derived from the measured analyzer outputs",
        }

    def _approach_section(
        self,
        use_case: Optional[UseCaseAnalysisResult],
        prompt: Optional[Dict[str, Any]],
        dataset: Optional[Dict[str, Any]],
        context: ProjectContext,
    ) -> Dict[str, Any]:
        """Prompting vs RAG vs fine-tuning, decided from measured evidence."""
        clarity = (prompt or {}).get("clarity_score")
        samples = (dataset or {}).get("sample_count") or 0

        reasons: List[str] = []
        if clarity is not None and float(clarity) < 0.6:
            reasons.append(
                f"Prompt clarity scored {float(clarity):.0%}, so unclear instructions are a "
                "live contributor to poor output — fix the prompt before blaming the model."
            )
        if samples == 0:
            reasons.append("No usable training data was measured, so fine-tuning is not an option yet.")

        return {
            "recommended_approach": use_case.recommended_approach if use_case else "prompting",
            "rationale": (
                use_case.approach_rationale if use_case
                else "No use case could be inferred, so start with prompting."
            ),
            "decision_rules": [
                "Missing knowledge in private or frequently changing documents → RAG. "
                "Fine-tuning does not add or refresh knowledge.",
                "Inconsistent behaviour, style or output format → fine-tuning (LoRA/QLoRA). "
                "RAG cannot make output formatting consistent.",
                "Unclear instructions or a weak prompt → improve prompting first. Training on "
                "an ambiguous task encodes the ambiguity.",
                "Private data that must not leave the machine → local model plus RAG; any "
                "hosted API sends that data off-device.",
            ],
            "supporting_reasons": reasons,
            "use_case_confidence": use_case.confidence if use_case else "very_low",
        }

    def _risks(
        self,
        results: Dict[str, Any],
        prediction: Any,
        hardware: Optional[Dict[str, Any]],
        context: ProjectContext,
        use_case: Optional[UseCaseAnalysisResult],
    ) -> List[RiskItem]:
        """Risks derived from measured values; each carries its severity basis.

        Severities are not arbitrary scores: each one names the measured
        quantity that triggered it, so the user can disagree with the threshold
        and still act on the finding.
        """
        risks: List[RiskItem] = []
        dataset = results.get("dataset") or {}
        hp = results.get("hyperparameters") or {}

        samples = int(dataset.get("sample_count") or 0)
        if samples == 0:
            risks.append(RiskItem(
                risk="No usable training data",
                why="The dataset analyzer measured 0 readable records.",
                severity="critical",
                basis="sample_count = 0 (measured)",
                mitigation="Provide a valid JSONL/JSON/CSV/TSV dataset before any training run.",
            ))
        elif samples < 100:
            risks.append(RiskItem(
                risk="Insufficient dataset size",
                why=f"Only {samples} records were measured; below ~100 the model tends to "
                    "memorise rather than generalise.",
                severity="high",
                basis=f"sample_count = {samples} (measured), below the small-dataset threshold",
                mitigation="Collect more data, or use prompting/RAG instead of fine-tuning.",
            ))

        duplicates = float(dataset.get("duplicate_percentage") or 0)
        if duplicates > 10:
            risks.append(RiskItem(
                risk="Excessive duplicates",
                why=f"{_pct(duplicates)} of records are exact duplicates, so training capacity "
                    "is spent on repeats.",
                severity="high",
                basis=f"duplicate_percentage = {_pct(duplicates)} (measured)",
                mitigation="Deduplicate the dataset before training.",
            ))

        missing = float(dataset.get("missing_field_percentage") or 0)
        if missing > 10:
            risks.append(RiskItem(
                risk="Incomplete records",
                why=f"{_pct(missing)} of records have empty input or output fields, which makes "
                    "the training signal inconsistent.",
                severity="high",
                basis=f"missing_field_percentage = {_pct(missing)} (measured)",
                mitigation="Fill or drop incomplete records; re-run the dataset analysis.",
            ))

        if hp.get("overfitting_risk") == "high":
            risks.append(RiskItem(
                risk="Overfitting",
                why="The hyperparameter analyzer rated overfitting risk high for the current configuration.",
                severity="high",
                basis="HyperparameterAnalyzer overfitting_risk = high (heuristic on measured config)",
                mitigation="Reduce epochs, hold out a validation split, and enable early stopping.",
            ))

        clarity = (results.get("prompt") or {}).get("clarity_score")
        if clarity is not None and float(clarity) < 0.5:
            risks.append(RiskItem(
                risk="Weak prompt specification",
                why=f"Prompt clarity scored {float(clarity):.0%}, so the model may be penalised "
                    "for ambiguity the prompt itself introduced.",
                severity="medium",
                basis=f"clarity_score = {_num(clarity)} (measured on the template)",
                mitigation="State the task, output format and one worked example explicitly.",
            ))

        if getattr(prediction, "hallucination_risk", None) == "high":
            risks.append(RiskItem(
                risk="Hallucination",
                why="The prediction engine derived a high hallucination risk from the measured "
                    "data and configuration.",
                severity="high",
                basis="PredictionEngine hallucination_risk = high (heuristic on measured inputs)",
                mitigation="Add grounding (RAG), reduce epochs, and enlarge the dataset.",
            ))

        if hardware is not None:
            if not hardware.get("gpu_count"):
                risks.append(RiskItem(
                    risk="No local GPU",
                    why="No CUDA GPU was detected, so self-hosted fine-tuning is not available "
                        "on this machine.",
                    severity="high",
                    basis="GPU detector reported 0 GPUs (measured)",
                    mitigation="Rent a GPU, or use a hosted training/API service.",
                ))

        if use_case and use_case.confidence in ("low", "very_low"):
            risks.append(RiskItem(
                risk="Ambiguous use case",
                why=f"The inferred use case rests on weak evidence (confidence: "
                    f"{use_case.confidence}), so the chosen approach may not match the real goal.",
                severity="medium",
                basis=(
                    f"use-case confidence = {use_case.confidence} "
                    f"(score {_num(use_case.confidence_score)})"
                ),
                mitigation="Confirm the intended task, or add explicit task documentation and "
                "prompts to the project, then re-analyze.",
            ))

        if not context.evaluation_scripts:
            risks.append(RiskItem(
                risk="No evaluation harness",
                why="No evaluation script was found, so there is no measured way to tell whether "
                    "training improved the model.",
                severity="high",
                basis="Scanner found 0 evaluation scripts (measured)",
                mitigation="Create a held-out evaluation set and score the base vs tuned model "
                "before shipping.",
            ))
        return risks

    def _next_experiment(
        self,
        use_case: Optional[UseCaseAnalysisResult],
        results: Dict[str, Any],
        model: Optional[Dict[str, Any]],
        training: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """A concrete, evidence-grounded next step."""
        dataset = results.get("dataset") or {}
        samples = int(dataset.get("sample_count") or 0)
        recommended = (training or {}).get("recommended_starting_point") or {}
        method = recommended.get("method", "LoRA")

        if samples == 0:
            return {
                "goal": "Get a measurable baseline before spending compute.",
                "steps": [
                    "Provide a valid dataset file (JSONL/JSON/CSV/TSV) with input and output fields.",
                    "Re-run Analyze to measure quality, length distribution and duplicates.",
                    "Score the base model on 20-50 hand-picked examples to get a baseline.",
                ],
                "success_criterion": "A recorded baseline score on a fixed evaluation set.",
            }

        task = use_case.primary_task if use_case else "your task"
        base_model = model.get("selected_model") if model else "the base model"
        return {
            "goal": (
                f"Validate that a small {method} run on the measured {samples} records "
                f"improves on the base model for '{task}'."
            ),
            "steps": [
                f"Baseline: score {base_model} on a held-out evaluation split of at least "
                "50 examples, so the comparison is measured rather than assumed.",
                f"Fine-tune with {method}, learning rate {recommended.get('learning_rate')}, "
                f"{recommended.get('epochs')} epochs, batch size "
                f"{recommended.get('batch_size')} x {recommended.get('gradient_accumulation')} "
                "gradient accumulation.",
                "Re-score the tuned model on the same split and compare against the baseline.",
                "Keep the run only if the score improves; otherwise fix the data or the prompt first.",
            ],
            "success_criterion": (
                "A measured improvement on the held-out split, with no regression on "
                "instruction-following or formatting."
            ),
        }

    @staticmethod
    def _recommendation_dict(recommendation: Any) -> Dict[str, Any]:
        """Normalise a Recommendation into a plain dict for the report."""
        if hasattr(recommendation, "model_dump"):
            return recommendation.model_dump()
        return dict(recommendation) if isinstance(recommendation, dict) else {"title": str(recommendation)}

    # ------------------------------------------------------------------
    # Markdown rendering (rendered into the existing chat window)
    # ------------------------------------------------------------------
    def render_markdown(self, analysis: CompleteAnalysis) -> str:
        """Render the complete analysis as the chat-window message."""
        out: List[str] = ["# Complete Project Analysis", ""]
        if analysis.analysis_status == "partial":
            out.append(
                f"> ⚠️ **Partial result.** {len(analysis.partial_failures)} analysis step(s) "
                "failed. Everything below that is present was measured successfully."
            )
            out.append("")

        self._render_project_understanding(out, analysis.project_understanding)
        self._render_use_case(out, analysis.use_case)
        self._render_dataset(out, analysis.dataset_analysis)
        self._render_prompt(out, analysis.prompt_analysis)
        self._render_models(out, analysis)
        self._render_hardware(out, analysis.hardware)
        self._render_training(out, analysis.training_configuration)
        self._render_timing(out, analysis.training_time_estimate)
        self._render_cost(out, analysis.cost_estimate, analysis.budget)
        self._render_approach(out, analysis.approach_recommendation)
        self._render_behavior(out, analysis.expected_behavior)
        self._render_risks(out, analysis.risks)
        self._render_recommendations(out, analysis.recommendations)
        self._render_next_experiment(out, analysis.next_experiment)
        self._render_unavailable(out, analysis.unavailable_sections, analysis.partial_failures)
        self._render_notice(out, analysis.model_information_notice)
        return "\n".join(out).rstrip() + "\n"

    def _render_project_understanding(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 1. Project Understanding")
        if not section:
            out.append("\n_Project discovery unavailable._\n")
            return
        out.extend([
            "",
            f"- **Project:** {section.get('project_name')}",
            f"- **Framework:** {section.get('detected_framework')}",
            f"- **Base model:** {section.get('base_model')}",
            f"- **Files inspected:** {section.get('file_count')}",
        ])
        for label, key in (
            ("Dataset files", "dataset_files"),
            ("Prompt templates", "prompt_templates"),
            ("Config files", "configuration_files"),
            ("Training scripts", "training_scripts"),
            ("Evaluation scripts", "evaluation_scripts"),
            ("Inference scripts", "inference_scripts"),
        ):
            values = section.get(key) or []
            out.append(
                f"- **{label}:** {len(values)}"
                + (f" — {', '.join(str(v) for v in values[:5])}" if values else " — none found")
            )
        out.append(f"\n_{section.get('note')}_\n")

    def _render_use_case(self, out: List[str], use_case: Any) -> None:
        out.append("## 2. Identified Use Case")
        if not use_case:
            out.append("\n_Use-case detection unavailable._\n")
            return
        out.extend(["", f"**USE CASE:** {use_case.primary_task}", "", "**EVIDENCE:**"])
        for signal in (use_case.evidence or [])[:6]:
            out.append(f"- ({signal.provenance}) {signal.evidence}")
        if not use_case.evidence:
            out.append("- No supporting evidence was found.")
        out.extend([
            "",
            f"**CONFIDENCE:** {use_case.confidence} (score {_num(use_case.confidence_score)})",
            "",
            f"**REASONING:** {use_case.reasoning}",
        ])
        if use_case.secondary_tasks:
            out.append(f"\n**Also considered:** {', '.join(use_case.secondary_tasks)}")
        if use_case.uncertainties:
            out.append("\n**Uncertainty:**")
            out.extend(f"- {item}" for item in use_case.uncertainties)
        out.append("")

    def _render_dataset(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 3. Dataset Analysis")
        if not section:
            out.append("\n_Dataset analysis unavailable._\n")
            return
        out.extend([
            "",
            f"- **Records measured:** {section.get('sample_count')}",
            f"- **Quality score:** {_ratio_pct(section.get('quality_score'), 0)}",
            f"- **Estimated tokens:** {section.get('token_count')}",
            f"- **Avg input length:** {_num(section.get('average_prompt_length'), 0)} chars",
            f"- **Avg output length:** {_num(section.get('average_response_length'), 0)} chars",
            f"- **Duplicates:** {_pct(section.get('duplicate_percentage'))}",
            f"- **Near-duplicates:** {_pct(section.get('near_duplicate_percentage'))}",
            f"- **Missing fields:** {_pct(section.get('missing_field_percentage'))}",
            f"- **Formatting consistency:** {_ratio_pct(section.get('formatting_consistency_score'), 0)}",
            f"- **Confidence:** {section.get('confidence')} _(measured)_",
        ])
        for title, key in (
            ("Findings", "findings"), ("Warnings", "warnings"), ("Suggestions", "recommendations")
        ):
            values = section.get(key) or []
            if values:
                out.append(f"\n**{title}:**")
                out.extend(f"- {item}" for item in values[:8])
        out.append("")

    def _render_prompt(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 4. Prompt Quality")
        if not section:
            out.append("\n_Prompt analysis unavailable._\n")
            return
        out.extend([
            "",
            f"- **Template:** {section.get('template_name')}",
            f"- **PROMPT QUALITY:** clarity {_ratio_pct(section.get('clarity_score'), 0)}, "
            f"ambiguity {_ratio_pct(section.get('ambiguity_score'), 0)}, "
            f"instruction quality {_ratio_pct(section.get('instruction_quality_score'), 0)}, "
            f"formatting {_ratio_pct(section.get('formatting_score'), 0)}",
            f"- **Complexity:** {section.get('prompt_complexity')}",
            f"- **Confidence:** {section.get('confidence')} _(measured)_",
        ])
        issues = section.get("detected_issues") or []
        if issues:
            out.append("\n**PROBLEMS FOUND:**")
            out.extend(f"- {item}" for item in issues[:8])
        recs = section.get("recommendations") or []
        if recs:
            out.append("\n**WHY THEY MATTER / SUGGESTED IMPROVEMENTS:**")
            out.extend(f"- {item}" for item in recs[:8])
        else:
            out.append(
                "\n**PROBLEMS FOUND:** none significant. No rewrite is recommended — changes "
                "are only suggested when there is a concrete reason."
            )
        out.append("")

    def _render_models(self, out: List[str], analysis: CompleteAnalysis) -> None:
        out.append("## 5. Model Recommendations")
        if not analysis.model_analysis:
            out.append("\n_Model analysis unavailable._\n")
        else:
            section = analysis.model_analysis
            out.extend([
                "",
                f"- **Configured model:** {section.get('selected_model')}",
                f"- **Parameters:** {section.get('parameter_count')}",
                f"- **Context length:** {section.get('context_length')}",
                f"- **VRAM (reference data):** {section.get('estimated_vram')}",
                f"- **Capabilities:** reasoning {section.get('reasoning_capability')}, "
                f"coding {section.get('coding_capability')}, "
                f"instruction following {section.get('instruction_following_capability')}",
            ])
        if not analysis.model_candidates:
            out.append("\n_No model candidates could be resolved from the reference database._")
        for candidate in analysis.model_candidates[:4]:
            out.extend([
                "",
                f"**MODEL:** {candidate.model}",
                f"- **WHY IT FITS:** {candidate.why_it_fits}",
                f"- **EXPECTED STRENGTHS:** {'; '.join(candidate.expected_strengths) or 'not characterised'}",
                f"- **EXPECTED LIMITATIONS:** {'; '.join(candidate.expected_limitations) or 'not characterised'}",
                f"- **RESOURCE REQUIREMENTS:** {candidate.resource_requirements}",
                f"- **COST CONSIDERATIONS:** {candidate.cost_considerations}",
                f"- **BEST USE CASE:** {candidate.best_use_case}",
            ])
        out.append("\n_No model is universally best — the right choice depends on your task, "
                   "hardware and budget._\n")

        out.append("## 6. Model Comparison")
        if not analysis.model_comparison:
            out.append("\n_Model comparison unavailable._\n")
            return
        out.append("")
        for key, value in (analysis.model_comparison.get("expected_tendencies") or {}).items():
            out.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        out.append(f"\n_{analysis.model_comparison.get('disclaimer')}_\n")

    def _render_hardware(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 7. Hardware / GPU Requirements")
        if not section:
            out.append(
                "\n_Hardware analysis unavailable — no GPU could be detected on this machine, "
                "so VRAM feasibility cannot be confirmed._\n"
            )
            return
        out.append("\n**ACTUAL HARDWARE:**")
        if section.get("gpu_count"):
            for gpu in section.get("gpus", []):
                out.append(
                    f"- GPU {gpu.get('index')}: {gpu.get('name')}, "
                    f"{gpu.get('vram_gb') or 'unknown'} GB VRAM"
                )
            out.append(f"- CUDA available: {section.get('cuda_available')}")
        else:
            out.append("- No CUDA GPU detected on this machine.")
        out.append(f"- Detection method: {section.get('detection_method')}")
        out.append("\n**FEASIBILITY:** see the training-time section for the VRAM estimate "
                   "against detected capacity.")
        out.append("\n_ASSUMPTIONS: the model size comes from reference data matched on the "
                   "model name, not from a measured file size._\n")

    def _render_training(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 8. Training Configuration")
        if not section:
            out.append("\n_Training configuration unavailable._\n")
            return
        current = section.get("current_configuration") or {}
        out.append("\n**CURRENT CONFIGURATION (from your files):**")
        for key, value in current.items():
            out.append(f"- {key}: {value if value is not None else 'not set'}")
        out.append(f"\n- Overfitting risk: {section.get('overfitting_risk')}")
        out.append(f"- Underfitting risk: {section.get('underfitting_risk')}")
        out.append(f"- Config efficiency: {_ratio_pct(section.get('efficiency_score'), 0)}")
        out.append(f"- Confidence: {section.get('confidence')}")

        recommended = section.get("recommended_starting_point") or {}
        if recommended:
            out.append("\n**RECOMMENDED STARTING POINT (heuristics, each explained):**")
            out.append(f"- Method: {recommended.get('method')}")
            for key, reason_key in (
                ("learning_rate", "learning_rate_reason"),
                ("epochs", "epochs_reason"),
                ("batch_size", "batch_size_reason"),
                ("gradient_accumulation", "gradient_accumulation_reason"),
                ("sequence_length", "sequence_length_reason"),
                ("precision", "precision_reason"),
                ("optimizer", "optimizer_reason"),
                ("gradient_checkpointing", "gradient_checkpointing_reason"),
                ("quantization", "quantization_reason"),
            ):
                out.append(f"- {key}: {recommended.get(key)} — {recommended.get(reason_key)}")
        out.append("")

    def _render_timing(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 9. Estimated Training Time")
        if not section:
            out.append(
                "\n_Training time unavailable — a GPU must be detected to project wall-clock "
                "time. Missing inputs: detected GPU and its throughput._\n"
            )
            return
        out.extend([
            "",
            f"**ESTIMATED TRAINING TIME:** {section.get('estimated_time')}",
            f"- **EXPECTED RANGE:** {section.get('expected_range')}",
            f"- Total steps: {section.get('total_steps')}",
            f"- Confidence: {section.get('confidence')} ({section.get('mode')} estimate)",
            f"- VRAM feasible: {section.get('vram_feasible')} "
            f"(estimated {section.get('vram_estimated_gb')} GB vs "
            f"{section.get('vram_available_gb')} GB available)",
        ])
        if section.get("warnings"):
            out.append("\n**Warnings:**")
            out.extend(f"- {item}" for item in section["warnings"][:5])
        out.append(f"\n**ASSUMPTIONS:** {'; '.join(section.get('assumptions', [])[:5]) or 'none recorded'}")
        out.append(f"\n**UNCERTAINTY:** {section.get('uncertainty')}\n")

    def _render_cost(self, out: List[str], cost: Optional[Dict[str, Any]],
                     budget: Optional[Dict[str, Any]]) -> None:
        out.append("## 10. Cost Estimate")
        if cost:
            out.extend([
                "",
                f"- **Training time:** {cost.get('estimated_training_time')}",
                f"- **GPU-hours:** {cost.get('estimated_gpu_hours')}",
                f"- **VRAM usage:** {cost.get('estimated_vram_usage')}",
                f"- **Checkpoint size:** {cost.get('estimated_checkpoint_size')}",
                f"- **Storage:** {cost.get('estimated_storage_requirement')}",
                f"- **Confidence:** {cost.get('confidence')}",
                f"\n_{cost.get('note')}_",
            ])
        else:
            out.append("\n_Cost estimation unavailable._")
        if not budget:
            return
        out.append(f"\n**Budget constraint:** {budget.get('budget_constraint')}")
        out.append("")
        for option in budget.get("options", []):
            out.append(f"- **{option.get('tier')}:** {option.get('approach')} — {option.get('rationale')}")
        out.append("\n**Cost drivers:**")
        out.extend(f"- {driver}" for driver in budget.get("cost_drivers", []))
        out.append(f"\n_{budget.get('pricing_note')}_")
        out.append(f"\n_{budget.get('local_inference_note')}_\n")

    def _render_approach(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 11. Fine-tuning vs RAG vs Prompting")
        if not section:
            out.append("\n_Approach analysis unavailable._\n")
            return
        out.extend([
            "",
            f"**RECOMMENDED APPROACH:** {section.get('recommended_approach')}",
            f"- **Why:** {section.get('rationale')}",
            f"- **Use-case confidence:** {section.get('use_case_confidence')}",
        ])
        for reason in section.get("supporting_reasons", []):
            out.append(f"- {reason}")
        out.append("\n**Decision rules:**")
        out.extend(f"- {rule}" for rule in section.get("decision_rules", []))
        out.append("")

    def _render_behavior(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 12. Expected Model Behaviour & Trade-offs")
        if not section:
            out.append("\n_Behaviour prediction unavailable._\n")
            return
        out.extend([
            "",
            f"- **Instruction following:** {section.get('instruction_following')}",
            f"- **Hallucination risk:** {section.get('hallucination_risk')}",
            f"- **Reasoning:** {section.get('reasoning')}",
            f"- **Formatting:** {section.get('formatting')}",
            f"- **Status / confidence:** {section.get('status')} / {section.get('confidence')}",
        ])
        for title, key in (
            ("Expected strengths", "expected_strengths"),
            ("Expected weaknesses", "expected_weaknesses"),
            ("Likely failure modes", "likely_failure_modes"),
        ):
            values = section.get(key) or []
            if values:
                out.append(f"\n**{title}:**")
                out.extend(f"- {item}" for item in values[:6])
        if section.get("unknowns"):
            out.append("\n**Could not be predicted (missing information):**")
            out.extend(f"- {item}" for item in section["unknowns"][:6])
        if section.get("quality_note"):
            out.append(f"\n_{section.get('quality_note')}_")
        out.append("")

    def _render_risks(self, out: List[str], risks: List[RiskItem]) -> None:
        out.append("## 13. Risks")
        if not risks:
            out.append("\n_No significant risks were identified from the measured data._\n")
            return
        for risk in risks:
            out.extend([
                "",
                f"**RISK:** {risk.risk} — *severity: {risk.severity}*",
                f"- **WHY:** {risk.why}",
                f"- **BASIS:** {risk.basis}",
                f"- **MITIGATION:** {risk.mitigation}",
            ])
        out.append("")

    def _render_recommendations(self, out: List[str], recommendations: List[Dict[str, Any]]) -> None:
        out.append("## 14. Recommendations")
        if not recommendations:
            out.append("\n_No recommendations were generated — the measured inputs did not "
                       "trigger any._\n")
            return
        for rec in recommendations[:12]:
            out.extend([
                "",
                f"- **[{rec.get('severity', 'medium')}] {rec.get('title')}** "
                f"({rec.get('category', 'general')})",
                f"  - {rec.get('description')}",
            ])
            if rec.get("reasoning"):
                out.append(f"  - Reasoning: {rec.get('reasoning')}")
            if rec.get("evidence"):
                out.append(f"  - Evidence: {rec.get('evidence')}")
        out.append("")

    def _render_next_experiment(self, out: List[str], section: Optional[Dict[str, Any]]) -> None:
        out.append("## 15. Suggested Next Experiment")
        if not section:
            out.append("\n_No experiment could be proposed._\n")
            return
        out.extend(["", f"**GOAL:** {section.get('goal')}", "", "**Steps:**"])
        out.extend(f"{index}. {step}" for index, step in enumerate(section.get("steps", []), 1))
        out.append(f"\n**Success criterion:** {section.get('success_criterion')}\n")

    def _render_unavailable(
        self,
        out: List[str],
        unavailable: List[Dict[str, str]],
        failures: List[Dict[str, Any]],
    ) -> None:
        if not unavailable and not failures:
            return
        out.append("## Partial Failures")
        out.append(
            "\nThe following could not be produced. They are listed rather than replaced "
            "with placeholder values."
        )
        for item in unavailable:
            out.append(f"- **{item.get('section')}** unavailable because: {item.get('reason')}")
        for failure in failures:
            name = failure.get("name") or failure.get("step") or "unknown step"
            out.append(f"- **{name}** failed: {failure.get('error', 'unknown error')}")
        out.append("")

    def _render_notice(self, out: List[str], notice: str) -> None:
        if notice:
            out.append(f"---\n\n_{notice}_")


def _parse_params(parameter_count: Any) -> Optional[float]:
    """Parse a parameter count such as ``"8B"`` or ``"7b"`` into a float."""
    if parameter_count is None:
        return None
    text = str(parameter_count).strip().lower().replace("b", "").replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return None
