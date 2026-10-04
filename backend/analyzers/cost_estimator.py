"""Training cost estimator — uses deterministic formulas based on model + dataset size."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.schemas import ProjectContext, CostEstimate
from cleaning import dataset_io

logger = logging.getLogger(__name__)

# GPU specifications: name -> (TFLOPS FP16, VRAM GB)
_GPU_DATABASE: Dict[str, Dict[str, Any]] = {
    "rtx_4090": {"tflops": 163, "vram": 24, "name": "NVIDIA RTX 4090"},
    "rtx_4080": {"tflops": 108, "vram": 16, "name": "NVIDIA RTX 4080"},
    "rtx_4070": {"tflops": 60, "vram": 12, "name": "NVIDIA RTX 4070"},
    "rtx_3090": {"tflops": 70, "vram": 24, "name": "NVIDIA RTX 3090"},
    "rtx_3080": {"tflops": 60, "vram": 12, "name": "NVIDIA RTX 3080"},
    "rtx_3060": {"tflops": 40, "vram": 12, "name": "NVIDIA RTX 3060"},
    "a100_40gb": {"tflops": 193, "vram": 40, "name": "NVIDIA A100 (40GB)"},
    "a100_80gb": {"tflops": 193, "vram": 80, "name": "NVIDIA A100 (80GB)"},
    "a10g": {"tflops": 125, "vram": 24, "name": "NVIDIA A10G"},
    "h100": {"tflops": 200, "vram": 80, "name": "NVIDIA H100"},
}


class CostEstimator:
    """Estimates computational cost of fine-tuning using deterministic formulas."""

    def __init__(self):
        # Note: cost estimation is deterministic — no LLM is involved, so the
        # result can never be "enhanced" with unverifiable numbers.
        pass

    async def analyze(
        self,
        context: ProjectContext,
        dataset_result: Optional[Any] = None,
        hp_result: Optional[Any] = None,
    ) -> CostEstimate:
        """Estimate training resources from model size, dataset size, and hyperparameters.

        All inputs are taken from real sources when available:
          * ``dataset_result``  - sample/token counts measured by the DatasetAnalyzer
          * ``hp_result``       - batch size / epochs / seq length from the user's config
          * ``context.hardware_information`` - actually detected GPU(s)
        Values that cannot be determined are reported as ``unknown`` with an
        explicit warning rather than replaced by placeholder constants.
        """
        logger.info("Estimating training cost")

        assumptions: List[str] = []

        # --- Model size (reference lookup from the model name) -------------
        param_count_billions = self._extract_param_count(context)
        if param_count_billions is None:
            assumptions.append(
                "Model parameter count is unknown (model name not recognized or missing) — "
                "time/VRAM/storage estimates are omitted."
            )
        else:
            assumptions.append(
                f"Model: ~{param_count_billions}B parameters (derived from the model name/reference data, not measured)"
            )

        # --- Dataset size (prefer real measured values) ---------------------
        dataset_samples: Optional[int] = None
        avg_tokens_per_sample: Optional[float] = None
        if dataset_result is not None:
            if getattr(dataset_result, "sample_count", 0):
                dataset_samples = int(dataset_result.sample_count)
            prompt_len = getattr(dataset_result, "average_prompt_length", 0.0) or 0.0
            resp_len = getattr(dataset_result, "average_response_length", 0.0) or 0.0
            measured_chars = prompt_len + resp_len
            if measured_chars > 0:
                # ~4 characters per token (statistical average for English text)
                avg_tokens_per_sample = measured_chars / 4.0
                assumptions.append(
                    f"Average sample length: ~{avg_tokens_per_sample:.0f} tokens "
                    f"(measured {measured_chars:.0f} chars/sample / 4 chars-per-token)"
                )
        if dataset_samples is None:
            for key, val in (context.project_statistics or {}).items():
                if "sample" in key.lower() and isinstance(val, (int, float)) and val:
                    dataset_samples = int(val)
                    assumptions.append(f"Dataset samples: {dataset_samples} (from project scan statistics)")
                    break
        if dataset_samples is None:
            assumptions.append(
                "Dataset size is unknown (no dataset found or none could be read) — "
                "training-time and storage estimates are omitted."
            )
        if avg_tokens_per_sample is None:
            assumptions.append("Average sample length unknown — assuming the full configured sequence length per sample")

        # --- Training parameters (user config first; defaults flagged) ------
        seq_length: Optional[int] = getattr(hp_result, "sequence_length", None)
        batch_size: Optional[int] = getattr(hp_result, "batch_size", None)
        epochs: Optional[int] = getattr(hp_result, "epochs", None)
        grad_accum: int = getattr(hp_result, "gradient_accumulation", None) or 1

        if seq_length is None:
            seq_length, seq_assumed = self._extract_seq_length_from_configs(context), True
        else:
            seq_assumed = False
        if epochs is None:
            epochs, epochs_assumed = self._extract_epochs_from_configs(context), True
        else:
            epochs_assumed = False
        if batch_size is None:
            batch_size, batch_assumed = self._extract_batch_size_from_configs(context), True
        else:
            batch_assumed = False

        if seq_assumed:
            assumptions.append(f"Sequence length: {seq_length} tokens (assumed default — not found in the project configuration)")
        if epochs_assumed:
            assumptions.append(f"Epochs: {epochs} (assumed default — not found in the project configuration)")
        if batch_assumed:
            assumptions.append(f"Batch size: {batch_size}, gradient accumulation: {grad_accum} (assumed defaults — not found in the project configuration)")

        # --- Hardware reference (prefer actually detected GPU) --------------
        reference_gpu = self._resolve_reference_gpu(context)
        if reference_gpu["source"] == "detected":
            assumptions.append(
                f"Reference GPU: {reference_gpu['name']} ({reference_gpu['tflops']} TFLOPS FP16, detected on this machine)"
            )
        else:
            assumptions.append(
                f"Reference GPU: {reference_gpu['name']} ({reference_gpu['tflops']} TFLOPS FP16) — "
                "assumed; no supported GPU was detected"
            )
        assumptions.append("Training method: QLoRA (assumed — quantized 8-bit base + LoRA adapters; the project does not specify the method)")
        assumptions.append("GPU utilization: ~70% (heuristic allowance for data loading and memory overhead)")

        # --- Compute estimates (only from real/flagged inputs) --------------
        estimates = self._compute_estimates(
            param_count_billions, dataset_samples, avg_tokens_per_sample,
            seq_length, batch_size, epochs, grad_accum, reference_gpu,
        )

        # --- Compatible hardware --------------------------------------------
        compatible = (
            self._find_compatible_hardware(param_count_billions)
            if param_count_billions is not None
            else ["unknown — model parameter count could not be determined"]
        )

        confidence = "high"
        if param_count_billions is None or dataset_samples is None:
            confidence = "very_low"
        elif any("assumed" in a for a in assumptions):
            confidence = "medium"

        return CostEstimate(
            estimated_training_time=estimates["training_time"],
            estimated_gpu_hours=estimates["gpu_hours"],
            estimated_vram_usage=estimates["vram"],
            estimated_checkpoint_size=estimates["checkpoint_size"],
            estimated_storage_requirement=estimates["storage"],
            compatible_hardware=compatible,
            assumptions=assumptions,
            confidence=confidence,
        )

    def _extract_param_count(self, context: ProjectContext) -> Optional[float]:
        """Extract parameter count (billions) from the model name or context.

        Returns None when the model is unknown — no placeholder value is used.
        """
        model_name = context.base_model or ""
        if not model_name or model_name == "unknown":
            return None

        name_lower = model_name.lower()

        # Reference data: model-name suffixes mapped to approximate parameter
        # counts. These are derived from the model name, not measured.
        param_map = [
            ("70b", 70.0), ("72b", 72.0), ("65b", 65.0), ("34b", 34.0), ("33b", 33.0),
            ("30b", 30.0), ("13b", 13.0), ("8b", 8.0), ("7b", 7.0), ("6b", 6.0),
            ("3b", 3.0), ("2.7b", 2.7), ("2b", 2.0), ("1.8b", 1.8),
            ("1.5b", 1.5), ("1.1b", 1.1), ("1b", 1.0),
        ]
        for pattern, params in param_map:
            if pattern in name_lower:
                return params

        # Check project statistics
        if context.project_statistics.get("param_count_billions"):
            return float(context.project_statistics["param_count_billions"])

        return None

    def _resolve_reference_gpu(self, context: ProjectContext) -> Dict[str, Any]:
        """Resolve the reference GPU: detected hardware first, otherwise a flagged assumption."""
        hw = context.hardware_information or {}
        gpus = hw.get("gpus") or []
        if gpus:
            name = str(gpus[0].get("name", ""))
            for key, info in _GPU_DATABASE.items():
                # Match database entry by the most distinctive part of its name
                distinctive = info["name"].replace("NVIDIA ", "").split(" ")[0].lower()
                if distinctive and distinctive in name.lower():
                    return {"name": info["name"], "tflops": info["tflops"], "vram": info["vram"], "source": "detected"}
            # Detected GPU that is not in the reference table — report it as
            # detected but fall back to a generic mid-range reference.
            return {"name": f"{name} (detected)", "tflops": 100.0, "vram": None, "source": "detected-unknown-tflops"}
        return {"name": "NVIDIA RTX 4090", "tflops": 163.0, "vram": 24, "source": "assumed"}

    def _extract_from_configs(self, context: ProjectContext, patterns: Dict[str, str]) -> Dict[str, Any]:
        """Extract config values from the project's configuration files via regex."""
        import re as _re
        found: Dict[str, Any] = {}
        for cf in context.configuration_files:
            try:
                config_path = (
                    Path(f"{context.project_path}/{cf}")
                    if not Path(cf).is_absolute()
                    else Path(cf)
                )
                if config_path.exists():
                    text = dataset_io.read_text(config_path)
                    for key, pattern in patterns.items():
                        if key in found:
                            continue
                        m = _re.search(pattern, text)
                        if m:
                            try:
                                found[key] = int(float(m.group(1)))
                            except ValueError:
                                pass
            except Exception as e:
                logger.debug(f"Failed to read config {cf}: {e}")
        return found

    def _extract_seq_length_from_configs(self, context: ProjectContext) -> int:
        found = self._extract_from_configs(context, {"sequence_length": r"max_seq_length\s*[:=]\s*(\d+)"})
        return found.get("sequence_length", 512)

    def _extract_epochs_from_configs(self, context: ProjectContext) -> int:
        found = self._extract_from_configs(context, {"epochs": r"num_train_epochs\s*[:=]\s*([\d.]+)"})
        return found.get("epochs", 3)

    def _extract_batch_size_from_configs(self, context: ProjectContext) -> int:
        found = self._extract_from_configs(
            context,
            {
                "batch_size": r"per_device_train_batch_size\s*[:=]\s*(\d+)",
                "gradient_accumulation": r"gradient_accumulation_steps\s*[:=]\s*(\d+)",
            },
        )
        return found.get("batch_size", 8)

    def _compute_estimates(
        self,
        param_billions: Optional[float],
        dataset_samples: Optional[int],
        avg_tokens_per_sample: Optional[float],
        seq_length: int,
        batch_size: int,
        epochs: int,
        grad_accum: int,
        reference_gpu: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute cost estimates using standard FLOPs-based formulas.

        Every input is either measured, from reference data, user-provided, or
        an explicitly flagged assumption (see ``assumptions``). When a required
        input is missing, the corresponding output is ``unknown`` (with
        ``0.0`` GPU-hours) instead of a fabricated number.
        """
        vram: Optional[str] = None
        checkpoint_size: Optional[str] = None
        storage: Optional[str] = None
        training_time: Optional[str] = None
        gpu_hours: Optional[float] = None

        # --- VRAM / checkpoint / storage (needs parameter count only) -------
        if param_billions is not None:
            # QLoRA assumption: FP16 params*2 bytes base is NOT fully loaded
            # (4-bit base ≈ 0.5 bytes/param) + LoRA adapters + activation overhead.
            vram_gb = param_billions * 1.0 + 4  # ~1GB per billion params (4-bit) + 4GB overhead
            vram_gb = max(vram_gb, 4.0)  # minimum practical footprint
            # Full fine-tune checkpoint: FP16 weights + Adam optimizer states ≈ 2 + 8 bytes/param
            checkpoint_gb_full = param_billions * 10.0
            # Storage: checkpoint + dataset + intermediate files (3x safety margin)
            storage_gb = checkpoint_gb_full * 3 + 10
            if dataset_samples is not None:
                dataset_bytes = (
                    dataset_samples * avg_tokens_per_sample * 4 / 1e9
                    if avg_tokens_per_sample is not None
                    else dataset_samples * seq_length * 4 / 1e9
                )
                storage_gb += dataset_bytes
            vram = f"~{vram_gb:.0f}GB (QLoRA assumption)"
            checkpoint_size = f"~{checkpoint_gb_full:.0f}GB (full fine-tune, FP16 + Adam states)"
            storage = f"~{storage_gb:.0f}GB"

        # --- Training time (needs model + dataset size) ----------------------
        if param_billions is not None and dataset_samples is not None:
            tokens_per_sample = avg_tokens_per_sample if avg_tokens_per_sample is not None else float(seq_length)
            total_tokens = dataset_samples * tokens_per_sample * epochs

            # Standard training FLOPs approximation: 6 * P * tokens
            flops_per_token = 6 * param_billions * 1e9
            total_flops = flops_per_token * total_tokens

            reference_tflops = reference_gpu["tflops"]
            theoretical_hours = total_flops / (reference_tflops * 1e12) / 3600.0

            # Heuristic utilization allowance (~70%) — not a measured value
            utilization = 0.70
            wall_clock_hours = theoretical_hours / utilization

            def _format_hours(h: float) -> str:
                seconds = h * 3600
                if seconds < 60:
                    return f"{seconds:.0f} seconds"
                if h < 1:
                    return f"{h * 60:.0f} minutes"
                if h < 24:
                    return f"{h:.1f} hours"
                return f"{h / 24:.1f} days"

            # ±50% heuristic range (theoretical peak is never achieved)
            training_time = f"{_format_hours(wall_clock_hours * 0.5)}-{_format_hours(wall_clock_hours * 1.5)}"
            # Keep enough precision that small jobs don't round to a fake "0.0"
            gpu_hours = round(wall_clock_hours, 4) or 0.0
        else:
            training_time = "unknown — model size or dataset size could not be determined"
            gpu_hours = 0.0

        return {
            "training_time": training_time,
            "gpu_hours": gpu_hours,
            "vram": vram or "unknown — model parameter count could not be determined",
            "checkpoint_size": checkpoint_size or "unknown — model parameter count could not be determined",
            "storage": storage or "unknown — model parameter count could not be determined",
        }

    def _find_compatible_hardware(self, param_billions: float) -> List[str]:
        """Find reference GPUs whose VRAM fits the (QLoRA) requirement estimate."""
        vram_needed = param_billions * 1.0 + 4  # QLoRA estimate (4-bit base + adapters)
        compatible = []
        for gpu_key, gpu_info in _GPU_DATABASE.items():
            if gpu_info["vram"] >= vram_needed:
                compatible.append(gpu_info["name"])
        return compatible if compatible else ["Any modern GPU with sufficient VRAM"]
