"""GPU training time estimation engine — theoretical + calibrated estimates."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .gpu_detector import (
    HardwareInfo,
    detect_gpus,
    get_gpu_spec,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration extracted from project."""
    model_name: Optional[str] = None
    param_count_billions: Optional[float] = None
    dataset_samples: Optional[int] = None
    sequence_length: int = 512
    batch_size: int = 8
    gradient_accumulation: int = 1
    epochs: int = 3
    max_steps: Optional[int] = None
    precision: str = "fp16"  # fp32, fp16, bf16, 8bit, 4bit
    training_method: str = "full"  # full, lora, qlora, peft
    gradient_checkpointing: bool = False
    dataloader_workers: int = 0
    framework: str = "huggingface"  # huggingface, pytorch, accelerate, deepspeed, fsdp
    distributed_strategy: str = "single"  # single, dp, ddp, fsdp, deepspeed, accelerate

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    @property
    def total_steps(self) -> int:
        """Total training steps."""
        if self.max_steps:
            return self.max_steps
        if not self.dataset_samples:
            return 0
        steps_per_epoch = math.ceil(self.dataset_samples / self.effective_batch_size)
        return steps_per_epoch * self.epochs

    @property
    def total_tokens(self) -> int:
        """Total tokens processed during training."""
        if not self.dataset_samples:
            return 0
        return self.dataset_samples * self.sequence_length * self.epochs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "param_count_billions": self.param_count_billions,
            "dataset_samples": self.dataset_samples,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "precision": self.precision,
            "training_method": self.training_method,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_workers": self.dataloader_workers,
            "framework": self.framework,
            "distributed_strategy": self.distributed_strategy,
        }


@dataclass
class EstimateResult:
    """Result of a GPU training time estimate."""
    mode: str  # "quick" or "calibrated"
    estimated_seconds: float
    lower_bound_seconds: float
    upper_bound_seconds: float
    confidence: str  # low, medium, high
    throughput_steps_per_sec: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    total_steps: int = 0
    vram_feasible: bool = True
    vram_warning: Optional[str] = None
    vram_estimated_gb: Optional[float] = None
    vram_available_gb: Optional[float] = None
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    calibration_used: bool = False

    @property
    def estimated_time_str(self) -> str:
        return _format_duration(self.estimated_seconds)

    @property
    def range_str(self) -> str:
        return f"{_format_duration(self.lower_bound_seconds)} – {_format_duration(self.upper_bound_seconds)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "estimated_seconds": self.estimated_seconds,
            "lower_bound_seconds": self.lower_bound_seconds,
            "upper_bound_seconds": self.upper_bound_seconds,
            "estimated_time": self.estimated_time_str,
            "range": self.range_str,
            "confidence": self.confidence,
            "throughput_steps_per_sec": self.throughput_steps_per_sec,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "total_steps": self.total_steps,
            "vram_feasible": self.vram_feasible,
            "vram_warning": self.vram_warning,
            "vram_estimated_gb": self.vram_estimated_gb,
            "vram_available_gb": self.vram_available_gb,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "gpu_info": self.gpu_info,
            "calibration_used": self.calibration_used,
        }


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.0f}m"
    if seconds < 86400:
        hours = seconds / 3600
        if hours < 10:
            return f"{hours:.1f}h"
        return f"{hours:.0f}h"
    days = seconds / 86400
    return f"{days:.1f}d"


class GPUTimeEstimator:
    """Estimate GPU training time using theoretical formulas and optional calibration."""

    # Multi-GPU scaling efficiency (fraction of linear speedup)
    _SCALING_EFFICIENCY = {
        "single": 1.0,
        "dp": 0.90,      # Data Parallel
        "ddp": 0.85,     # DistributedDataParallel
        "fsdp": 0.80,    # Fully Sharded Data Parallel
        "deepspeed": 0.80,
        "accelerate": 0.85,
    }

    # Precision multipliers (relative compute cost)
    _PRECISION_FACTOR = {
        "fp32": 1.0,
        "fp16": 0.5,
        "bf16": 0.5,
        "8bit": 0.4,
        "4bit": 0.3,
    }

    # Training method compute multipliers
    _METHOD_FACTOR = {
        "full": 1.0,
        "lora": 0.7,
        "qlora": 0.6,
        "peft": 0.7,
    }

    def __init__(self, hardware: Optional[HardwareInfo] = None):
        self.hardware = hardware or detect_gpus()

    def estimate(
        self,
        config: TrainingConfig,
        mode: str = "quick",
        measured_steps_per_sec: Optional[float] = None,
    ) -> EstimateResult:
        """Estimate training time.

        mode: "quick" (theoretical) or "calibrated" (uses measured throughput).
        measured_steps_per_sec: optional measured throughput from calibration.
        """
        if mode == "calibrated" and measured_steps_per_sec:
            return self._estimate_calibrated(config, measured_steps_per_sec)
        return self._estimate_quick(config)

    def _estimate_quick(self, config: TrainingConfig) -> EstimateResult:
        """Theoretical estimate based on GPU specs and training config."""
        warnings: List[str] = []
        assumptions: List[str] = []

        # No GPU detected
        if not self.hardware.gpu_count:
            warnings.append("No supported GPU detected. GPU time estimation may be unavailable or less accurate.")
            return EstimateResult(
                mode="quick",
                estimated_seconds=0,
                lower_bound_seconds=0,
                upper_bound_seconds=0,
                confidence="low",
                total_steps=config.total_steps,
                vram_feasible=False,
                vram_warning="No GPU detected — cannot estimate VRAM feasibility.",
                assumptions=["No GPU detected — estimate unavailable."],
                warnings=warnings,
                gpu_info=self.hardware.to_dict(),
            )

        # Determine primary GPU (first one)
        primary_gpu = self.hardware.gpus[0]
        gpu_spec = get_gpu_spec(primary_gpu.name)

        # Heterogeneous GPU warning
        if self.hardware.heterogeneous:
            warnings.append(
                "⚠ Heterogeneous GPUs detected. Multi-GPU performance may be limited by the slower device."
            )

        # Determine effective TFLOPS
        if gpu_spec:
            tflops = float(gpu_spec["tflops_fp16"])
            assumptions.append(f"GPU: {primary_gpu.name} ({tflops:.0f} TFLOPS FP16)")
        else:
            # Unknown GPU — use generic estimate
            tflops = 25.0  # Conservative generic estimate
            warnings.append(
                f"Unknown GPU '{primary_gpu.name}'. Using generic performance estimate with lower confidence."
            )
            assumptions.append(f"GPU: {primary_gpu.name} (unknown specs — generic estimate)")

        # Apply precision factor
        precision_factor = self._PRECISION_FACTOR.get(config.precision, 0.5)
        assumptions.append(f"Precision: {config.precision} (compute factor {precision_factor})")

        # Apply training method factor
        method_factor = self._METHOD_FACTOR.get(config.training_method, 0.7)
        assumptions.append(f"Training method: {config.training_method} (compute factor {method_factor})")

        # Effective TFLOPS
        effective_tflops = tflops * precision_factor * method_factor

        # Multi-GPU scaling
        gpu_count = self.hardware.gpu_count
        scaling = self._SCALING_EFFICIENCY.get(config.distributed_strategy, 0.85)
        if gpu_count > 1:
            # Not perfectly linear — account for communication overhead
            scaling_factor = gpu_count * scaling
            assumptions.append(
                f"Multi-GPU: {gpu_count} GPUs, {config.distributed_strategy} strategy "
                f"(scaling efficiency {scaling:.0%})"
            )
            if self.hardware.heterogeneous:
                # Heterogeneous — limited by slowest GPU
                scaling_factor = max(1.0, scaling_factor * 0.7)
                warnings.append("Heterogeneous GPUs — effective scaling reduced.")
        else:
            scaling_factor = 1.0
            assumptions.append("Single GPU")

        # Total FLOPs
        if not config.param_count_billions:
            warnings.append("Model parameter count unknown — using default estimate.")
            param_count = 7.0  # Default
            assumptions.append("Model size: unknown — assuming 7B parameters")
        else:
            param_count = config.param_count_billions
            assumptions.append(f"Model size: {param_count}B parameters")

        # FLOPs = 6 * P * tokens (for training)
        total_flops = 6 * param_count * 1e9 * config.total_tokens

        # Compute time in seconds
        compute_seconds = total_flops / (effective_tflops * 1e12 * scaling_factor)

        # GPU utilization factor (not 100% efficient)
        utilization = 0.85  # ~85% GPU utilization
        if config.gradient_checkpointing:
            utilization *= 0.90  # Slight overhead from recomputation
            assumptions.append("Gradient checkpointing enabled (slight overhead)")

        # Data loading overhead
        dataloading_factor = 1.0
        if config.dataloader_workers == 0:
            dataloading_factor = 1.10  # 10% overhead with no workers
            assumptions.append("Dataloader workers: 0 (10% data-loading overhead)")

        estimated_seconds = compute_seconds / utilization * dataloading_factor

        # Confidence
        confidence = "medium"
        if not gpu_spec:
            confidence = "low"
        if not config.param_count_billions or not config.dataset_samples:
            confidence = "low"

        # Bounds: ±20% for medium confidence, ±35% for low
        if confidence == "medium":
            lower = estimated_seconds * 0.8
            upper = estimated_seconds * 1.2
        else:
            lower = estimated_seconds * 0.65
            upper = estimated_seconds * 1.35

        # VRAM feasibility
        vram_estimated_gb = self._estimate_vram_gb(config, param_count)
        vram_available_gb = self.hardware.max_vram_mb / 1024.0 if self.hardware.max_vram_mb else None
        vram_feasible = True
        vram_warning = None

        if vram_available_gb and vram_estimated_gb > vram_available_gb:
            vram_feasible = False
            vram_warning = (
                f"⚠ Estimated configuration may exceed available VRAM "
                f"({vram_estimated_gb:.1f}GB needed vs {vram_available_gb:.1f}GB available). "
                f"Consider: reduce batch size, increase gradient accumulation, "
                f"use LoRA/QLoRA, reduce sequence length, enable gradient checkpointing, "
                f"or use lower precision."
            )
            warnings.append(vram_warning)

        # Throughput estimates
        total_steps = config.total_steps
        steps_per_sec = total_steps / estimated_seconds if total_steps > 0 and estimated_seconds > 0 else None
        samples_per_sec = steps_per_sec * config.batch_size if steps_per_sec else None
        tokens_per_sec = samples_per_sec * config.sequence_length if samples_per_sec else None

        return EstimateResult(
            mode="quick",
            estimated_seconds=estimated_seconds,
            lower_bound_seconds=lower,
            upper_bound_seconds=upper,
            confidence=confidence,
            throughput_steps_per_sec=steps_per_sec,
            throughput_samples_per_sec=samples_per_sec,
            throughput_tokens_per_sec=tokens_per_sec,
            total_steps=total_steps,
            vram_feasible=vram_feasible,
            vram_warning=vram_warning,
            vram_estimated_gb=vram_estimated_gb,
            vram_available_gb=vram_available_gb,
            assumptions=assumptions,
            warnings=warnings,
            gpu_info=self.hardware.to_dict(),
            calibration_used=False,
        )

    def _estimate_calibrated(
        self, config: TrainingConfig, measured_steps_per_sec: float
    ) -> EstimateResult:
        """Estimate using measured throughput from calibration."""
        warnings: List[str] = []
        assumptions: List[str] = []

        if not self.hardware.gpu_count:
            warnings.append("No supported GPU detected.")
            return EstimateResult(
                mode="calibrated",
                estimated_seconds=0,
                lower_bound_seconds=0,
                upper_bound_seconds=0,
                confidence="low",
                total_steps=config.total_steps,
                vram_feasible=False,
                vram_warning="No GPU detected — cannot estimate VRAM feasibility.",
                assumptions=["No GPU detected — estimate unavailable."],
                warnings=warnings,
                gpu_info=self.hardware.to_dict(),
            )

        total_steps = config.total_steps
        if total_steps <= 0:
            warnings.append("Total training steps unknown — cannot estimate.")
            return EstimateResult(
                mode="calibrated",
                estimated_seconds=0,
                lower_bound_seconds=0,
                upper_bound_seconds=0,
                confidence="low",
                total_steps=0,
                vram_feasible=True,
                assumptions=["Total steps unknown."],
                warnings=warnings,
                gpu_info=self.hardware.to_dict(),
            )

        estimated_seconds = total_steps / measured_steps_per_sec

        # Calibrated estimates are more reliable
        confidence = "high"
        lower = estimated_seconds * 0.85
        upper = estimated_seconds * 1.15

        # VRAM feasibility
        param_count = config.param_count_billions or 7.0
        vram_estimated_gb = self._estimate_vram_gb(config, param_count)
        vram_available_gb = self.hardware.max_vram_mb / 1024.0 if self.hardware.max_vram_mb else None
        vram_feasible = True
        vram_warning = None

        if vram_available_gb and vram_estimated_gb > vram_available_gb:
            vram_feasible = False
            vram_warning = (
                f"⚠ Estimated configuration may exceed available VRAM "
                f"({vram_estimated_gb:.1f}GB needed vs {vram_available_gb:.1f}GB available)."
            )
            warnings.append(vram_warning)

        # Throughput
        samples_per_sec = measured_steps_per_sec * config.batch_size
        tokens_per_sec = samples_per_sec * config.sequence_length

        assumptions.extend([
            f"Calibration measured: {measured_steps_per_sec:.2f} steps/sec",
            f"Total steps: {total_steps}",
            f"Estimated time: {_format_duration(estimated_seconds)}",
            "Calibrated estimate — based on measured throughput",
        ])

        return EstimateResult(
            mode="calibrated",
            estimated_seconds=estimated_seconds,
            lower_bound_seconds=lower,
            upper_bound_seconds=upper,
            confidence=confidence,
            throughput_steps_per_sec=measured_steps_per_sec,
            throughput_samples_per_sec=samples_per_sec,
            throughput_tokens_per_sec=tokens_per_sec,
            total_steps=total_steps,
            vram_feasible=vram_feasible,
            vram_warning=vram_warning,
            vram_estimated_gb=vram_estimated_gb,
            vram_available_gb=vram_available_gb,
            assumptions=assumptions,
            warnings=warnings,
            gpu_info=self.hardware.to_dict(),
            calibration_used=True,
        )

    def _estimate_vram_gb(self, config: TrainingConfig, param_count_billions: float) -> float:
        """Estimate VRAM usage in GB based on training config."""
        # Base model weights
        if config.precision == "4bit":
            bytes_per_param = 0.5
        elif config.precision == "8bit":
            bytes_per_param = 1.0
        elif config.precision in ("fp16", "bf16"):
            bytes_per_param = 2.0
        else:  # fp32
            bytes_per_param = 4.0

        # Model weights
        weights_gb = param_count_billions * bytes_per_param

        # Optimizer states (Adam: 2x params in fp32)
        if config.training_method in ("lora", "qlora", "peft"):
            # Only adapter params need optimizer states
            adapter_fraction = 0.01  # ~1% of params for LoRA
            optimizer_gb = param_count_billions * adapter_fraction * 8.0  # 2 states * 4 bytes
        else:
            optimizer_gb = param_count_billions * 8.0  # 2 states * 4 bytes

        # Activations (rough estimate based on batch size and sequence length)
        # ~2 bytes per activation element
        activation_gb = (
            config.batch_size * config.sequence_length * param_count_billions * 2.0 / 1e9
        )
        if config.gradient_checkpointing:
            activation_gb *= 0.3  # 70% reduction with checkpointing

        # Gradients
        if config.training_method in ("lora", "qlora", "peft"):
            gradients_gb = param_count_billions * 0.01 * bytes_per_param
        else:
            gradients_gb = param_count_billions * bytes_per_param

        # Overhead
        overhead_gb = 2.0

        total_gb = weights_gb + optimizer_gb + activation_gb + gradients_gb + overhead_gb
        return max(total_gb, 2.0)


def extract_training_config_from_context(context: Any) -> TrainingConfig:
    """Extract training configuration from a ProjectContext object."""
    config = TrainingConfig()

    # Model name
    config.model_name = getattr(context, "base_model", None)

    # Dataset samples
    stats = getattr(context, "project_statistics", {}) or {}
    for key, val in stats.items():
        if "sample" in key.lower() and isinstance(val, (int, float)):
            config.dataset_samples = int(val)
            break

    # Try to extract from configuration files
    config_files = getattr(context, "configuration_files", []) or []
    project_path = getattr(context, "project_path", "")

    import re
    from pathlib import Path

    for cf in config_files:
        try:
            cfg_path = Path(cf) if Path(cf).is_absolute() else Path(project_path) / cf
            if not cfg_path.exists():
                continue
            text = cfg_path.read_text(encoding="utf-8", errors="ignore")

            # Batch size
            m = re.search(r"per_device_train_batch_size\s*[:=]\s*(\d+)", text)
            if m:
                config.batch_size = int(m.group(1))

            # Gradient accumulation
            m = re.search(r"gradient_accumulation_steps\s*[:=]\s*(\d+)", text)
            if m:
                config.gradient_accumulation = int(m.group(1))

            # Epochs
            m = re.search(r"num_train_epochs\s*[:=]\s*([\d.]+)", text)
            if m:
                config.epochs = int(float(m.group(1)))

            # Max steps
            m = re.search(r"max_steps\s*[:=]\s*(\d+)", text)
            if m:
                config.max_steps = int(m.group(1))

            # Sequence length
            m = re.search(r"max_seq_length\s*[:=]\s*(\d+)", text)
            if m:
                config.sequence_length = int(m.group(1))

            # Precision
            if "fp16" in text.lower() or "float16" in text.lower():
                config.precision = "fp16"
            elif "bf16" in text.lower() or "bfloat16" in text.lower():
                config.precision = "bf16"
            elif "4bit" in text.lower() or "load_in_4bit" in text.lower():
                config.precision = "4bit"
            elif "8bit" in text.lower() or "load_in_8bit" in text.lower():
                config.precision = "8bit"

            # Training method
            if "qlora" in text.lower():
                config.training_method = "qlora"
            elif "lora" in text.lower() or "peft" in text.lower():
                config.training_method = "lora"
            elif "peft" in text.lower():
                config.training_method = "peft"

            # Gradient checkpointing
            if "gradient_checkpointing" in text.lower() and "true" in text.lower():
                config.gradient_checkpointing = True

            # Dataloader workers
            m = re.search(r"dataloader_num_workers\s*[:=]\s*(\d+)", text)
            if m:
                config.dataloader_workers = int(m.group(1))

            # Framework / distributed strategy
            if "deepspeed" in text.lower():
                config.framework = "deepspeed"
                config.distributed_strategy = "deepspeed"
            elif "fsdp" in text.lower():
                config.framework = "fsdp"
                config.distributed_strategy = "fsdp"
            elif "accelerate" in text.lower():
                config.framework = "accelerate"
                config.distributed_strategy = "accelerate"
            elif "ddp" in text.lower() or "distributed" in text.lower():
                config.framework = "pytorch"
                config.distributed_strategy = "ddp"
        except Exception as e:
            logger.debug(f"Failed to parse config {cf}: {e}")

    # Extract param count from model name
    if config.model_name:
        name_lower = config.model_name.lower()
        param_map = [
            ("70b", 70.0), ("72b", 72.0), ("7b", 7.0), ("8b", 8.0),
            ("2.7b", 2.7), ("2b", 2.0), ("1.1b", 1.1), ("1b", 1.0),
            ("13b", 13.0), ("3b", 3.0), ("30b", 30.0), ("34b", 34.0),
        ]
        for pattern, params in param_map:
            if pattern in name_lower:
                config.param_count_billions = params
                break

    return config


def run_calibration_benchmark(
    config: TrainingConfig,
    duration_seconds: float = 30.0,
    hardware: Optional[HardwareInfo] = None,
) -> Dict[str, Any]:
    """Run a short calibration benchmark to measure actual throughput.

    This is a SAFE benchmark that:
    - Runs for a short duration (default 30s, max 60s)
    - Does NOT modify any training state
    - Does NOT start full training
    - Is cancellable by the caller

    Returns measured throughput metrics.
    """
    hw = hardware or detect_gpus()
    if not hw.gpu_count:
        return {
            "success": False,
            "error": "No supported GPU detected. Calibration benchmark requires a GPU.",
            "measured_steps_per_sec": None,
        }

    # Clamp duration to safe range
    duration_seconds = max(10.0, min(duration_seconds, 60.0))

    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "success": False,
                "error": "CUDA not available. Calibration benchmark requires CUDA.",
                "measured_steps_per_sec": None,
            }

        device = torch.device("cuda:0")

        # Create a small representative workload
        # Use a small model to avoid OOM
        param_count = config.param_count_billions or 1.0
        # Scale model size down for benchmark (max ~1B params)
        benchmark_params = min(param_count, 1.0)

        # Create a simple model with the right parameter count
        hidden_size = 512
        num_layers = max(1, int(benchmark_params * 1e9 / (hidden_size * hidden_size * 12)))
        num_layers = min(num_layers, 8)

        import torch.nn as nn

        class BenchmarkModel(nn.Module):
            def __init__(self, hidden=512, layers=4):
                super().__init__()
                self.embed = nn.Embedding(32000, hidden)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=hidden,
                        nhead=8,
                        dim_feedforward=hidden * 4,
                        batch_first=True,
                    ) for _ in range(layers)
                ])
                self.ln = nn.LayerNorm(hidden)

            def forward(self, x):
                h = self.embed(x)
                for layer in self.layers:
                    h = layer(h)
                return self.ln(h)

        model = BenchmarkModel(hidden=hidden_size, layers=num_layers).to(device)
        model.train()

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create dummy batch
        batch_size = min(config.batch_size, 8)
        seq_len = min(config.sequence_length, 512)
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            output = model(input_ids)
            loss = output.mean()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        # Measure throughput
        start_time = time.time()
        steps = 0
        while time.time() - start_time < duration_seconds:
            optimizer.zero_grad()
            output = model(input_ids)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            steps += 1

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        steps_per_sec = steps / elapsed
        samples_per_sec = steps_per_sec * batch_size
        tokens_per_sec = samples_per_sec * seq_len

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

        return {
            "success": True,
            "measured_steps_per_sec": round(steps_per_sec, 4),
            "measured_samples_per_sec": round(samples_per_sec, 4),
            "measured_tokens_per_sec": round(tokens_per_sec, 4),
            "benchmark_duration_seconds": round(elapsed, 2),
            "benchmark_batch_size": batch_size,
            "benchmark_seq_len": seq_len,
            "benchmark_model_params": round(benchmark_params, 2),
            "gpu_name": hw.gpus[0].name if hw.gpus else "unknown",
        }
    except ImportError:
        return {
            "success": False,
            "error": "PyTorch not installed. Calibration benchmark requires PyTorch with CUDA.",
            "measured_steps_per_sec": None,
        }
    except Exception as e:
        logger.warning(f"Calibration benchmark failed: {e}")
        return {
            "success": False,
            "error": f"Calibration benchmark failed: {str(e)}",
            "measured_steps_per_sec": None,
        }