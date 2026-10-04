"""Tests for GPU detection and time estimation (mocked hardware)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

# Ensure hardware module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


from hardware.gpu_detector import (  # noqa: E402
    GPUInfo,
    GPUSpec,
    HardwareInfo,
    get_gpu_spec,
)
from hardware.gpu_time_estimator import (  # noqa: E402
    GPUTimeEstimator,
    TrainingConfig,
    EstimateResult,
    extract_training_config_from_context,
    run_calibration_benchmark,
)


def _make_hardware(gpus: List[Dict[str, Any]], cuda_available: bool = True) -> HardwareInfo:
    """Create a HardwareInfo from a list of GPU dicts."""
    gpu_objs = []
    for idx, g in enumerate(gpus):
        gpu_objs.append(GPUInfo(
            index=g.get("index", idx),
            name=g.get("name", f"GPU {idx}"),
            vram_mb=g.get("vram_mb", 8192),
        ))
    return HardwareInfo(
        cuda_available=cuda_available,
        cuda_version="12.1",
        gpus=gpu_objs,
        detection_method="test",
    )


def _make_config(**kwargs) -> TrainingConfig:
    """Create a TrainingConfig with defaults."""
    defaults = dict(
        model_name="TinyLlama 1.1B",
        param_count_billions=1.1,
        dataset_samples=50000,
        sequence_length=512,
        batch_size=4,
        gradient_accumulation=8,
        epochs=3,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


class TestGPUSpec:
    """Test GPU spec database lookups."""

    def test_find_known_gpu(self):
        spec = get_gpu_spec("NVIDIA RTX 4090")
        assert spec is not None
        assert spec["tflops_fp16"] > 100

    def test_find_unknown_gpu_returns_none(self):
        spec = get_gpu_spec("NVIDIA AMD Radeon 9999")
        assert spec is None

    def test_find_laptop_gpu(self):
        spec = get_gpu_spec("NVIDIA RTX 4060 Laptop GPU")
        assert spec is not None
        assert "Laptop" in spec["name"]

    def test_find_case_insensitive(self):
        spec = get_gpu_spec("nvidia rtx 3080")
        assert spec is not None

    def test_add_new_gpu(self):
        GPUSpec.add_gpu("test_gpu", "Test GPU XYZ", 55.0, 16, "test")
        spec = GPUSpec.find("Test GPU XYZ")
        assert spec is not None
        assert spec["tflops_fp16"] == 55.0


class TestHardwareInfo:
    """Test HardwareInfo properties."""

    def test_heterogeneous_gpu(self):
        hardware = _make_hardware([
            {"name": "NVIDIA RTX 4090", "vram_mb": 24576},
            {"name": "NVIDIA RTX 3060", "vram_mb": 12288},
        ])
        assert hardware.heterogeneous

    def test_identical_gpus_not_heterogeneous(self):
        hardware = _make_hardware([
            {"name": "NVIDIA RTX 4090", "vram_mb": 24576},
            {"name": "NVIDIA RTX 4090", "vram_mb": 24576},
        ])
        assert not hardware.heterogeneous

    def test_gpu_count(self):
        hardware = _make_hardware([
            {"name": "NVIDIA RTX 4090", "vram_mb": 24576},
            {"name": "NVIDIA RTX 4090", "vram_mb": 24576},
        ])
        assert hardware.gpu_count == 2

    def test_vram_properties(self):
        hardware = _make_hardware([
            {"name": "GPU A", "vram_mb": 8192},
            {"name": "GPU B", "vram_mb": 24576},
        ])
        assert hardware.total_vram_mb == 32768
        assert hardware.max_vram_mb == 24576
        assert hardware.min_vram_mb == 8192

    def test_no_gpu(self):
        hardware = _make_hardware([], cuda_available=False)
        assert hardware.gpu_count == 0
        assert not hardware.heterogeneous


class TestGPUTimeEstimator:
    """Test the GPU time estimation engine."""

    def test_single_gpu_quick_estimate(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        estimator = GPUTimeEstimator(hardware=hardware)
        config = _make_config()
        result = estimator.estimate(config, mode="quick")
        assert isinstance(result, EstimateResult)
        assert result.mode == "quick"
        assert result.estimated_seconds > 0
        assert result.lower_bound_seconds < result.estimated_seconds < result.upper_bound_seconds
        assert result.confidence in ("low", "medium", "high")
        assert result.total_steps > 0

    def test_no_gpu_estimate(self):
        hardware = _make_hardware([], cuda_available=False)
        estimator = GPUTimeEstimator(hardware=hardware)
        config = _make_config()
        result = estimator.estimate(config, mode="quick")
        assert result.estimated_seconds == 0
        assert result.confidence == "low"
        assert any("No supported GPU" in w for w in result.warnings)

    def test_unknown_gpu_lower_confidence(self):
        hardware = _make_hardware([{"name": "Unknown GPU X", "vram_mb": 8192}])
        estimator = GPUTimeEstimator(hardware=hardware)
        config = _make_config()
        result = estimator.estimate(config, mode="quick")
        assert result.confidence == "low"
        assert any("Unknown GPU" in w for w in result.warnings)

    def test_multi_gpu_scaling(self):
        single_hw = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        multi_hw = _make_hardware([
            {"name": "NVIDIA RTX 4090", "vram_mb": 24576},
            {"name": "NVIDIA RTX 4090", "vram_mb": 24576},
        ])
        estimator_single = GPUTimeEstimator(hardware=single_hw)
        estimator_multi = GPUTimeEstimator(hardware=multi_hw)
        config = _make_config(distributed_strategy="ddp")
        single_result = estimator_single.estimate(config, mode="quick")
        multi_result = estimator_multi.estimate(config, mode="quick")
        # Multi-GPU should be faster but not perfect linear (at least 1.5x speedup, max 2.5x)
        assert multi_result.estimated_seconds < single_result.estimated_seconds
        assert multi_result.estimated_seconds > single_result.estimated_seconds / 2.5

    def test_heterogeneous_gpu_warning(self):
        hardware = _make_hardware([
            {"name": "NVIDIA RTX 4090", "vram_mb": 24576},
            {"name": "NVIDIA RTX 3060", "vram_mb": 12288},
        ])
        estimator = GPUTimeEstimator(hardware=hardware)
        config = _make_config(distributed_strategy="ddp")
        result = estimator.estimate(config, mode="quick")
        assert any("Heterogeneous" in w for w in result.warnings)

    def test_vram_warning_when_insufficient(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 3050", "vram_mb": 4096}])
        estimator = GPUTimeEstimator(hardware=hardware)
        config = _make_config(
            param_count_billions=7.0,
            training_method="full",
            precision="fp32",
            batch_size=8,
            sequence_length=2048,
        )
        result = estimator.estimate(config, mode="quick")
        assert not result.vram_feasible
        assert result.vram_warning is not None

    def test_vram_sufficient(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        estimator = GPUTimeEstimator(hardware=hardware)
        config = _make_config(
            param_count_billions=7.0,
            training_method="qlora",
            precision="4bit",
        )
        result = estimator.estimate(config, mode="quick")
        assert result.vram_feasible

    def test_different_model_sizes(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        estimator = GPUTimeEstimator(hardware=hardware)
        small = estimator.estimate(_make_config(param_count_billions=1.1), mode="quick")
        large = estimator.estimate(_make_config(param_count_billions=70.0), mode="quick")
        assert large.estimated_seconds > small.estimated_seconds

    def test_different_seq_lengths(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        estimator = GPUTimeEstimator(hardware=hardware)
        short = estimator.estimate(_make_config(sequence_length=128), mode="quick")
        long = estimator.estimate(_make_config(sequence_length=2048), mode="quick")
        assert long.estimated_seconds > short.estimated_seconds

    def test_different_epochs(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        estimator = GPUTimeEstimator(hardware=hardware)
        one_epoch = estimator.estimate(_make_config(epochs=1), mode="quick")
        ten_epochs = estimator.estimate(_make_config(epochs=10), mode="quick")
        assert ten_epochs.estimated_seconds > one_epoch.estimated_seconds

    def test_calibrated_estimate(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        estimator = GPUTimeEstimator(hardware=hardware)
        config = _make_config()
        result = estimator.estimate(config, mode="calibrated", measured_steps_per_sec=2.84)
        assert result.mode == "calibrated"
        assert result.calibration_used
        assert result.confidence == "high"
        assert result.throughput_steps_per_sec == 2.84
        # ~4500 steps at 2.84 steps/sec ≈ 1584 sec
        assert 5 * 60 < result.estimated_seconds < 60 * 60

    def test_missing_config_estimates_total_steps_zero(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        estimator = GPUTimeEstimator(hardware=hardware)
        config = _make_config(dataset_samples=None)
        result = estimator.estimate(config, mode="quick")
        assert result.total_steps == 0


class TestTrainingConfigExtraction:
    """Test extracting training config from project context."""

    def test_extract_from_config_file(self):
        from models.schemas import ProjectContext
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_content = """
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 3
max_seq_length: 512
load_in_4bit: true
use_peft: true
"""
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(config_content)

            context = ProjectContext(
                project_name="test",
                project_path=str(tmpdir),
                base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                configuration_files=["config.yaml"],
            )
            config = extract_training_config_from_context(context)
            assert config.batch_size == 4
            assert config.gradient_accumulation == 8
            assert config.epochs == 3
            assert config.sequence_length == 512
            assert config.precision == "4bit"
            assert config.param_count_billions == 1.1

    def test_extract_from_empty_context(self):
        from models.schemas import ProjectContext
        context = ProjectContext(project_name="test", project_path="/tmp")
        config = extract_training_config_from_context(context)
        assert config.batch_size == 8
        assert config.epochs == 3
        assert config.sequence_length == 512
        assert config.param_count_billions is None


class TestCalibrationBenchmark:
    """Test calibration benchmark failure handling."""

    def test_no_gpu_fails_gracefully(self):
        hardware = _make_hardware([], cuda_available=False)
        config = _make_config()
        result = run_calibration_benchmark(config, duration_seconds=10, hardware=hardware)
        assert not result["success"]
        assert "GPU" in result["error"]

    def test_torch_not_installed_fails_gracefully(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        config = _make_config()
        with patch.dict(sys.modules, {"torch": None}):
            result = run_calibration_benchmark(config, duration_seconds=10, hardware=hardware)
            assert not result["success"]

    def test_benchmark_failure_handling(self):
        hardware = _make_hardware([{"name": "NVIDIA RTX 4090", "vram_mb": 24576}])
        config = _make_config()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = run_calibration_benchmark(config, duration_seconds=10, hardware=hardware)
            assert not result["success"]