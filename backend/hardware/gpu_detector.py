"""GPU hardware detection using multiple mechanisms (PyTorch CUDA, nvidia-smi, CUDA runtime)."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single detected GPU."""
    index: int
    name: str
    vram_mb: Optional[int] = None
    compute_capability: Optional[str] = None
    utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None

    @property
    def vram_gb(self) -> Optional[float]:
        if self.vram_mb is None:
            return None
        return round(self.vram_mb / 1024.0, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "vram_mb": self.vram_mb,
            "vram_gb": self.vram_gb,
            "compute_capability": self.compute_capability,
            "utilization_percent": self.utilization_percent,
            "memory_utilization_percent": self.memory_utilization_percent,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
        }


@dataclass
class HardwareInfo:
    """Overall hardware detection result."""
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpus: List[GPUInfo] = field(default_factory=list)
    detection_method: str = "none"
    error: Optional[str] = None

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def heterogeneous(self) -> bool:
        """True if there are GPUs with different names."""
        names = {g.name for g in self.gpus if g.name}
        return len(names) > 1

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_mb or 0 for g in self.gpus)

    @property
    def max_vram_mb(self) -> int:
        return max((g.vram_mb or 0 for g in self.gpus), default=0)

    @property
    def min_vram_mb(self) -> int:
        return min((g.vram_mb or 0 for g in self.gpus), default=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpus": [g.to_dict() for g in self.gpus],
            "gpu_count": self.gpu_count,
            "heterogeneous": self.heterogeneous,
            "total_vram_mb": self.total_vram_mb,
            "max_vram_mb": self.max_vram_mb,
            "min_vram_mb": self.min_vram_mb,
            "detection_method": self.detection_method,
            "error": self.error,
        }


class GPUSpec:
    """Known GPU specifications for performance baseline (modular, extensible)."""
    # spec_key -> {name, tflops_fp16, vram_gb, family}
    _DATABASE: Dict[str, Dict[str, Any]] = {
        # Consumer GPUs
        "rtx_4090": {"name": "NVIDIA RTX 4090", "tflops_fp16": 165.0, "vram_gb": 24, "family": "consumer"},
        "rtx_4080": {"name": "NVIDIA RTX 4080", "tflops_fp16": 97.0, "vram_gb": 16, "family": "consumer"},
        "rtx_4070": {"name": "NVIDIA RTX 4070", "tflops_fp16": 59.0, "vram_gb": 12, "family": "consumer"},
        "rtx_4060": {"name": "NVIDIA RTX 4060", "tflops_fp16": 32.0, "vram_gb": 8, "family": "consumer"},
        "rtx_3090": {"name": "NVIDIA RTX 3090", "tflops_fp16": 71.0, "vram_gb": 24, "family": "consumer"},
        "rtx_3080": {"name": "NVIDIA RTX 3080", "tflops_fp16": 60.0, "vram_gb": 10, "family": "consumer"},
        "rtx_3070": {"name": "NVIDIA RTX 3070", "tflops_fp16": 40.0, "vram_gb": 8, "family": "consumer"},
        "rtx_3060": {"name": "NVIDIA RTX 3060", "tflops_fp16": 25.0, "vram_gb": 12, "family": "consumer"},
        "rtx_3050": {"name": "NVIDIA RTX 3050", "tflops_fp16": 18.0, "vram_gb": 8, "family": "consumer"},
        # Laptop GPUs
        "rtx_4060_laptop": {"name": "NVIDIA RTX 4060 Laptop GPU", "tflops_fp16": 25.0, "vram_gb": 8, "family": "laptop"},
        "rtx_4070_laptop": {"name": "NVIDIA RTX 4070 Laptop GPU", "tflops_fp16": 35.0, "vram_gb": 8, "family": "laptop"},
        "rtx_4080_laptop": {"name": "NVIDIA RTX 4080 Laptop GPU", "tflops_fp16": 45.0, "vram_gb": 12, "family": "laptop"},
        "rtx_4090_laptop": {"name": "NVIDIA RTX 4090 Laptop GPU", "tflops_fp16": 50.0, "vram_gb": 16, "family": "laptop"},
        # Data center / Enterprise
        "a100_40gb": {"name": "NVIDIA A100 (40GB)", "tflops_fp16": 312.0, "vram_gb": 40, "family": "datacenter"},
        "a100_80gb": {"name": "NVIDIA A100 (80GB)", "tflops_fp16": 312.0, "vram_gb": 80, "family": "datacenter"},
        "a10g": {"name": "NVIDIA A10G", "tflops_fp16": 125.0, "vram_gb": 24, "family": "datacenter"},
        "h100": {"name": "NVIDIA H100", "tflops_fp16": 989.0, "vram_gb": 80, "family": "datacenter"},
        "h800": {"name": "NVIDIA H800", "tflops_fp16": 989.0, "vram_gb": 80, "family": "datacenter"},
        "h200": {"name": "NVIDIA H200", "tflops_fp16": 989.0, "vram_gb": 141, "family": "datacenter"},
        "b200": {"name": "NVIDIA B200", "tflops_fp16": 2250.0, "vram_gb": 192, "family": "datacenter"},
        "l40": {"name": "NVIDIA L40", "tflops_fp16": 181.0, "vram_gb": 48, "family": "datacenter"},
        "l4": {"name": "NVIDIA L4", "tflops_fp16": 95.0, "vram_gb": 24, "family": "datacenter"},
        "v100": {"name": "NVIDIA V100", "tflops_fp16": 112.0, "vram_gb": 32, "family": "datacenter"},
        "p100": {"name": "NVIDIA P100", "tflops_fp16": 21.0, "vram_gb": 16, "family": "datacenter"},
        # Older / Other
        "gtx_1650": {"name": "NVIDIA GTX 1650", "tflops_fp16": 5.0, "vram_gb": 4, "family": "consumer"},
        "gtx_1660": {"name": "NVIDIA GTX 1660", "tflops_fp16": 6.0, "vram_gb": 6, "family": "consumer"},
        "gtx_1080": {"name": "NVIDIA GTX 1080", "tflops_fp16": 11.0, "vram_gb": 8, "family": "consumer"},
        "gtx_1080_ti": {"name": "NVIDIA GTX 1080 Ti", "tflops_fp16": 13.0, "vram_gb": 11, "family": "consumer"},
        # RTX 20-series
        "rtx_2050": {"name": "NVIDIA GeForce RTX 2050", "tflops_fp16": 8.0, "vram_gb": 4, "family": "laptop"},
        "rtx_2060": {"name": "NVIDIA RTX 2060", "tflops_fp16": 16.0, "vram_gb": 6, "family": "consumer"},
        "rtx_2070": {"name": "NVIDIA RTX 2070", "tflops_fp16": 21.0, "vram_gb": 8, "family": "consumer"},
        "rtx_2080": {"name": "NVIDIA RTX 2080", "tflops_fp16": 28.0, "vram_gb": 8, "family": "consumer"},
        "rtx_2080_ti": {"name": "NVIDIA RTX 2080 Ti", "tflops_fp16": 33.0, "vram_gb": 11, "family": "consumer"},
    }

    @classmethod
    def add_gpu(cls, key: str, name: str, tflops_fp16: float, vram_gb: float, family: str = "unknown") -> None:
        """Allow adding new GPU specs at runtime (modular DB)."""
        cls._DATABASE[key] = {
            "name": name,
            "tflops_fp16": float(tflops_fp16),
            "vram_gb": float(vram_gb),
            "family": family,
        }

    @classmethod
    def find(cls, gpu_name: str) -> Optional[Dict[str, Any]]:
        """Find a GPU spec by matching the detected GPU name."""
        if not gpu_name:
            return None
        name_lower = gpu_name.lower()

        # Try direct key match first
        direct = cls._DATABASE.get(name_lower)
        if direct:
            return direct

        # Fuzzy matching - check more specific (laptop) first
        laptop_keys = [k for k in cls._DATABASE if "laptop" in k]
        for key in laptop_keys:
            spec = cls._DATABASE[key]
            spec_name = spec["name"].lower()
            if spec_name in name_lower:
                return spec
        # Then check regular GPUs
        for key, spec in cls._DATABASE.items():
            spec_name = spec["name"].lower()
            # Check if spec name appears in the detected name, or vice versa
            if spec_name in name_lower or name_lower in spec_name:
                return spec

        # Match by patterns
        patterns = [
            ("rtx 4090", "rtx_4090", "RTX 4090"),
            ("rtx 4080", "rtx_4080", "RTX 4080"),
            ("rtx 4070", "rtx_4070", "RTX 4070"),
            ("rtx 4060", "rtx_4060", "RTX 4060"),
            ("rtx 3090", "rtx_3090", "RTX 3090"),
            ("rtx 3080", "rtx_3080", "RTX 3080"),
            ("rtx 3070", "rtx_3070", "RTX 3070"),
            ("rtx 3060", "rtx_3060", "RTX 3060"),
            ("rtx 3050", "rtx_3050", "RTX 3050"),
            ("a100", "a100_80gb", "NVIDIA A100"),
            ("h100", "h100", "NVIDIA H100"),
            ("h200", "h200", "NVIDIA H200"),
            ("b200", "b200", "NVIDIA B200"),
            ("v100", "v100", "NVIDIA V100"),
            ("p100", "p100", "NVIDIA P100"),
            ("l40", "l40", "NVIDIA L40"),
            ("l4", "l4", "NVIDIA L4"),
            ("a10g", "a10g", "NVIDIA A10G"),
            ("gtx 1650", "gtx_1650", "NVIDIA GTX 1650"),
            ("gtx 1660", "gtx_1660", "NVIDIA GTX 1660"),
            ("gtx 1080", "gtx_1080_ti", "NVIDIA GTX 1080 Ti"),
        ]
        for pattern, key, _ in patterns:
            if pattern in name_lower:
                return cls._DATABASE.get(key)

        # Check VRAM-based matching as fallback (using name patterns)
        return None

    @classmethod
    def get_known_keys(cls) -> List[str]:
        return list(cls._DATABASE.keys())


class GPUDetector:
    """Detect GPU hardware using multiple strategies."""

    def __init__(self):
        self._hardware: Optional[HardwareInfo] = None

    def detect(self) -> HardwareInfo:
        """Detect GPU hardware. Tries PyTorch CUDA first, then nvidia-smi."""
        if self._hardware:
            return self._hardware

        # Strategy 1: PyTorch CUDA
        hardware = self._detect_with_torch()
        if hardware and hardware.gpu_count > 0:
            self._hardware = hardware
            return hardware

        # Strategy 2: nvidia-smi
        hardware = self._detect_with_nvidia_smi()
        if hardware and hardware.gpu_count > 0:
            self._hardware = hardware
            return hardware

        # Strategy 3: CUDA runtime / environment
        hardware = self._detect_with_env()
        if hardware and hardware.gpu_count > 0:
            self._hardware = hardware
            return hardware

        # No GPU found
        hardware = HardwareInfo(
            cuda_available=False,
            gpus=[],
            detection_method="none",
            error="No supported GPU detected",
        )
        self._hardware = hardware
        return hardware

    def detect_with_torch(self) -> HardwareInfo:
        return self._detect_with_torch()

    def _detect_with_torch(self) -> HardwareInfo:
        """Detect GPUs using PyTorch CUDA APIs."""
        try:
            import torch

            if not torch.cuda.is_available():
                # CUDA not available, but check if torch exists at least
                return HardwareInfo(cuda_available=False, gpus=[], detection_method="torch")

            gpus: List[GPUInfo] = []
            count = torch.cuda.device_count()
            cuda_version = torch.version.cuda

            for i in range(count):
                try:
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    vram_mb = int(getattr(props, "total_memory", 0) / (1024 * 1024))
                    cc = f"{props.major}.{props.minor}"

                    # Try to get utilization
                    utilization = None
                    memory_util = None
                    try:
                        utilization = float(torch.cuda.utilization(device=i) if hasattr(torch.cuda, "utilization") else 0)
                    except Exception:
                        pass

                    gpus.append(GPUInfo(
                        index=i,
                        name=name,
                        vram_mb=vram_mb or None,
                        compute_capability=cc,
                        utilization_percent=utilization,
                        memory_utilization_percent=memory_util,
                        cuda_version=cuda_version,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to get GPU properties for index {i}: {e}")
                    gpus.append(GPUInfo(index=i, name=f"Unknown GPU {i}"))

            if not gpus:
                return HardwareInfo(cuda_available=False, gpus=[], detection_method="torch")

            return HardwareInfo(
                cuda_available=True,
                cuda_version=cuda_version,
                gpus=gpus,
                detection_method="torch",
            )
        except ImportError:
            logger.debug("PyTorch not installed — trying nvidia-smi")
            return HardwareInfo(cuda_available=False, gpus=[], detection_method="none")
        except Exception as e:
            logger.warning(f"PyTorch GPU detection failed: {e}")
            return HardwareInfo(cuda_available=False, gpus=[], detection_method="none", error=str(e))

    def _detect_with_nvidia_smi(self) -> HardwareInfo:
        """Detect GPUs using nvidia-smi (no PyTorch required)."""
        try:
            nvidia_smi = shutil.which("nvidia-smi")
            if not nvidia_smi:
                # Try common Windows paths
                for path in [
                    r"C:\Windows\System32\nvidia-smi.exe",
                    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                ]:
                    if os.path.exists(path):
                        nvidia_smi = path
                        break

            if not nvidia_smi:
                return HardwareInfo(cuda_available=False, gpus=[], detection_method="none")

            # Query via nvidia-smi --query-gpu
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index,name,memory.total,driver_version,compute_cap,utilization.gpu,utilization.memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                logger.warning(f"nvidia-smi query failed: {result.stderr}")
                return HardwareInfo(cuda_available=False, gpus=[], detection_method="none")

            # Also query CUDA version. Only one extra invocation is needed:
            # the bare `nvidia-smi` banner carries "CUDA Version: x.y".
            cuda_version = None
            try:
                smi_output = subprocess.run(
                    [nvidia_smi],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if smi_output.returncode == 0:
                    match = re.search(r"CUDA Version:\s*([\d.]+)", smi_output.stdout)
                    if match:
                        cuda_version = match.group(1)
            except Exception:
                pass

            gpus: List[GPUInfo] = []
            lines = [
                line.strip() for line in result.stdout.strip().splitlines() if line.strip()
            ]
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    index = int(parts[0])
                    name = parts[1]
                    vram_mb = int(float(parts[2]))
                    driver = parts[3]
                    cc = parts[4] if len(parts) > 4 else None
                    util = float(parts[5]) if len(parts) > 5 else None
                    mem_util = float(parts[6]) if len(parts) > 6 else None

                    gpus.append(GPUInfo(
                        index=index,
                        name=name,
                        vram_mb=vram_mb,
                        compute_capability=cc,
                        utilization_percent=util,
                        memory_utilization_percent=mem_util,
                        driver_version=driver,
                        cuda_version=cuda_version,
                    ))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse nvidia-smi line '{line}': {e}")

            if not gpus:
                return HardwareInfo(cuda_available=False, gpus=[], detection_method="none")

            return HardwareInfo(
                cuda_available=True,
                cuda_version=cuda_version,
                gpus=gpus,
                detection_method="nvidia-smi",
            )
        except FileNotFoundError:
            return HardwareInfo(cuda_available=False, gpus=[], detection_method="none")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out")
            return HardwareInfo(cuda_available=False, gpus=[], detection_method="none")
        except Exception as e:
            logger.warning(f"nvidia-smi detection failed: {e}")
            return HardwareInfo(cuda_available=False, gpus=[], detection_method="none", error=str(e))

    def _detect_with_env(self) -> HardwareInfo:
        """Detect GPUs using environment variables and known paths."""
        gpus: List[GPUInfo] = []
        cuda_available = False

        # Check CUDA_VISIBLE_DEVICES
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            cuda_available = True
            devices = [d.strip() for d in visible.split(",") if d.strip()]
            for i, dev in enumerate(devices):
                if dev == "-1":
                    continue
                try:
                    index = int(dev)
                except ValueError:
                    index = i
                gpus.append(GPUInfo(index=index, name=f"GPU {index} (CUDA env)", cuda_version=os.environ.get("CUDA_VERSION")))

        if gpus:
            return HardwareInfo(
                cuda_available=cuda_available,
                gpus=gpus,
                detection_method="environment",
            )

        return HardwareInfo(cuda_available=False, gpus=[], detection_method="none")


# Convenience singleton
_detector = GPUDetector()


def detect_gpus() -> HardwareInfo:
    """Detect GPUs (cached per process lifetime)."""
    return _detector.detect()


def detect_gpus_dict() -> Dict[str, Any]:
    """Detect GPUs and return as dict for API responses."""
    return detect_gpus().to_dict()


def get_gpu_spec(gpu_name: str) -> Optional[Dict[str, Any]]:
    """Get known GPU spec for a detected GPU name."""
    return GPUSpec.find(gpu_name)