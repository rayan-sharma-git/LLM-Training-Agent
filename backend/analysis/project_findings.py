"""Concise, project-specific findings for the Analyze action.

Everything in here is read from the project itself or computed from values that
were read from it.  The module never supplies a model, GPU, training setting,
runtime or price of its own:

* the base model, method and training arguments come from
  :mod:`analysis.code_inspector` (with file and line);
* the GPU is the one detected on this machine, or the one the user *stated*
  (VS Code setting or chat message) for a cloud job;
* runtime comes from :class:`hardware.gpu_time_estimator.GPUTimeEstimator`;
* a cost is produced only for a **cloud** GPU **and** only when the user
  supplied the hourly rate.  A local GPU never gets a rental cost.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hardware.gpu_detector import GPUInfo, HardwareInfo, get_gpu_spec
from hardware.gpu_time_estimator import (
    EstimateResult,
    GPUTimeEstimator,
    TrainingConfig,
    extract_training_config_from_context,
)

logger = logging.getLogger(__name__)

HARDWARE_MODES = ("auto", "local", "cloud")
_MAX_RATE = 1000.0


@dataclass
class HardwareProfile:
    """Where the user trains: this machine's GPU, or a rented cloud GPU."""

    mode: str = "auto"
    gpu_name: str = ""
    hourly_rate_usd: Optional[float] = None
    provider: str = ""
    gpu_count: int = 1

    @classmethod
    def from_dict(cls, data: Any) -> "HardwareProfile":
        """Build a validated profile from untrusted input (request body, settings)."""
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode") or "auto").strip().lower()
        if mode not in HARDWARE_MODES:
            mode = "auto"
        gpu_name = str(data.get("gpuName") or data.get("gpu_name") or "").strip()[:80]
        provider = str(data.get("provider") or "").strip()[:60]
        rate_raw = data.get("hourlyRateUsd", data.get("hourly_rate_usd"))
        rate: Optional[float] = None
        try:
            if rate_raw not in (None, ""):
                parsed = float(rate_raw)
                if 0 < parsed <= _MAX_RATE:
                    rate = round(parsed, 4)
        except (TypeError, ValueError):
            rate = None
        try:
            count = int(data.get("gpuCount") or data.get("gpu_count") or 1)
        except (TypeError, ValueError):
            count = 1
        return cls(mode=mode, gpu_name=gpu_name, hourly_rate_usd=rate, provider=provider,
                   gpu_count=min(max(count, 1), 64))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "gpuName": self.gpu_name,
            "hourlyRateUsd": self.hourly_rate_usd,
            "provider": self.provider,
            "gpuCount": self.gpu_count,
        }

    @property
    def is_set(self) -> bool:
        return self.mode != "auto" or bool(self.gpu_name) or self.hourly_rate_usd is not None


_RATE_RE = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:/|per|an|a)?\s*(?:hr|hour|h)\b|"
    r"(\d+(?:\.\d+)?)\s*(?:usd|dollars?)\s*(?:/|per|an|a)\s*(?:hr|hour|h)\b", re.I)
_GPU_RE = re.compile(
    r"\b(h200|h100|h800|b200|a100(?:[ -]?(?:40|80)\s?gb)?|a10g|l40s?|l4|v100|p100|t4|"
    r"rtx[ -]?(?:[2-4]0[5-9]0)(?:[ -]?ti)?|gtx[ -]?1[0-9]{3}(?:[ -]?ti)?)\b", re.I)
_PROVIDERS = ("runpod", "vast.ai", "vastai", "lambda labs", "lambda", "colab", "paperspace",
              "aws", "gcp", "google cloud", "azure", "modal", "coreweave", "kaggle", "sagemaker")


def parse_hardware_statement(text: str) -> Optional[HardwareProfile]:
    """Read a hardware statement such as "I train on a RunPod A100 at $1.89/hr".

    Returns ``None`` unless the message actually states a cloud/local choice, a
    GPU or an hourly rate, so ordinary questions are never mistaken for one.
    """
    if not text:
        return None
    lowered = text.lower()
    rate_match = _RATE_RE.search(text)
    gpu_match = _GPU_RE.search(text)
    provider = next((p for p in _PROVIDERS if re.search(rf"\b{re.escape(p)}\b", lowered)), "")
    says_cloud = bool(provider) or bool(re.search(r"\b(cloud|rent(?:ed|ing)?|hosted)\b", lowered))
    says_local = bool(re.search(r"\b(local|my own|my pc|my laptop|my machine|on[- ]prem)\b", lowered))
    uses_verb = bool(re.search(r"\b(i(?:'m| am)? (?:using|training|running|renting)|i use|i train|we use|"
                               r"my gpu|training on|running on)\b", lowered))
    if not (rate_match or ((gpu_match or provider or says_local) and uses_verb)):
        return None
    mode = "auto"
    if says_cloud or rate_match:
        mode = "cloud"
    elif says_local:
        mode = "local"
    rate: Optional[float] = None
    if rate_match:
        rate = float(rate_match.group(1) or rate_match.group(2))
    gpu = ""
    if gpu_match:
        gpu = re.sub(r"\s+", " ", gpu_match.group(1)).strip().upper().replace("-", " ")
        gpu = re.sub(r"^(RTX|GTX)(\d)", r"\1 \2", gpu)
    return HardwareProfile.from_dict({
        "mode": mode, "gpuName": gpu, "hourlyRateUsd": rate,
        "provider": provider.title() if provider else "",
    })


def _hardware_info_for(name: str, count: int) -> Optional[HardwareInfo]:
    """A HardwareInfo for a *named* GPU, only when its specs are in the reference table."""
    spec = get_gpu_spec(name)
    if not spec:
        return None
    gpus = [GPUInfo(index=i, name=spec["name"], vram_mb=int(spec["vram_gb"] * 1024))
            for i in range(max(1, count))]
    return HardwareInfo(cuda_available=True, gpus=gpus, detection_method="user-specified")


def _evidence(entry: Any) -> str:
    if isinstance(entry, dict):
        where = f"{entry.get('file')}:{entry.get('line')}" if entry.get("file") else ""
        return where
    return ""


def _val(source: Dict[str, Any], key: str) -> Any:
    entry = source.get(key)
    return entry.get("value") if isinstance(entry, dict) else None


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = seconds / 3600
    if hours < 48:
        h = int(hours)
        m = int(round((hours - h) * 60))
        if m == 60:
            h, m = h + 1, 0
        return f"{h} h {m:02d} min" if m else f"{h} h"
    return f"{hours / 24:.1f} days"


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def build_project_findings(
    context: Any,
    *,
    dataset_samples: Optional[int],
    detected_hardware: Optional[Dict[str, Any]],
    profile: Optional[HardwareProfile] = None,
    tokens_per_sample: Optional[float] = None,
    advisor_param_billions: Optional[float] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[EstimateResult], TrainingConfig]:
    """Compose the findings dictionary (also rendered by :func:`render_findings`).

    Returns ``(findings, estimate, config)``.  ``estimate`` is ``None`` when no
    known GPU or no dataset size is available; the caller then reports the gap
    instead of a number.
    """
    profile = profile or HardwareProfile()
    insights = getattr(context, "training_insights", None) or {}
    args = insights.get("training_args") or {}
    lora = insights.get("lora") or {}
    quant = insights.get("quantization") or {}
    base = insights.get("base_model") or None
    method = insights.get("method") or "unknown"

    findings: Dict[str, Any] = {
        "training_scripts": insights.get("training_scripts") or [],
        "base_model": base,
        "parameter_count_billions": insights.get("parameter_count_billions"),
        "method": method,
        "method_reason": insights.get("method_reason"),
        "lora": lora,
        "quantization": quant,
        "training_args": args,
        "datasets_in_code": insights.get("datasets") or [],
        "dataset_files": list(getattr(context, "dataset_paths", []) or []),
        "dataset_samples": dataset_samples,
        "libraries": insights.get("libraries") or [],
        "cloud_signals": insights.get("cloud_signals") or [],
        "hardware": {},
        "time": None,
        "cost": None,
        "missing": [],
    }

    # --- which GPU does the training run on? ---------------------------------
    detected_gpus = (detected_hardware or {}).get("gpus") or []
    cloud_names = [c.get("value") for c in findings["cloud_signals"] if isinstance(c, dict)]
    hardware_info: Optional[HardwareInfo] = None
    hw: Dict[str, Any] = {"mode": None, "gpu": None, "source": None, "notes": []}

    mode = profile.mode
    if mode == "auto" and profile.gpu_name and profile.hourly_rate_usd:
        mode = "cloud"  # a stated rate only makes sense for a rented GPU
    if mode == "auto":
        mode = "local" if detected_gpus else ("cloud" if profile.gpu_name else "unknown")

    if mode == "cloud":
        hw["mode"] = "cloud"
        if profile.gpu_name:
            hardware_info = _hardware_info_for(profile.gpu_name, profile.gpu_count)
            hw["gpu"] = profile.gpu_name
            hw["source"] = "stated by you"
            if hardware_info is None:
                hw["notes"].append(
                    f"'{profile.gpu_name}' is not in the GPU reference table, so no runtime can be computed for it.")
        else:
            hw["notes"].append("You said the job runs on a cloud GPU but not which one.")
            findings["missing"].append("the cloud GPU model (for example 'A100 80GB')")
        if profile.provider:
            hw["provider"] = profile.provider
    elif mode == "local":
        hw["mode"] = "local"
        if detected_gpus:
            first = detected_gpus[0]
            hw["gpu"] = first.get("name")
            hw["vram_gb"] = round((first.get("vram_mb") or 0) / 1024, 1) or None
            hw["gpu_count"] = len(detected_gpus)
            hw["source"] = "detected on this machine"
            hardware_info = HardwareInfo(
                cuda_available=True,
                gpus=[GPUInfo(index=g.get("index", i), name=g.get("name") or "unknown",
                              vram_mb=g.get("vram_mb")) for i, g in enumerate(detected_gpus)],
                detection_method=(detected_hardware or {}).get("detection_method") or "detected",
            )
        elif profile.gpu_name:
            hw["gpu"] = profile.gpu_name
            hw["source"] = "stated by you"
            hardware_info = _hardware_info_for(profile.gpu_name, profile.gpu_count)
            if hardware_info is None:
                hw["notes"].append(f"'{profile.gpu_name}' is not in the GPU reference table.")
        else:
            hw["notes"].append("No GPU was detected on this machine.")
    else:
        hw["mode"] = "unknown"
        hw["notes"].append("No GPU was detected on this machine and none was stated.")
        if cloud_names:
            hw["notes"].append(
                f"The code references {', '.join(sorted(set(map(str, cloud_names))))}; if that is where you "
                "train, tell me the GPU and its hourly rate.")
            findings["missing"].append("the GPU you train on")
    findings["hardware"] = hw

    # --- runtime ----------------------------------------------------------------
    config: TrainingConfig = extract_training_config_from_context(context)
    for attr, raw in (config_overrides or {}).items():
        if raw and hasattr(config, attr):
            setattr(config, attr, raw)
    if config.param_count_billions is None and advisor_param_billions:
        config.param_count_billions = advisor_param_billions
    if dataset_samples:
        config.dataset_samples = int(dataset_samples)
    if config.dataset_samples and tokens_per_sample:
        # Measured average sample length caps the padded sequence length.
        config.sequence_length = max(16, min(config.sequence_length, int(round(tokens_per_sample))))
    config_summary = {
        "epochs": config.epochs,
        "effective_batch_size": config.effective_batch_size,
        "sequence_length": config.sequence_length,
        "precision": config.precision,
        "method": config.training_method,
        "total_steps": config.total_steps,
    }
    findings["time_inputs"] = config_summary
    estimate: Optional[EstimateResult] = None
    if hardware_info is None:
        if hw["mode"] != "cloud" or profile.gpu_name:
            findings["missing"].append("a known GPU to estimate runtime")
    elif not config.dataset_samples and not config.max_steps:
        findings["missing"].append("the dataset size (no readable training records were found)")
    else:
        try:
            estimate = GPUTimeEstimator(hardware_info).estimate(config, mode="quick")
        except Exception as error:  # pragma: no cover - estimator is defensive already
            logger.warning("Runtime estimate failed: %s", error)
            estimate = None
        if estimate is not None and estimate.estimated_seconds > 0:
            findings["time"] = {
                "seconds": estimate.estimated_seconds,
                "low_seconds": estimate.lower_bound_seconds,
                "high_seconds": estimate.upper_bound_seconds,
                "human": _fmt_duration(estimate.estimated_seconds),
                "range": f"{_fmt_duration(estimate.lower_bound_seconds)} to "
                         f"{_fmt_duration(estimate.upper_bound_seconds)}",
                "total_steps": estimate.total_steps,
                "confidence": estimate.confidence,
                "vram_estimated_gb": estimate.vram_estimated_gb,
                "vram_available_gb": estimate.vram_available_gb,
                "vram_feasible": estimate.vram_feasible,
                "warnings": [w for w in estimate.warnings][:3],
            }

    # --- cost: cloud GPU with a stated rate only ---------------------------------
    if hw["mode"] == "cloud":
        t = findings["time"]
        if profile.hourly_rate_usd and t:
            gpus = max(1, profile.gpu_count)
            rate = profile.hourly_rate_usd
            findings["cost"] = {
                "hourly_rate_usd": rate,
                "gpu_count": gpus,
                "estimate_usd": round(t["seconds"] / 3600 * rate * gpus, 2),
                "low_usd": round(t["low_seconds"] / 3600 * rate * gpus, 2),
                "high_usd": round(t["high_seconds"] / 3600 * rate * gpus, 2),
                "assumptions": [
                    f"Rate of {_fmt_money(rate)}/GPU-hour is the one you provided, not a looked-up price.",
                    "Billed only for the estimated training runtime.",
                    "Excludes setup and idle time, storage, data transfer and failed or repeated runs.",
                ],
            }
        elif not profile.hourly_rate_usd:
            findings["missing"].append(
                "the hourly rate of your cloud GPU (for example \"I use an A100 at $1.89/hr\")")
    return findings, estimate, config


def _row(label: str, value: Any, where: str = "") -> str:
    suffix = f"  ({where})" if where else ""
    return f"- **{label}:** {value}{suffix}"


def render_findings(findings: Dict[str, Any]) -> List[str]:
    """Render the findings as compact Markdown lines."""
    out: List[str] = []
    scripts = findings.get("training_scripts") or []
    base = findings.get("base_model")
    args = findings.get("training_args") or {}
    lora = findings.get("lora") or {}
    quant = findings.get("quantization") or {}

    if scripts:
        out.append(_row("Training script", ", ".join(f"`{s['path']}`" for s in scripts[:4])))
    else:
        out.append("- **Training script:** none found")
    if base:
        params = findings.get("parameter_count_billions")
        size = f", about {params:g}B parameters" if params else ""
        out.append(_row("Base model", f"`{base['value']}`{size}", _evidence(base)))
    else:
        out.append("- **Base model:** not set in the code or config files")

    method = findings.get("method")
    method_names = {"qlora": "QLoRA", "lora": "LoRA", "full": "Full fine-tuning"}
    if method in method_names:
        detail = ""
        if method in ("lora", "qlora") and lora:
            bits = []
            if _val(lora, "r") is not None:
                bits.append(f"r={_val(lora, 'r')}")
            if _val(lora, "lora_alpha") is not None:
                bits.append(f"alpha={_val(lora, 'lora_alpha')}")
            if _val(lora, "lora_dropout") is not None:
                bits.append(f"dropout={_val(lora, 'lora_dropout')}")
            targets = _val(lora, "target_modules")
            if targets:
                bits.append("targets=" + (",".join(map(str, targets)) if isinstance(targets, list)
                                          else str(targets)))
            detail = " (" + ", ".join(bits) + ")" if bits else ""
        if method == "qlora" and _val(quant, "load_in_4bit"):
            q = _val(quant, "bnb_4bit_quant_type")
            detail += f", 4-bit base{' ' + str(q) if q else ''}"
        where = _evidence(next(iter(lora.values()), None)) if lora else ""
        out.append(_row("Method", f"{method_names[method]}{detail}", where))

    labels = (
        ("num_train_epochs", "Epochs"), ("learning_rate", "Learning rate"),
        ("per_device_train_batch_size", "Batch size"),
        ("gradient_accumulation_steps", "Gradient accumulation"),
        ("max_seq_length", "Max sequence length"), ("max_steps", "Max steps"),
        ("warmup_ratio", "Warmup ratio"), ("weight_decay", "Weight decay"),
        ("lr_scheduler_type", "LR scheduler"), ("optim", "Optimizer"),
    )
    shown = []
    for key, label in labels:
        if key in args:
            shown.append(f"{label} {args[key]['value']} (`{_evidence(args[key])}`)")
    precision = []
    if _val(args, "bf16"):
        precision.append("bf16")
    if _val(args, "fp16"):
        precision.append("fp16")
    if precision:
        shown.append("Precision " + "/".join(precision))
    if _val(args, "gradient_checkpointing"):
        shown.append("Gradient checkpointing on")
    if shown:
        out.append("- **Training settings:** " + "; ".join(shown))
    else:
        out.append("- **Training settings:** none found in the code")

    ds_bits = []
    samples = findings.get("dataset_samples")
    if samples:
        ds_bits.append(f"{samples:,} records")
    files = findings.get("dataset_files") or []
    if files:
        ds_bits.append(", ".join(f"`{f}`" for f in files[:3]) + (f" +{len(files) - 3} more" if len(files) > 3 else ""))
    refs = findings.get("datasets_in_code") or []
    if refs:
        ds_bits.append("loaded in code as " + ", ".join(f"`{r['value']}`" for r in refs[:2]))
    if ds_bits:
        out.append(_row("Dataset", "; ".join(ds_bits)))

    hw = findings.get("hardware") or {}
    if hw.get("gpu"):
        mode_label = {"cloud": "cloud GPU", "local": "local GPU"}.get(hw.get("mode"), "GPU")
        extras = []
        if hw.get("vram_gb"):
            extras.append(f"{hw['vram_gb']:g} GB VRAM")
        if hw.get("provider"):
            extras.append(hw["provider"])
        extras.append(hw.get("source") or "")
        out.append(_row("Hardware", f"{hw['gpu']} ({mode_label}; {', '.join(e for e in extras if e)})"))
    for note in hw.get("notes", []):
        out.append(f"- **Hardware:** {note}")

    t = findings.get("time")
    if t:
        out.append(_row("Estimated training time", f"about {t['human']} (likely {t['range']})"))
        out.append(f"- **Basis:** {t['total_steps']:,} steps from the settings above; "
                   f"FLOPs-based estimate, confidence {t['confidence']}. Real runs vary with data loading.")
        if t.get("vram_estimated_gb") and t.get("vram_available_gb"):
            verdict = "should fit" if t.get("vram_feasible") else "may not fit"
            out.append(f"- **Memory:** about {t['vram_estimated_gb']:.1f} GB needed vs "
                       f"{t['vram_available_gb']:.0f} GB available, {verdict}.")

    cost = findings.get("cost")
    if cost:
        out.append(_row("Estimated GPU cost",
                        f"about {_fmt_money(cost['estimate_usd'])} "
                        f"({_fmt_money(cost['low_usd'])} to {_fmt_money(cost['high_usd'])}) at "
                        f"{_fmt_money(cost['hourly_rate_usd'])}/hour"))
        for assumption in cost["assumptions"]:
            out.append(f"  - {assumption}")

    missing = findings.get("missing") or []
    if missing:
        out.append("- **Needed for a complete estimate:** " + "; ".join(missing) + ".")
    return out


__all__ = [
    "HardwareProfile",
    "HARDWARE_MODES",
    "build_project_findings",
    "parse_hardware_statement",
    "render_findings",
]
