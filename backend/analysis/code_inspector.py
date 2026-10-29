"""Static inspection of a project's training code.

The scanner only looked at file *names* (``train*.py``) and YAML/JSON configs,
so a project whose whole setup lives in ``fine_tune.py`` was reported as
"no training script, no base model, no hyper-parameters".  This module reads
the Python sources instead, using :mod:`ast`, and records **where** every value
came from (file and line) so nothing in the analysis is invented.

What it extracts (only when it is literally present in the code):

* training scripts, from their content (``Trainer``, ``TrainingArguments``,
  ``get_peft_model``, ``LoraConfig``...) as well as their names;
* the base model identifier (``from_pretrained("TinyLlama/...")``,
  ``model_name = "..."`` ...);
* the fine-tuning method: full, LoRA or QLoRA, with the adapter settings and
  the quantization settings;
* training arguments (epochs, learning rate, batch size, accumulation, sequence
  length, precision, ...), resolved through module constants and argparse
  defaults;
* dataset references and signals that the job runs on a rented cloud GPU.

Nothing here executes project code.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Directories that never contain the user's own training code.
SKIP_DIRS = frozenset({
    ".git", ".svn", ".hg", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".venv", "venv", "venv310", ".venv310", "env", ".env",
    ".vscode", ".idea", "dist", "build", "site-packages",
    ".llm_training_agent", ".llm_training_agent_backups", ".llm-training-agent",
})

MAX_PY_FILES = 400
MAX_FILE_BYTES = 512 * 1024

#: Arguments of ``TrainingArguments``-like constructors that matter for the analysis.
TRAINING_ARG_KEYS = (
    "num_train_epochs", "learning_rate", "per_device_train_batch_size",
    "gradient_accumulation_steps", "max_steps", "warmup_ratio", "warmup_steps",
    "weight_decay", "lr_scheduler_type", "optim", "fp16", "bf16",
    "gradient_checkpointing", "max_seq_length", "max_length", "logging_steps",
    "save_steps", "eval_steps", "dataloader_num_workers", "max_grad_norm",
    "per_device_eval_batch_size", "output_dir", "packing",
)

#: Calls whose keyword arguments are training arguments.
_TRAINING_ARG_CALLS = (
    "TrainingArguments", "Seq2SeqTrainingArguments", "SFTConfig", "DPOConfig",
    "ORPOConfig", "KTOConfig", "RewardConfig", "GRPOConfig", "PPOConfig",
    "UnslothTrainingArguments",
)

_LORA_KEYS = ("r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type",
              "use_rslora", "use_dora")

_MODEL_FAMILY_RE = re.compile(
    r"(llama|mistral|mixtral|tinyllama|qwen|phi|gemma|falcon|gpt2|gpt-j|gptj|gpt-neo|"
    r"opt-|bloom|t5|flan|bert|roberta|deepseek|yi-|olmo|smollm|starcoder|codellama|"
    r"pythia|stablelm|internlm|baichuan|chatglm|solar|zephyr|vicuna)", re.I)

_CLOUD_SIGNALS = (
    (re.compile(r"google\.colab|from google import colab", re.I), "Google Colab"),
    (re.compile(r"\brunpod\b", re.I), "RunPod"),
    (re.compile(r"vast\.ai|vastai", re.I), "Vast.ai"),
    (re.compile(r"lambda[ _-]?labs|lambdalabs", re.I), "Lambda Labs"),
    (re.compile(r"\bmodal\.(?:com|Stub|App)|import modal\b", re.I), "Modal"),
    (re.compile(r"paperspace", re.I), "Paperspace"),
    (re.compile(r"sagemaker", re.I), "AWS SageMaker"),
    (re.compile(r"azureml|azure\.ai\.ml", re.I), "Azure ML"),
    (re.compile(r"vertexai|google\.cloud\.aiplatform", re.I), "Google Vertex AI"),
    (re.compile(r"kaggle_secrets|/kaggle/", re.I), "Kaggle"),
)

_TRAIN_NAME_RE = re.compile(
    r"(^|[_\-])(train|training|fine[_\-]?tun\w*|finetun\w*|sft|lora|qlora|dpo|orpo|peft|run_clm|run_sft)"
    r"([_\-.]|$)", re.I)
_EVAL_NAME_RE = re.compile(r"(^|[_\-])(eval|evaluate|evaluation|benchmark)([_\-.]|$)", re.I)
_INFER_NAME_RE = re.compile(r"(^|[_\-])(inference|infer|predict|serve|generate|chat)([_\-.]|$)", re.I)

#: Content markers and the weight each adds to "this is a training script".
_TRAIN_CONTENT = (
    (re.compile(r"\bSFTTrainer\b|\bDPOTrainer\b|\bORPOTrainer\b|\bGRPOTrainer\b"), 3, "SFTTrainer"),
    (re.compile(r"\bTrainer\s*\("), 3, "Trainer"),
    (re.compile(r"\bTrainingArguments\b|\bSFTConfig\b|\bSeq2SeqTrainingArguments\b"), 2, "TrainingArguments"),
    (re.compile(r"\bget_peft_model\b|\bLoraConfig\b"), 2, "LoRA/PEFT"),
    (re.compile(r"\.train\s*\("), 2, ".train()"),
    (re.compile(r"\bloss\.backward\s*\("), 2, "loss.backward()"),
    (re.compile(r"\boptimizer\.step\s*\("), 1, "optimizer.step()"),
    (re.compile(r"\bFastLanguageModel\b"), 2, "Unsloth"),
)


@dataclass
class Evidence:
    """A value together with the file and line it was read from."""

    value: Any
    file: str
    line: int

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "file": self.file, "line": self.line}


@dataclass
class ScriptInfo:
    path: str
    role: str
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "role": self.role, "reasons": list(self.reasons)}


@dataclass
class TrainingInspection:
    """Everything the inspector could establish from the code."""

    training_scripts: List[ScriptInfo] = field(default_factory=list)
    evaluation_scripts: List[str] = field(default_factory=list)
    inference_scripts: List[str] = field(default_factory=list)
    base_model: Optional[Evidence] = None
    method: str = "unknown"  # full | lora | qlora | unknown
    method_reason: str = ""
    lora: Dict[str, Evidence] = field(default_factory=dict)
    quantization: Dict[str, Evidence] = field(default_factory=dict)
    training_args: Dict[str, Evidence] = field(default_factory=dict)
    datasets: List[Evidence] = field(default_factory=list)
    cloud_signals: List[Evidence] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    files_scanned: int = 0

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_scripts": [s.to_dict() for s in self.training_scripts],
            "evaluation_scripts": list(self.evaluation_scripts),
            "inference_scripts": list(self.inference_scripts),
            "base_model": self.base_model.to_dict() if self.base_model else None,
            "parameter_count_billions": parse_param_count(
                str(self.base_model.value) if self.base_model else None
            ),
            "method": self.method,
            "method_reason": self.method_reason,
            "lora": {k: v.to_dict() for k, v in self.lora.items()},
            "quantization": {k: v.to_dict() for k, v in self.quantization.items()},
            "training_args": {k: v.to_dict() for k, v in self.training_args.items()},
            "datasets": [d.to_dict() for d in self.datasets],
            "cloud_signals": [c.to_dict() for c in self.cloud_signals],
            "libraries": list(self.libraries),
            "files_scanned": self.files_scanned,
        }

    @property
    def found_anything(self) -> bool:
        return bool(self.training_scripts or self.base_model or self.training_args or self.lora)


# ---------------------------------------------------------------------------
# Parameter counts from model names
# ---------------------------------------------------------------------------

_PARAM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z])")

#: Well-known models whose size is not in their name.
_KNOWN_SIZES = {
    "gpt2": 0.124, "gpt2-medium": 0.355, "gpt2-large": 0.774, "gpt2-xl": 1.5,
    "distilgpt2": 0.082, "bert-base": 0.11, "bert-large": 0.34,
}


def parse_param_count(model_id: Optional[str]) -> Optional[float]:
    """Return the parameter count in billions encoded in *model_id* (or ``None``).

    ``TinyLlama-1.1B`` -> 1.1, ``Llama-2-7b-hf`` -> 7.0, ``opt-350m`` -> 0.35,
    ``Mixtral-8x7B`` -> 56.0 (experts x size, an upper bound for total weights).
    The old substring table matched ``"7b"`` inside ``"2.7b"`` and ``"17b"``.
    """
    if not model_id:
        return None
    name = model_id.strip().split("/")[-1].lower()
    for key, size in _KNOWN_SIZES.items():
        if name == key or name.startswith(key + "-") and key == "gpt2":
            return size
    moe = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", name)
    if moe:
        return round(int(moe.group(1)) * float(moe.group(2)), 3)
    matches = _PARAM_RE.findall(name)
    if not matches:
        return None
    number, unit = matches[0]
    value = float(number)
    return round(value / 1000.0, 4) if unit.lower() == "m" else value


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _literal(node: ast.AST, constants: Dict[str, Any]) -> Tuple[bool, Any]:
    """Return ``(resolved, value)`` for a constant-like AST node."""
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        ok, value = _literal(node.operand, constants)
        if ok and isinstance(value, (int, float)):
            return True, -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Name) and node.id in constants:
        return True, constants[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        key = f"{node.value.id}.{node.attr}"
        if key in constants:
            return True, constants[key]
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for element in node.elts:
            ok, value = _literal(element, constants)
            if not ok:
                return False, None
            values.append(value)
        return True, values
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.Div, ast.Pow)):
        lok, left = _literal(node.left, constants)
        rok, right = _literal(node.right, constants)
        if lok and rok and isinstance(left, (int, float)) and isinstance(right, (int, float)):
            try:
                if isinstance(node.op, ast.Mult):
                    return True, left * right
                if isinstance(node.op, ast.Add):
                    return True, left + right
                if isinstance(node.op, ast.Sub):
                    return True, left - right
                if isinstance(node.op, ast.Div):
                    return True, left / right
                return True, left ** right
            except (ZeroDivisionError, OverflowError):
                return False, None
    return False, None


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _collect_constants(tree: ast.AST) -> Dict[str, Any]:
    """Module-level ``NAME = literal`` assignments plus argparse defaults.

    argparse defaults are stored as ``args.<dest>`` / ``<dest>`` so that
    ``num_train_epochs=args.epochs`` resolves to the declared default.
    """
    constants: Dict[str, Any] = {}

    def record(name: str, value_node: ast.AST) -> None:
        ok, value = _literal(value_node, constants)
        if ok:
            constants[name] = value

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    record(target.id, node.value)
                elif isinstance(target, ast.Attribute):
                    record(_dotted(target), node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                record(node.target.id, node.value)
        elif isinstance(node, ast.Call) and _call_name(node) == "add_argument":
            flags = [a.value for a in node.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            default: Optional[ast.AST] = None
            dest: Optional[str] = None
            for kw in node.keywords:
                if kw.arg == "default":
                    default = kw.value
                elif kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    dest = str(kw.value.value)
            if default is None or not flags:
                continue
            long_flag = next((f for f in flags if f.startswith("--")), flags[0])
            name = dest or long_flag.lstrip("-").replace("-", "_")
            ok, value = _literal(default, constants)
            if ok:
                constants[f"args.{name}"] = value
                constants.setdefault(name, value)
    return constants


def _keywords(call: ast.Call, constants: Dict[str, Any]) -> Dict[str, Tuple[Any, int]]:
    found: Dict[str, Tuple[Any, int]] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        ok, value = _literal(kw.value, constants)
        if ok:
            found[kw.arg] = (value, getattr(kw.value, "lineno", call.lineno))
    return found


def _looks_like_model_id(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        return False
    text = value.strip()
    if text.endswith((".json", ".jsonl", ".csv", ".txt", ".py", ".yaml", ".yml")):
        return False
    if "/" in text or "\\" in text:
        return bool(_MODEL_FAMILY_RE.search(text)) or text.count("/") == 1 and " " not in text
    return bool(_MODEL_FAMILY_RE.search(text)) and " " not in text


# ---------------------------------------------------------------------------
# The inspector
# ---------------------------------------------------------------------------

class CodeInspector:
    """Inspect a project directory's Python sources for training settings."""

    def __init__(self, project_path: str | Path):
        self.root = Path(project_path)

    # -- discovery -----------------------------------------------------
    def _python_files(self) -> List[Path]:
        files: List[Path] = []
        for path in sorted(self.root.rglob("*.py")):
            try:
                rel = path.relative_to(self.root)
            except ValueError:
                continue
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
            if len(files) >= MAX_PY_FILES:
                break
        return files

    # -- main entry ----------------------------------------------------
    def inspect(self) -> TrainingInspection:
        result = TrainingInspection()
        model_candidates: List[Tuple[int, Evidence]] = []
        libs: set[str] = set()
        has_lora = False
        has_4bit = False
        has_8bit = False

        for path in self._python_files():
            rel = path.relative_to(self.root).as_posix()
            try:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            result.files_scanned += 1

            reasons = [label for pattern, _w, label in _TRAIN_CONTENT if pattern.search(text)]
            score = sum(w for pattern, w, _l in _TRAIN_CONTENT if pattern.search(text))
            name_hit = bool(_TRAIN_NAME_RE.search(path.stem))
            if score >= 4 or (name_hit and score >= 2):
                result.training_scripts.append(ScriptInfo(rel, "training", reasons))
            elif name_hit and score >= 1:
                result.training_scripts.append(ScriptInfo(rel, "training", reasons + ["file name"]))
            elif _EVAL_NAME_RE.search(path.stem):
                result.evaluation_scripts.append(rel)
            elif _INFER_NAME_RE.search(path.stem):
                result.inference_scripts.append(rel)

            for pattern, label in _CLOUD_SIGNALS:
                match = pattern.search(text)
                if match:
                    line = text.count("\n", 0, match.start()) + 1
                    result.cloud_signals.append(Evidence(label, rel, line))

            for lib in ("transformers", "peft", "trl", "bitsandbytes", "accelerate",
                        "unsloth", "deepspeed", "axolotl", "torch", "datasets"):
                if re.search(rf"^\s*(?:from|import)\s+{lib}\b", text, re.M):
                    libs.add(lib)

            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                continue

            constants = _collect_constants(tree)
            file_score = score + (2 if name_hit else 0)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node)
                kws = _keywords(node, constants)

                if name in _TRAINING_ARG_CALLS or (name.endswith("TrainingArguments")):
                    for key in TRAINING_ARG_KEYS:
                        if key in kws and key not in result.training_args:
                            value, line = kws[key]
                            result.training_args[key] = Evidence(value, rel, line)
                    if "bf16" in kws and kws["bf16"][0] is True:
                        result.training_args["bf16"] = Evidence(True, rel, kws["bf16"][1])

                elif name == "LoraConfig":
                    has_lora = True
                    for key in _LORA_KEYS:
                        if key in kws and key not in result.lora:
                            value, line = kws[key]
                            result.lora[key] = Evidence(value, rel, line)

                elif name == "get_peft_model":
                    has_lora = True
                    for key in _LORA_KEYS:
                        if key in kws and key not in result.lora:
                            value, line = kws[key]
                            result.lora[key] = Evidence(value, rel, line)

                elif name == "BitsAndBytesConfig":
                    for key in ("load_in_4bit", "load_in_8bit", "bnb_4bit_quant_type",
                                "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype"):
                        if key in kws and key not in result.quantization:
                            result.quantization[key] = Evidence(kws[key][0], rel, kws[key][1])
                    if kws.get("load_in_4bit", (False, 0))[0] is True:
                        has_4bit = True
                    if kws.get("load_in_8bit", (False, 0))[0] is True:
                        has_8bit = True
                    for key in ("bnb_4bit_compute_dtype",):
                        for kw in node.keywords:
                            if kw.arg == key and key not in result.quantization:
                                result.quantization[key] = Evidence(_dotted(kw.value) or "unknown",
                                                                    rel, kw.value.lineno)

                elif name == "from_pretrained":
                    target = _dotted(node.func.value) if isinstance(node.func, ast.Attribute) else ""
                    arg_value: Any = None
                    arg_line = node.lineno
                    if node.args:
                        ok, arg_value = _literal(node.args[0], constants)
                        if not ok:
                            arg_value = None
                        arg_line = getattr(node.args[0], "lineno", node.lineno)
                    for key in ("pretrained_model_name_or_path", "model_name"):
                        if arg_value is None and key in kws:
                            arg_value, arg_line = kws[key]
                    if _looks_like_model_id(arg_value):
                        is_tokenizer = "Tokenizer" in target or "Processor" in target
                        is_model = ("Model" in target or "FastLanguageModel" in target
                                    or "PeftModel" in target or "ForCausalLM" in target)
                        priority = 3 if (is_model and not is_tokenizer) else (1 if is_tokenizer else 2)
                        model_candidates.append(
                            (priority * 10 + (5 if file_score >= 4 else 0),
                             Evidence(arg_value, rel, arg_line)))
                    for key in ("load_in_4bit", "load_in_8bit"):
                        if key in kws:
                            result.quantization.setdefault(key, Evidence(kws[key][0], rel, kws[key][1]))
                            if kws[key][0] is True:
                                has_4bit = has_4bit or key == "load_in_4bit"
                                has_8bit = has_8bit or key == "load_in_8bit"
                    if "FastLanguageModel" in target and "max_seq_length" in kws:
                        result.training_args.setdefault(
                            "max_seq_length", Evidence(kws["max_seq_length"][0], rel, kws["max_seq_length"][1]))

                elif name in ("SFTTrainer", "Trainer", "DPOTrainer"):
                    for key in ("max_seq_length", "max_length", "packing"):
                        if key in kws and key not in result.training_args:
                            result.training_args[key] = Evidence(kws[key][0], rel, kws[key][1])

                elif name == "load_dataset":
                    if node.args:
                        ok, value = _literal(node.args[0], constants)
                        if ok and isinstance(value, str):
                            extra = ""
                            data_files = kws.get("data_files")
                            if data_files:
                                extra = f" ({data_files[0]})"
                            result.datasets.append(Evidence(value + extra, rel, node.lineno))

            # Module/class level model name constants (MODEL_NAME = "...")
            for const_name, const_value in constants.items():
                if re.search(r"(^|\.|_)(model_?name|base_?model|model_?id|model_?path|checkpoint)$",
                             const_name, re.I) and _looks_like_model_id(const_value):
                    line = self._line_of_assignment(tree, const_name)
                    model_candidates.append((15 + (5 if file_score >= 4 else 0),
                                             Evidence(const_value, rel, line)))

            self._plain_loop_settings(tree, constants, rel, result, file_score)

        if model_candidates:
            model_candidates.sort(key=lambda item: -item[0])
            result.base_model = model_candidates[0][1]

        # Method decision --------------------------------------------------
        r_val = result.lora.get("r")
        if has_lora and has_4bit:
            result.method = "qlora"
            result.method_reason = "LoRA adapters on a 4-bit quantized base model"
        elif has_lora:
            result.method = "lora"
            result.method_reason = (
                "LoRA adapters on an 8-bit base model" if has_8bit else "LoRA adapters (LoraConfig / get_peft_model)"
            )
        elif result.training_scripts or result.training_args:
            result.method = "full"
            result.method_reason = "no PEFT adapter configuration found in the training code"
        if r_val is None and has_lora:
            result.lora.setdefault("r", Evidence("not set (library default 8)", "", 0))

        # De-duplicate dataset and cloud evidence.
        result.datasets = _dedupe(result.datasets)
        result.cloud_signals = _dedupe_by_value(result.cloud_signals)
        result.libraries = sorted(libs)
        result.training_scripts = _dedupe_scripts(result.training_scripts)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _line_of_assignment(tree: ast.AST, dotted: str) -> int:
        short = dotted.split(".")[-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name) and target.id == short) or (
                            isinstance(target, ast.Attribute) and _dotted(target) == dotted):
                        return node.lineno
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == short:
                return node.lineno
        return 0

    def _plain_loop_settings(self, tree: ast.AST, constants: Dict[str, Any], rel: str,
                             result: TrainingInspection, file_score: int) -> None:
        """Hand-written PyTorch loops: ``EPOCHS = 3``, ``LR = 2e-5`` and the like."""
        if file_score < 3:
            return
        aliases = {
            "num_train_epochs": ("epochs", "num_epochs", "n_epochs", "num_train_epochs", "max_epochs"),
            "learning_rate": ("learning_rate", "lr", "base_lr"),
            "per_device_train_batch_size": ("batch_size", "train_batch_size", "per_device_train_batch_size"),
            "max_seq_length": ("max_seq_length", "max_seq_len", "max_length", "block_size", "seq_len"),
            "gradient_accumulation_steps": ("gradient_accumulation_steps", "grad_accum", "accumulation_steps"),
        }
        for node in ast.walk(tree):
            targets: List[Tuple[str, ast.AST]] = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        targets.append((target.id, node.value))
                    elif isinstance(target, ast.Attribute):
                        targets.append((target.attr, node.value))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
                targets.append((node.target.id, node.value))
            for name, value_node in targets:
                lowered = name.lower()
                for canonical, names in aliases.items():
                    if lowered in names and canonical not in result.training_args:
                        ok, value = _literal(value_node, constants)
                        if ok and isinstance(value, (int, float)) and not isinstance(value, bool):
                            result.training_args[canonical] = Evidence(
                                value, rel, getattr(value_node, "lineno", node.lineno))


def _dedupe(items: Iterable[Evidence]) -> List[Evidence]:
    seen: set[Tuple[Any, str, int]] = set()
    out: List[Evidence] = []
    for item in items:
        key = (str(item.value), item.file, item.line)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _dedupe_by_value(items: Iterable[Evidence]) -> List[Evidence]:
    seen: set[str] = set()
    out: List[Evidence] = []
    for item in items:
        if str(item.value) not in seen:
            seen.add(str(item.value))
            out.append(item)
    return out


def _dedupe_scripts(items: Iterable[ScriptInfo]) -> List[ScriptInfo]:
    seen: set[str] = set()
    out: List[ScriptInfo] = []
    for item in items:
        if item.path not in seen:
            seen.add(item.path)
            out.append(item)
    return out


def inspect_project(project_path: str | Path) -> TrainingInspection:
    """Inspect *project_path*; never raises (inspection is best-effort)."""
    try:
        return CodeInspector(project_path).inspect()
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("Training-code inspection failed: %s", error)
        return TrainingInspection()


__all__ = [
    "CodeInspector",
    "Evidence",
    "ScriptInfo",
    "TrainingInspection",
    "inspect_project",
    "parse_param_count",
]
