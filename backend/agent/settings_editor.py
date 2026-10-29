"""Find a training setting in a project and change it with a minimal edit.

"Change the epochs from 3 to 5" has to work in projects that name the setting
``num_train_epochs``, ``EPOCHS``, ``--epochs`` or ``training.epochs`` and keep
it in Python, YAML, JSON, TOML, INI or a shell script.  The editor therefore:

1. maps the request to a *concept* (epochs, learning rate, batch size ...) with
   the identifiers projects commonly use for it;
2. finds every literal that holds that concept - Python through the AST (keyword
   arguments, module constants, ``argparse`` defaults, dict entries, dataclass
   fields; a keyword that points at a constant leads to the constant), other
   formats through key matching;
3. picks the authoritative site (training script > config > other; exact old
   value match first) and asks the user when two plausible sites remain;
4. replaces **only that literal**, re-parses the result and refuses to produce
   a file that no longer parses or whose value is not the requested one.

Nothing is written here: the result is a new file text handed to the change
store, which owns approval, conflict detection, backups and revert.
"""
from __future__ import annotations

import ast
import configparser
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Concept:
    key: str
    label: str
    aliases: Tuple[str, ...]
    kind: str                         # "int" | "float" | "str"
    phrase: str                       # regex over natural language
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_min: bool = False
    #: alias -> callee names it is only valid inside (e.g. LoRA's bare ``r``)
    scoped: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


CONCEPTS: Tuple[Concept, ...] = (
    Concept("gradient_accumulation", "gradient accumulation steps",
            ("gradient_accumulation_steps", "gradient_accumulation", "grad_accum", "grad_accum_steps",
             "accumulate_grad_batches", "accumulation_steps", "grad_accumulation_steps"),
            "int", r"gradient[\s_-]*accumulation(?:[\s_-]*steps)?|grad[\s_-]*accum(?:ulation)?(?:[\s_-]*steps)?",
            minimum=1),
    Concept("learning_rate", "learning rate",
            ("learning_rate", "lr", "base_lr", "peak_lr", "init_lr", "learning_rate_init"),
            "float", r"learning[\s_-]*rate|\blr\b", minimum=0, maximum=1, exclusive_min=True),
    Concept("batch_size", "batch size",
            ("per_device_train_batch_size", "train_batch_size", "batch_size", "micro_batch_size",
             "train_micro_batch_size_per_gpu", "per_gpu_train_batch_size"),
            "int", r"(?:train(?:ing)?[\s_-]*)?batch[\s_-]*size", minimum=1),
    Concept("max_seq_length", "max sequence length",
            ("max_seq_length", "max_seq_len", "sequence_length", "seq_length", "max_length",
             "block_size", "cutoff_len", "model_max_length", "max_sequence_length"),
            "int", r"(?:max(?:imum)?[\s_-]*)?(?:sequence|seq|context)[\s_-]*(?:length|len)|max[\s_-]*length|cutoff[\s_-]*len",
            minimum=1, maximum=10_000_000),
    Concept("lora_alpha", "LoRA alpha", ("lora_alpha",), "float", r"lora[\s_-]*alpha|\balpha\b", minimum=0,
            exclusive_min=True),
    Concept("lora_dropout", "LoRA dropout", ("lora_dropout",), "float", r"lora[\s_-]*dropout|\bdropout\b",
            minimum=0, maximum=1),
    Concept("lora_r", "LoRA rank", ("lora_r", "lora_rank"), "int", r"lora[\s_-]*(?:rank|r)\b|\brank\b",
            minimum=1, maximum=4096, scoped=(("r", ("LoraConfig", "AdaLoraConfig")),)),
    Concept("warmup_ratio", "warmup ratio", ("warmup_ratio", "warm_up_ratio"), "float",
            r"warm[\s_-]*up[\s_-]*ratio", minimum=0, maximum=1),
    Concept("warmup_steps", "warmup steps", ("warmup_steps", "num_warmup_steps"), "int",
            r"warm[\s_-]*up[\s_-]*steps?", minimum=0),
    Concept("weight_decay", "weight decay", ("weight_decay", "wd"), "float", r"weight[\s_-]*decay", minimum=0),
    Concept("max_steps", "max steps", ("max_steps", "num_train_steps", "total_steps", "train_steps"), "int",
            r"max(?:imum)?[\s_-]*steps?|total[\s_-]*steps", minimum=1),
    Concept("epochs", "epochs",
            ("num_train_epochs", "epochs", "n_epochs", "num_epochs", "max_epochs", "epoch", "train_epochs",
             "total_epochs", "nb_epochs"),
            "float", r"(?:number[\s_-]*of[\s_-]*)?(?:training[\s_-]*|train[\s_-]*)?epochs?", minimum=0,
            exclusive_min=True),
    Concept("seed", "random seed", ("seed", "random_seed", "random_state"), "int", r"(?:random[\s_-]*)?seed",
            minimum=0, maximum=2 ** 63),
    Concept("lr_scheduler", "LR scheduler", ("lr_scheduler_type", "scheduler", "lr_scheduler", "scheduler_type"),
            "str", r"(?:lr[\s_-]*|learning[\s_-]*rate[\s_-]*)?scheduler(?:[\s_-]*type)?"),
    Concept("save_steps", "save steps", ("save_steps", "checkpoint_steps"), "int", r"save[\s_-]*steps?",
            minimum=1),
    Concept("logging_steps", "logging steps", ("logging_steps", "log_steps", "log_every_n_steps"), "int",
            r"logging[\s_-]*steps?|log[\s_-]*steps?", minimum=1),
    Concept("eval_steps", "evaluation steps", ("eval_steps", "evaluation_steps"), "int",
            r"eval(?:uation)?[\s_-]*steps?", minimum=1),
    Concept("base_model", "base model",
            ("model_name_or_path", "model_name", "base_model", "base_model_name", "model_id", "model_path",
             "pretrained_model_name_or_path", "pretrained_model", "model"),
            "str", r"(?:base[\s_-]*)?model(?:[\s_-]*(?:name|id|path))?"),
)
_BY_KEY = {c.key: c for c in CONCEPTS}


def concept_by_key(key: str) -> Optional[Concept]:
    return _BY_KEY.get(key)


def _norm(name: str) -> str:
    return re.sub(r"[-\s]+", "_", name.strip().lstrip("-")).lower()


# ---------------------------------------------------------------------------
# Natural-language request
# ---------------------------------------------------------------------------
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_QUESTION = re.compile(r"^\s*(what|why|how|when|where|which|who|should|is|are|does|do|did|can|could|would|will|"
                       r"explain|tell me|describe)\b", re.I)
_VERB = re.compile(r"\b(change|set|update|make|switch|modify|adjust|edit|put|use|using|bump|raise|lower|"
                   r"increase|decrease|reduce|decrease|double|halve|triple|train(?:ing)? for|run for|"
                   r"go with|replace)\b", re.I)


@dataclass
class SettingChange:
    concept: Concept
    op: str                 # "set" | "add" | "mul"
    operand: str            # literal text of the new value / delta / factor
    old: Optional[str] = None
    file_hint: Optional[str] = None   # "... in config.yaml"


_PHRASE_ORDER = [c for c in CONCEPTS]


def parse_setting_change(text: str) -> Optional[SettingChange]:
    """Read "change epochs from 3 to 5" style requests; ``None`` if this is not one."""
    change = _parse_setting_change(text)
    if change is not None:
        hint = re.search(r"\b(?:in|inside|within)\s+([\w./\\-]+\.\w+)\s*$", (text or "").strip())
        if hint:
            change.file_hint = hint.group(1).replace("\\", "/")
    return change


def _parse_setting_change(text: str) -> Optional[SettingChange]:
    raw = (text or "").strip()
    if not raw or len(raw) > 600:
        return None
    if _QUESTION.match(raw) and raw.rstrip().endswith("?"):
        return None

    # Strip a trailing "in <file>" so file names do not look like values.
    body = re.sub(r"\s+(?:in|inside|within)\s+[\w./\\-]+\.\w+\s*$", "", raw)

    concept: Optional[Concept] = None
    match_span: Tuple[int, int] = (0, 0)
    for candidate in _PHRASE_ORDER:
        found = re.search(rf"(?<![\w-])(?:{candidate.phrase})(?![\w-])", body, re.I)
        if found:
            concept, match_span = candidate, found.span()
            break
    if concept is None:
        return None
    after = body[match_span[1]:]
    before = body[:match_span[0]]
    val = _NUM if concept.kind != "str" else r"[A-Za-z0-9_./:@-]+"

    # relative: "double the batch size", "increase epochs by 2"
    if re.search(r"\bdouble\b", body, re.I):
        return SettingChange(concept, "mul", "2")
    if re.search(r"\bhalve\b|\bhalf\b", body, re.I) and concept.kind != "str":
        return SettingChange(concept, "mul", "0.5")
    if re.search(r"\btriple\b", body, re.I):
        return SettingChange(concept, "mul", "3")
    rel = re.search(rf"\b(increase|raise|bump|add|decrease|reduce|lower|decrement|increment)\b.*?\bby\s+({_NUM})", body, re.I)
    if rel and concept.kind != "str":
        sign = -1 if rel.group(1) in ("decrease", "reduce", "lower", "decrement") else 1
        delta = float(rel.group(2)) * sign
        return SettingChange(concept, "add", str(int(delta)) if delta == int(delta) else repr(delta))

    m = re.search(rf"\bfrom\s+({val})\s+to\s+({val})", body, re.I)
    if m:
        return SettingChange(concept, "set", m.group(2), old=m.group(1))
    m = re.search(rf"^\s*(?:the\s+|a\s+|an\s+)?(?:to|=|:|as|at|of|->|→)\s*[\"'`]?({val})[\"'`]?(?:\s|$|[.,!])", after, re.I)
    if m:
        return SettingChange(concept, "set", m.group(1))
    m = re.search(rf"(?:\bto\b|=|->|→)\s*[\"'`]?({val})[\"'`]?(?:\s|$|[.,!])", after, re.I)
    if m and _VERB.search(body):
        return SettingChange(concept, "set", m.group(1))
    if concept.kind != "str" and _VERB.search(body):
        m = re.match(rf"\s*({_NUM})(?:\s|$|[.,!])", after)
        if m:  # "make the seed 7"
            return SettingChange(concept, "set", m.group(1))
    if concept.kind != "str":
        m = re.search(rf"({_NUM})\s*(?:[\w-]+\s+){{0,2}}$", before)
        if m and _VERB.search(body):  # "train for 5 epochs", "use 8 as batch size" handled below
            return SettingChange(concept, "set", m.group(1))
        m = re.search(rf"\b({_NUM})\s+as\b", before)
        if m and _VERB.search(body):
            return SettingChange(concept, "set", m.group(1))
    return None


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------
@dataclass
class Site:
    file: str
    line: int
    kind: str
    key: str
    literal: str
    start: int
    end: int
    value: Any
    score: int = 0
    via: Optional[str] = None
    quote: str = ""

    def where(self) -> str:
        return f"{self.file}:{self.line}"


SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox", ".mypy_cache",
             ".pytest_cache", ".idea", ".vscode", ".llm_training_agent", ".llm-training-agent",
             ".llm_training_agent_backups", "site-packages", "dist", "build", ".ipynb_checkpoints", "wandb",
             "outputs", "checkpoints", "runs", "lightning_logs"}
TEXT_EXTS = {".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".sh", ".bash", ".bat", ".cmd", ".ps1"}
SPECIAL_NAMES = {"makefile", "dockerfile"}
MAX_FILE_BYTES = 512 * 1024
MAX_FILES = 600
_DEMOTED_DIRS = {"tests", "test", "examples", "example", "docs", "doc", "old", "legacy", "archive", "backup"}


def iter_project_files(root: Path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() in TEXT_EXTS or name.lower() in SPECIAL_NAMES:
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                count += 1
                if count > MAX_FILES:
                    return
                yield path


def read_text(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _line_starts(text: str) -> List[int]:
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())
    return starts


def _char_offset(text: str, starts: List[int], lineno: int, col_bytes: int) -> int:
    line_start = starts[lineno - 1]
    end = starts[lineno] if lineno < len(starts) else len(text)
    line = text[line_start:end]
    return line_start + len(line.encode("utf-8")[:col_bytes].decode("utf-8", errors="ignore"))


def _numeric(literal: str) -> Optional[float]:
    try:
        return float(literal)
    except (TypeError, ValueError):
        return None


def _matches(concept: Concept, name: str, callee: Optional[str] = None) -> bool:
    n = _norm(name)
    if n in concept.aliases:
        return True
    for alias, callees in concept.scoped:
        if n == alias and callee in callees:
            return True
    return False


def _callee_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _const_value(node: ast.AST) -> Tuple[bool, Any]:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) and \
            isinstance(node.value, (int, float, str)):
        return True, node.value
    return False, None


def _python_sites(rel: str, text: str, concept: Concept) -> List[Site]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    starts = _line_starts(text)

    def span(node: ast.AST) -> Tuple[int, int]:
        return (_char_offset(text, starts, node.lineno, node.col_offset),
                _char_offset(text, starts, node.end_lineno, node.end_col_offset))

    def make(node: ast.AST, kind: str, key: str, via: Optional[str] = None, score: int = 0) -> Optional[Site]:
        ok, value = _const_value(node)
        if not ok:
            return None
        if concept.kind == "str" and not isinstance(value, str):
            return None
        if concept.kind != "str" and isinstance(value, str):
            return None
        a, b = span(node)
        literal = text[a:b]
        quote = literal[0] if isinstance(value, str) and literal[:1] in "\"'" else ""
        return Site(rel, node.lineno, kind, key, literal, a, b, value, score, via, quote)

    # module-level constants a keyword may point at
    constants: Dict[str, ast.Assign] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if _const_value(stmt.value)[0]:
                constants[stmt.targets[0].id] = stmt
    sites: List[Site] = []
    seen: set = set()

    def add(site: Optional[Site]) -> None:
        if site and (site.start, site.end) not in seen:
            seen.add((site.start, site.end))
            sites.append(site)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = target.id if isinstance(target, ast.Name) else (
                    target.attr if isinstance(target, ast.Attribute) else None)
                if name and _matches(concept, name):
                    toplevel = node in tree.body
                    add(make(node.value, "assign", name, score=2 if toplevel else 1))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            t = node.target
            name = t.id if isinstance(t, ast.Name) else (t.attr if isinstance(t, ast.Attribute) else None)
            if name and _matches(concept, name):
                add(make(node.value, "field", name, score=2))
        elif isinstance(node, ast.Call):
            callee = _callee_name(node)
            if callee == "add_argument" and node.args:
                flag = node.args[0]
                names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and a.value.startswith("-")]
                if any(_matches(concept, n) for n in names):
                    for kw in node.keywords:
                        if kw.arg == "default":
                            add(make(kw.value, "argparse_default", names[0], score=3))
                continue
            for kw in node.keywords:
                if kw.arg and _matches(concept, kw.arg, callee):
                    if isinstance(kw.value, ast.Name) and kw.value.id in constants:
                        stmt = constants[kw.value.id]
                        site = make(stmt.value, "assign", kw.value.id, via=f"{kw.arg}={kw.value.id}", score=5)
                        add(site)
                    else:
                        add(make(kw.value, "kwarg", kw.arg, score=4))
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and _matches(concept, k.value):
                    add(make(v, "dict", k.value, score=2))
    return sites


_YAML_LINE = re.compile(r"^(?P<pre>[ \t]*(?:-[ \t]+)?(?P<key>[A-Za-z0-9_.\-]+)[ \t]*:[ \t]*)"
                        r"(?P<val>\"[^\"\n]*\"|'[^'\n]*'|[^\s#\[\]{}|>&*!\"'][^#\n]*?)(?P<tail>[ \t]*(?:#.*)?)\r?$")
_TOML_LINE = re.compile(r"^(?P<pre>[ \t]*(?P<key>[A-Za-z0-9_.\-]+)[ \t]*=[ \t]*)"
                        r"(?P<val>\"[^\"\n]*\"|'[^'\n]*'|[^\s#\[\]{}\"'][^#\n]*?)(?P<tail>[ \t]*(?:#.*)?)\r?$")
_INI_LINE = re.compile(r"^(?P<pre>[ \t]*(?P<key>[A-Za-z0-9_.\-]+)[ \t]*[=:][ \t]*)"
                       r"(?P<val>[^\s;#][^;#\n]*?)(?P<tail>[ \t]*(?:[;#].*)?)\r?$")
_JSON_PAIR = re.compile(r"\"(?P<key>[^\"\\\n]+)\"\s*:\s*(?P<val>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\"[^\"\\\n]*\")")
_CLI_FLAG = re.compile(r"(?<![\w-])--(?P<key>[A-Za-z0-9][A-Za-z0-9_-]*)(?:=|[ \t]+)(?P<val>[\"']?[A-Za-z0-9_./:@+-]+[\"']?)")


def _scalar(concept: Concept, raw: str) -> Optional[Tuple[Any, str]]:
    raw = raw.strip()
    if concept.kind == "str":
        if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
            return raw[1:-1], raw[0]
        return (raw, "") if raw and not _numeric(raw) is not None else None
    number = _numeric(raw)
    if number is None or raw.lower() in ("nan", "inf", "-inf"):
        return None
    return (int(number) if re.fullmatch(r"-?\d+", raw) else number), ""


def _line_sites(rel: str, text: str, concept: Concept, pattern: "re.Pattern[str]", kind: str,
                key_matches: bool = True) -> List[Site]:
    sites: List[Site] = []
    offset = 0
    for lineno, line in enumerate(text.split("\n"), 1):
        m = pattern.match(line)
        if m and _matches(concept, m.group("key").split(".")[-1]):
            parsed = _scalar(concept, m.group("val"))
            if parsed is not None:
                value, quote = parsed
                start = offset + m.start("val")
                sites.append(Site(rel, lineno, kind, m.group("key"), m.group("val"), start, offset + m.end("val"),
                                  value, 2, None, quote))
        offset += len(line) + 1
    return sites


def _json_sites(rel: str, text: str, concept: Concept) -> List[Site]:
    sites: List[Site] = []
    for m in _JSON_PAIR.finditer(text):
        if not _matches(concept, m.group("key")):
            continue
        parsed = _scalar(concept, m.group("val"))
        if parsed is None:
            continue
        value, quote = parsed
        sites.append(Site(rel, text.count("\n", 0, m.start("val")) + 1, "json", m.group("key"), m.group("val"),
                          m.start("val"), m.end("val"), value, 2, None, quote))
    return sites


def _cli_sites(rel: str, text: str, concept: Concept) -> List[Site]:
    sites: List[Site] = []
    for m in _CLI_FLAG.finditer(text):
        if not _matches(concept, m.group("key")):
            continue
        parsed = _scalar(concept, m.group("val"))
        if parsed is None:
            continue
        value, quote = parsed
        sites.append(Site(rel, text.count("\n", 0, m.start("val")) + 1, "cli", "--" + m.group("key"),
                          m.group("val"), m.start("val"), m.end("val"), value, 3, None, quote))
    return sites


def find_sites(project_root: Path, concept: Concept, training_scripts: Optional[List[str]] = None,
               referenced_configs: Optional[List[str]] = None) -> List[Site]:
    """Every literal in the project that holds *concept*, scored by how authoritative it is."""
    scripts = {s.replace("\\", "/") for s in (training_scripts or [])}
    configs = {c.replace("\\", "/") for c in (referenced_configs or [])}
    out: List[Site] = []
    root = Path(project_root)
    for path in iter_project_files(root):
        text = read_text(path)
        if text is None:
            continue
        rel = path.relative_to(root).as_posix()
        ext = path.suffix.lower()
        if ext == ".py":
            sites = _python_sites(rel, text, concept)
        elif ext in (".yaml", ".yml"):
            sites = _line_sites(rel, text, concept, _YAML_LINE, "yaml") + _cli_sites(rel, text, concept)
        elif ext == ".json":
            sites = _json_sites(rel, text, concept)
        elif ext == ".toml":
            sites = _line_sites(rel, text, concept, _TOML_LINE, "toml")
        elif ext in (".ini", ".cfg"):
            sites = _line_sites(rel, text, concept, _INI_LINE, "ini")
        else:
            sites = _cli_sites(rel, text, concept)
        parts = set(Path(rel).parts[:-1])
        for site in sites:
            if rel in scripts:
                site.score += 2
            if rel in configs:
                site.score += 2
            if parts & _DEMOTED_DIRS:
                site.score -= 3
            if "/" not in rel and ext in (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"):
                site.score += 1
            out.append(site)
    out.sort(key=lambda s: (-s.score, s.file, s.line))
    return out


# ---------------------------------------------------------------------------
# Planning an edit
# ---------------------------------------------------------------------------
@dataclass
class EditPlan:
    status: str                       # ready | ambiguous | not_found | mismatch | invalid | unchanged
    message: str
    concept: Optional[Concept] = None
    site: Optional[Site] = None
    candidates: List[Site] = field(default_factory=list)
    new_text: Optional[str] = None
    old_literal: str = ""
    new_literal: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def _format_number(concept: Concept, value: float, template: str, user_literal: Optional[str]) -> str:
    """Render *value* the way the file already writes numbers (and the user wrote it)."""
    if user_literal is not None and _numeric(user_literal) is not None and float(user_literal) == value:
        text = user_literal
        if concept.kind == "int" and re.fullmatch(r"-?\d+(\.0+)?", text):
            return str(int(value))
        return text
    if concept.kind == "int" or (float(value).is_integer() and re.fullmatch(r"-?\d+", template.strip() or "0")):
        return str(int(value))
    if "e" in template.lower() and abs(value) < 1e-2:
        mantissa, exp = f"{value:e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exp)}"
    return repr(float(value)) if not float(value).is_integer() else str(float(value))


def _validate_value(concept: Concept, value: float) -> Optional[str]:
    if concept.kind == "int" and not float(value).is_integer():
        return f"{concept.label.capitalize()} must be a whole number."
    if concept.minimum is not None:
        if value < concept.minimum or (concept.exclusive_min and value == concept.minimum):
            cmp = ">" if concept.exclusive_min else "≥"
            return f"{concept.label.capitalize()} must be {cmp} {concept.minimum:g}."
    if concept.maximum is not None and value > concept.maximum:
        return f"{concept.label.capitalize()} must be ≤ {concept.maximum:g}."
    return None


def _equal(site: Site, literal: str, concept: Concept) -> bool:
    if concept.kind == "str":
        return str(site.value) == literal
    a, b = _numeric(str(site.value)), _numeric(literal)
    return a is not None and b is not None and abs(a - b) <= 1e-12 * max(1.0, abs(a))


def plan_edit(project_root: Path, change: SettingChange, training_scripts: Optional[List[str]] = None,
              referenced_configs: Optional[List[str]] = None, only: Optional[Tuple[str, int]] = None) -> EditPlan:
    """Locate the setting and build the edited file text (nothing is written).

    ``only=(file, line)`` restricts the choice to one site - used after the
    user picked a candidate from an "ambiguous" card.
    """
    concept = change.concept
    sites = find_sites(project_root, concept, training_scripts, referenced_configs)
    if only:
        sites = [s for s in sites if s.file == only[0] and s.line == only[1]]
    if change.file_hint:
        hinted = [s for s in sites if s.file == change.file_hint or s.file.endswith("/" + change.file_hint)]
        if not hinted and sites:
            return EditPlan("not_found", f"I found no {concept.label} setting in {change.file_hint}.", concept,
                            candidates=sites[:4])
        sites = hinted
    if not sites:
        return EditPlan("not_found", f"I could not find a {concept.label} setting with a literal value in this "
                        "project's Python, YAML, JSON, TOML, INI or shell files. If it is computed or read from "
                        "somewhere else, tell me which file to edit.", concept)

    pool = sites
    if change.old is not None:
        matching = [s for s in sites if _equal(s, change.old, concept)]
        if not matching:
            found = ", ".join(f"{s.where()} = {s.literal}" for s in sites[:4])
            return EditPlan("mismatch", f"You said {concept.label} is {change.old}, but I found {found}. "
                            f"Say \"set {concept.label} to …\" if you want to change that value anyway.",
                            concept, candidates=sites[:4])
        pool = matching

    top = pool[0].score
    best = [s for s in pool if s.score == top]
    # several sites that all hold the same value in different places: ask
    distinct_places = {(s.file) for s in best}
    if len(best) > 1 and len(distinct_places) > 1 and not only:
        return EditPlan("ambiguous", f"I found {len(best)} places that set the {concept.label}. "
                        "Which one should I change?", concept, candidates=best[:4])
    site = best[0]

    # compute the new literal
    current = site.value
    if change.op == "set":
        user_literal = change.operand
        if concept.kind == "str":
            new_value: Any = change.operand
        else:
            number = _numeric(change.operand)
            if number is None:
                return EditPlan("invalid", f"'{change.operand}' is not a valid number.", concept)
            new_value = number
    else:
        base = _numeric(str(current))
        operand = _numeric(change.operand)
        if base is None or operand is None:
            return EditPlan("invalid", f"I cannot compute a relative change on {concept.label}.", concept)
        new_value = base * operand if change.op == "mul" else base + operand
        user_literal = None
        if concept.kind == "int":
            new_value = int(round(new_value))
    if concept.kind != "str":
        problem = _validate_value(concept, float(new_value))
        if problem:
            return EditPlan("invalid", problem, concept, site=site)

    if concept.kind == "str":
        if not re.fullmatch(r"[A-Za-z0-9_./:@+\- ]{1,200}", str(new_value)):
            return EditPlan("invalid", "That value contains characters I will not write into a config.", concept)
        q = site.quote
        new_literal = f"{q}{new_value}{q}" if q else str(new_value)
        if site.kind in ("assign", "kwarg", "argparse_default", "dict", "field") and not q:
            new_literal = f'"{new_value}"'
    else:
        new_literal = _format_number(concept, float(new_value), site.literal, user_literal)
    if site.literal == new_literal or (concept.kind != "str" and _equal(site, new_literal, concept)):
        return EditPlan("unchanged", f"The {concept.label} is already {site.literal} in {site.where()}.",
                        concept, site=site, old_literal=site.literal, new_literal=new_literal)

    path = Path(project_root) / site.file
    text = read_text(path)
    if text is None or text[site.start:site.end] != site.literal:
        return EditPlan("invalid", f"{site.file} changed while I was reading it. Please try again.", concept)
    new_text = text[:site.start] + new_literal + text[site.end:]

    problem = validate_edited(site.file, new_text, concept, site, new_literal)
    if problem:
        return EditPlan("invalid", f"The edit would break {site.file}: {problem} Nothing was changed.", concept,
                        site=site)
    return EditPlan("ready", f"{concept.label.capitalize()}: {site.literal} → {new_literal} in {site.where()}",
                    concept, site=site, new_text=new_text, old_literal=site.literal, new_literal=new_literal,
                    candidates=best[:4])


def validate_edited(file: str, new_text: str, concept: Concept, site: Site, new_literal: str) -> Optional[str]:
    """``None`` when the edited file still parses and holds the requested value."""
    ext = Path(file).suffix.lower()
    try:
        if ext == ".py":
            tree = ast.parse(new_text)
            compile(tree, file, "exec")
            starts = _line_starts(new_text)
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and getattr(node, "lineno", None) == site.line:
                    a = _char_offset(new_text, starts, node.lineno, node.col_offset)
                    b = _char_offset(new_text, starts, node.end_lineno, node.end_col_offset)
                    if a == site.start and new_text[a:b] == new_literal:
                        found = True
                        break
            if not found:
                return "the new value was not found where it should be."
        elif ext == ".json":
            json.loads(new_text)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
                list(yaml.safe_load_all(new_text))
            except ImportError:  # pragma: no cover - PyYAML is a backend dependency
                pass
        elif ext == ".toml":
            try:
                import tomllib  # type: ignore
            except ImportError:  # pragma: no cover - Python < 3.11
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    tomllib = None
            if tomllib is not None:
                tomllib.loads(new_text)
        elif ext in (".ini", ".cfg"):
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            parser.read_string(new_text)
    except Exception as error:  # noqa: BLE001 - any parser failure means "do not write"
        return f"it no longer parses ({type(error).__name__})."
    if new_text[site.start:site.start + len(new_literal)] != new_literal:
        return "the new value was not written at the expected position."
    return None


__all__ = [
    "CONCEPTS", "Concept", "SettingChange", "Site", "EditPlan",
    "parse_setting_change", "find_sites", "plan_edit", "concept_by_key",
]
