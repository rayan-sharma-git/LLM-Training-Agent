"""Project scanner for discovering fine-tuning project structure."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .framework_detector import FrameworkDetector
from cleaning import dataset_io
from hardware.gpu_detector import detect_gpus

logger = logging.getLogger(__name__)

# Directories to always skip during scanning.
_SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".venv", "venv", "venv310", "env",
    ".venv310", ".env",
    ".vscode", ".idea",
    "dist", "build",
    ".DS_Store", "Thumbs.db",
}


class ProjectScanner:
    """Scan a project directory to discover fine-tuning related files and configuration."""

    def __init__(self, project_path: str):
        if not Path(project_path).exists():
            raise FileNotFoundError(f"Project path does not exist: {project_path}")
        self.project_path = Path(project_path)
        self.framework_detector = FrameworkDetector()

    def scan(self) -> Dict[str, Any]:
        """Scan the project and return structured summary."""
        logger.info(f"Scanning project: {self.project_path}")

        # Detect framework
        framework = self.framework_detector.detect(self.project_path)

        # Discover files
        datasets = self._discover_datasets()
        prompts = self._discover_prompts()
        configs = self._discover_configs()
        training_scripts = self._discover_scripts(["train*.py", "finetune*.py", "run_*.py"])
        eval_scripts = self._discover_scripts(["eval*.py", "evaluate*.py", "test*.py"])
        inference_scripts = self._discover_scripts(["inference*.py", "predict*.py", "serve*.py"])

        # Detect model
        base_model = self._detect_model()

        result = {
            "project_name": self.project_path.name,
            "project_path": str(self.project_path),
            **framework,
            "base_model": base_model,
            "dataset_paths": datasets,
            "prompt_templates": prompts,
            "configuration_files": configs,
            "training_scripts": training_scripts,
            "evaluation_scripts": eval_scripts,
            "inference_scripts": inference_scripts,
            "hardware_information": self._detect_hardware(),
            "project_statistics": self._compute_statistics(),
        }

        logger.info("Project scan complete")
        return result

    def _is_skipped(self, path: Path) -> bool:
        """Return True if *path* is inside a skipped directory."""
        try:
            rel = path.relative_to(self.project_path)
        except ValueError:
            return True
        for part in rel.parts:
            if part in _SKIP_DIRS:
                return True
            if part.startswith(".") and part not in _SKIP_DIRS:
                # Skip dot-directories like .venv, .mypy_cache, etc.
                return True
            if part.startswith("._"):
                return True
        return False

    def _discover_datasets(self) -> List[str]:
        """Discover dataset files (internal).

        Discovery is deliberately permissive — the extension patterns
        (``*.json``/``*.txt``) match almost anything, and a real dataset with an
        unusual name must never be silently hidden from the analysis. Only
        filenames that cannot be training data in any project are excluded, so
        ``requirements.txt`` is no longer analysed and rewritten as if it were
        data. Anything else stays in, and is reported honestly by the analyzers
        when it turns out to hold no readable records.
        """
        patterns = ["*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt", "*.tsv"]
        files = []
        for pattern in patterns:
            for f in self.project_path.rglob(pattern):
                if f.is_file() and not self._is_skipped(f) and not self._is_non_dataset(f):
                    files.append(str(f.relative_to(self.project_path)))
        return sorted(set(files))

    #: Filenames that are never training data in any project.
    _NEVER_DATASETS = frozenset({
        "requirements.txt",
        "readme.txt",
        "license.txt",
        "licence.txt",
        "notice.txt",
        "changelog.txt",
        "authors.txt",
        "contributing.txt",
        "codeowners",
        ".gitignore",
        ".gitattributes",
    })

    def _is_non_dataset(self, path: Path) -> bool:
        """True when *path* is a well-known non-dataset file."""
        return path.name.lower() in self._NEVER_DATASETS

    def discover_datasets(self) -> List[str]:
        """Public accessor for dataset discovery.

        Returns project-relative dataset paths using exactly the same rules as
        :meth:`scan`, without probing hardware or model configuration.  Used by
        the chunked dataset cleaning endpoint.
        """
        return self._discover_datasets()

    def _discover_prompts(self) -> List[str]:
        """Discover prompt templates."""
        patterns = ["prompt*.txt", "prompt*.md", "templates/**/*.txt", "templates/**/*.md"]
        files = []
        for pattern in patterns:
            for f in self.project_path.rglob(pattern):
                if f.is_file() and not self._is_skipped(f):
                    files.append(str(f.relative_to(self.project_path)))
        return sorted(set(files))

    def _discover_configs(self) -> List[str]:
        """Discover configuration files.

        JSON is a standard training-config format (HuggingFace
        ``TrainingArguments`` dumps, ``training_config.json``), and omitting it
        meant a project whose only config was JSON reported "Config files: none
        found" and had no hyperparameters extracted at all.

        Only JSON files whose *name* looks like a configuration are taken. Every
        ``*.json`` would also match the project's datasets, and listing a
        dataset as a config file is misleading in the report and makes the
        hyperparameter analyzer read data files as settings.
        """
        patterns = ["*.yaml", "*.yml", "*.toml", "config*.py"]

        files: List[str] = []
        for pattern in patterns:
            for f in self.project_path.rglob(pattern):
                if f.is_file() and not self._is_skipped(f):
                    files.append(str(f.relative_to(self.project_path)))

        config_words = ("config", "arg", "param", "setting", "hyper", "train")
        for f in self.project_path.rglob("*.json"):
            if not f.is_file() or self._is_skipped(f):
                continue
            if any(word in f.stem.lower() for word in config_words):
                files.append(str(f.relative_to(self.project_path)))

        return sorted(set(files))

    def _discover_scripts(self, patterns: List[str]) -> List[str]:
        """Discover scripts matching patterns."""
        files = []
        for pattern in patterns:
            for f in self.project_path.rglob(pattern):
                if f.is_file() and not self._is_skipped(f):
                    files.append(str(f.relative_to(self.project_path)))
        return sorted(set(files))

    def _detect_model(self) -> Optional[str]:
        """Attempt to detect the base model from training configs.

        Returns the actual model name found in ``model_name_or_path`` (not a
        placeholder string) so downstream analyzers can resolve it.
        """
        import re as _re

        for config in self.project_path.rglob("*.y*ml"):
            if self._is_skipped(config):
                continue
            try:
                text = dataset_io.read_text(config)
            except (OSError, PermissionError, ValueError):
                continue
            m = _re.search(r"model_name_or_path\s*[:=]\s*[\"']?([\w./\\-]+)", text)
            if m:
                return m.group(1)
        # Also check JSON configs (e.g. HF trainer configs)
        for config in self.project_path.rglob("*.json"):
            if self._is_skipped(config):
                continue
            try:
                data = json.loads(dataset_io.read_text(config))
            except Exception:
                continue
            if isinstance(data, dict):
                value = data.get("model_name_or_path") or data.get("_name_or_path")
                if isinstance(value, str) and value:
                    return value
        return None

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware (real GPU detection)."""
        try:
            hardware = detect_gpus()
            hw_dict = hardware.to_dict()
            hw_dict["gpu_available"] = hardware.gpu_count > 0
            return hw_dict
        except Exception as e:
            logger.warning(f"GPU detection failed during scan: {e}")
            return {"gpu_available": False, "cpu_count": 4, "error": str(e)}

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute project statistics."""
        file_count = 0
        for f in self.project_path.rglob("*"):
            if f.is_file() and not self._is_skipped(f):
                file_count += 1
        return {"file_count": file_count}
