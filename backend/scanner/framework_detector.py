"""Framework detection logic."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


class FrameworkDetector:
    """Detect fine-tuning frameworks in a project."""

    FRAMEWORK_INDICATORS = {
        "huggingface": ["transformers", "datasets", "peft", "trl", "accelerate"],
        "axolotl": ["axolotl", "train"],
        "unsloth": ["unsloth"],
    }

    def detect(self, project_path: Path) -> Dict[str, Optional[str]]:
        """Detect framework used in the project."""
        project_path = Path(project_path)

        # Check requirements files
        req_files = ["requirements.txt", "pyproject.toml", "setup.py"]
        dependencies: List[str] = []

        for req_file in req_files:
            req_path = project_path / req_file
            if req_path.exists():
                dependencies.extend(self._extract_dependencies(req_path))

        # Check for framework indicators
        detected_framework = "unknown"
        detected_version = None

        for framework, indicators in self.FRAMEWORK_INDICATORS.items():
            for indicator in indicators:
                if any(indicator in dep for dep in dependencies):
                    detected_framework = framework
                    break

        return {
            "framework": detected_framework,
            "framework_version": detected_version,
        }

    def _extract_dependencies(self, file_path: Path) -> List[str]:
        """Extract dependencies from requirements file."""
        deps = []
        try:
            if file_path.name == "requirements.txt":
                deps = file_path.read_text(encoding="utf-8").splitlines()
            elif file_path.name == "pyproject.toml":
                deps = self._extract_pyproject_deps(file_path)
            elif file_path.name == "setup.py":
                deps = self._extract_setup_py_deps(file_path)
        except Exception as e:
            logger.debug(f"Failed to extract deps from {file_path}: {e}")
        return [d.strip().split("==")[0].split(">=")[0].split("<=")[0].lower() for d in deps if d.strip()]

    def _extract_pyproject_deps(self, file_path: Path) -> List[str]:
        """Extract dependencies from pyproject.toml using tomllib (3.11+) or tomli."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                # Fallback: regex extraction
                return self._extract_regex_deps(file_path.read_text(encoding="utf-8"))

        with open(file_path, "rb") as f:
            data = tomllib.load(f)
        project_data = data.get("project", {}) if isinstance(data, dict) else {}
        return project_data.get("dependencies", [])

    def _extract_setup_py_deps(self, file_path: Path) -> List[str]:
        """Extract dependencies from setup.py (best-effort regex)."""
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return self._extract_regex_deps(text)

    def _extract_regex_deps(self, text: str) -> List[str]:
        """Fallback regex-based dependency extraction."""
        import re
        deps = []
        # Match install_requires list
        match = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if match:
            for line in match.group(1).splitlines():
                m = re.search(r'"([^"]+)"', line.strip())
                if m:
                    deps.append(m.group(1))
        return deps
