"""Build structured project context from scan results."""
from __future__ import annotations

from typing import Any, Dict
from models.schemas import ProjectContext


class ContextBuilder:
    """Build ProjectContext from scanner results."""
    
    def build(self, scan_result: Dict[str, Any]) -> ProjectContext:
        """Build structured context from scan results."""
        return ProjectContext(
            project_name=scan_result.get("project_name", "unknown"),
            project_path=scan_result.get("project_path", ""),
            detected_framework=scan_result.get("framework"),
            framework_version=scan_result.get("framework_version"),
            base_model=scan_result.get("base_model"),
            dataset_paths=scan_result.get("dataset_paths", []),
            prompt_templates=scan_result.get("prompt_templates", []),
            configuration_files=scan_result.get("configuration_files", []),
            training_scripts=scan_result.get("training_scripts", []),
            evaluation_scripts=scan_result.get("evaluation_scripts", []),
            inference_scripts=scan_result.get("inference_scripts", []),
            hardware_information=scan_result.get("hardware_information", {}),
            project_statistics=scan_result.get("project_statistics", {}),
        )