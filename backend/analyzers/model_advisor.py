"""Model selection advisor — evaluates model-task fit using a knowledge base."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from models.schemas import ProjectContext, ModelAnalysisResult
from ai.llm import LLMHelper

logger = logging.getLogger(__name__)


# Known model specifications database
# In production, this could be extended or fetched from HuggingFace API
_MODEL_DATABASE: Dict[str, Dict[str, Any]] = {
    # Llama 3 family
    "meta-llama/llama-3-8b-instruct": {
        "parameter_count": "8B",
        "context_length": 8192,
        "estimated_vram": "16GB (FP16) / 9GB (Q4)",
        "reasoning_capability": "high",
        "coding_capability": "high",
        "multilingual_capability": "high",
        "instruction_following_capability": "high",
        "speed_score": "fast",
        "memory_efficiency": "high",
        "strengths": ["Strong reasoning", "Excellent coding support", "Good multilingual"],
        "weaknesses": ["Requires ~16GB VRAM for FP16 inference"],
    },
    "meta-llama/llama-3-70b-instruct": {
        "parameter_count": "70B",
        "context_length": 8192,
        "estimated_vram": "140GB (FP16) / 40GB (Q4)",
        "reasoning_capability": "very_high",
        "coding_capability": "very_high",
        "multilingual_capability": "very_high",
        "instruction_following_capability": "very_high",
        "speed_score": "slow",
        "memory_efficiency": "low",
        "strengths": ["Excellent reasoning", "State-of-the-art coding", "Multilingual mastery"],
        "weaknesses": ["Requires multiple GPUs", "Slow inference", "High VRAM cost"],
    },
    # Gemma family
    "google/gemma-2b": {
        "parameter_count": "2B",
        "context_length": 8192,
        "estimated_vram": "4GB (FP16) / 2GB (Q4)",
        "reasoning_capability": "medium",
        "coding_capability": "medium",
        "multilingual_capability": "medium",
        "instruction_following_capability": "medium",
        "speed_score": "fast",
        "memory_efficiency": "high",
        "strengths": ["Lightweight", "Fast inference", "Low VRAM"],
        "weaknesses": ["Limited reasoning", "Weaker coding performance"],
    },
    "google/gemma-7b": {
        "parameter_count": "7B",
        "context_length": 8192,
        "estimated_vram": "15GB (FP16) / 4GB (Q4)",
        "reasoning_capability": "high",
        "coding_capability": "high",
        "multilingual_capability": "high",
        "instruction_following_capability": "high",
        "speed_score": "medium",
        "memory_efficiency": "medium",
        "strengths": ["Good balance of quality and size", "Strong instruction following"],
        "weaknesses": ["Higher VRAM requirement"],
    },
    # Mistral family
    "mistralai/mistral-7b-instruct": {
        "parameter_count": "7B",
        "context_length": 8192,
        "estimated_vram": "15GB (FP16) / 4GB (Q4)",
        "reasoning_capability": "medium",
        "coding_capability": "high",
        "multilingual_capability": "high",
        "instruction_following_capability": "high",
        "speed_score": "fast",
        "memory_efficiency": "medium",
        "strengths": ["Strong coding", "Fast inference", "Good multilingual"],
        "weaknesses": ["Weaker reasoning than Llama 3"],
    },
    # Qwen family
    "qwen/qwen-7b-chat": {
        "parameter_count": "7B",
        "context_length": 8192,
        "estimated_vram": "15GB (FP16) / 4GB (Q4)",
        "reasoning_capability": "medium",
        "coding_capability": "high",
        "multilingual_capability": "very_high",
        "instruction_following_capability": "high",
        "speed_score": "fast",
        "memory_efficiency": "medium",
        "strengths": ["Excellent multilingual (especially Chinese)", "Good coding"],
        "weaknesses": ["Reasoning is moderate"],
    },
    # DeepSeek
    "deepseek/deepseek-coder-7b-base": {
        "parameter_count": "7B",
        "context_length": 128000,
        "estimated_vram": "15GB (FP16) / 4GB (Q4)",
        "reasoning_capability": "high",
        "coding_capability": "very_high",
        "multilingual_capability": "medium",
        "instruction_following_capability": "high",
        "speed_score": "medium",
        "memory_efficiency": "medium",
        "strengths": ["Best-in-class coding", "128K context", "Strong reasoning"],
        "weaknesses": ["Weaker multilingual"],
    },
    # TinyLlama
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "parameter_count": "1.1B",
        "context_length": 4096,
        "estimated_vram": "3GB (FP16) / 1GB (Q4)",
        "reasoning_capability": "low",
        "coding_capability": "low",
        "multilingual_capability": "medium",
        "instruction_following_capability": "medium",
        "speed_score": "fast",
        "memory_efficiency": "high",
        "strengths": ["Very lightweight", "Runs on CPU", "Fast"],
        "weaknesses": ["Poor reasoning", "Limited coding", "Low quality"],
    },
    # Phi
    "microsoft/phi-2": {
        "parameter_count": "2.7B",
        "context_length": 2048,
        "estimated_vram": "6GB (FP16) / 2GB (Q4)",
        "reasoning_capability": "medium",
        "coding_capability": "medium",
        "multilingual_capability": "medium",
        "instruction_following_capability": "medium",
        "speed_score": "fast",
        "memory_efficiency": "high",
        "strengths": ["Compact size", "Good for mobile"],
        "weaknesses": ["Small context", "Limited multilingual"],
    },
}


class ModelAdvisor:
    """Advises on base model selection based on known model specifications."""

    def __init__(self):
        self._llm = LLMHelper()

    async def analyze(self, context: ProjectContext) -> ModelAnalysisResult:
        """Analyze selected model suitability for the task."""
        model_name = context.base_model or "unknown"
        logger.info(f"Analyzing model: {model_name}")

        # Step 1: Lookup model in database
        model_info = self._lookup_model(model_name)

        if model_info is None:
            # Try partial match
            model_info = self._partial_lookup(model_name)

        if model_info is None:
            # Unknown model — report unknowns honestly instead of inventing
            # "medium" capability ratings. The schema has no Optional for these
            # fields, so "unknown"/0 are used as explicit unknown markers.
            model_info = {
                "parameter_count": "unknown",
                "context_length": 0,
                "estimated_vram": "unknown (model not in the reference database)",
                "reasoning_capability": "unknown",
                "coding_capability": "unknown",
                "multilingual_capability": "unknown",
                "instruction_following_capability": "unknown",
                "speed_score": "unknown",
                "memory_efficiency": "unknown",
                "strengths": ["Unknown model — specifications not in the reference database"],
                "weaknesses": [
                    "Model not recognized; specifications cannot be verified.",
                    "Provide the model name (or a recognized alias) for a data-based assessment.",
                ],
            }

        # Add an honest note that the specs come from a static reference
        # database, not from verified/measure hardware performance.
        model_info.setdefault("weaknesses", []).append(
            "Specifications are reference data from a static knowledge base — not measured on your hardware."
        )

        # Step 2: Determine task and recommend alternatives
        task = self._infer_task(context)
        alternatives = self._recommend_alternatives(task, model_info)

        confidence = "high" if model_info["parameter_count"] != "unknown" else "low"

        # Hardware-aware check: compare the model's (reference) VRAM estimate
        # with the VRAM actually detected on this machine, when available.
        detected_gpu = self._first_detected_gpu(context)
        if detected_gpu and model_info["parameter_count"] != "unknown":
            vram_gb = detected_gpu.get("vram_gb")
            if vram_gb:
                model_info["weaknesses"].append(
                    f"Detected GPU '{detected_gpu.get('name')}' has {vram_gb:.0f}GB VRAM "
                    f"(model reference: {model_info['estimated_vram']}). "
                    "Verify feasibility for your intended precision/fine-tuning method."
                )

        # Step 3: Optional LLM enhancement — only accept well-formed output and
        # never let it overwrite the reference-derived numeric specifications.
        if self._llm.is_available:
            llm_result = await self._llm_enhance(model_name, task, model_info)
            if llm_result:
                if isinstance(llm_result.get("strengths"), list) and llm_result["strengths"]:
                    model_info["strengths"] = [str(s) for s in llm_result["strengths"]]
                if isinstance(llm_result.get("weaknesses"), list) and llm_result["weaknesses"]:
                    model_info["weaknesses"] = [str(w) for w in llm_result["weaknesses"]]
                if isinstance(llm_result.get("recommendedAlternatives"), list) and llm_result["recommendedAlternatives"]:
                    alternatives = [str(a) for a in llm_result["recommendedAlternatives"]]

        return ModelAnalysisResult(
            selected_model=model_name,
            parameter_count=model_info["parameter_count"],
            context_length=model_info["context_length"],
            estimated_vram=model_info["estimated_vram"],
            reasoning_capability=model_info["reasoning_capability"],
            coding_capability=model_info["coding_capability"],
            multilingual_capability=model_info["multilingual_capability"],
            instruction_following_capability=model_info["instruction_following_capability"],
            speed_score=model_info["speed_score"],
            memory_efficiency=model_info["memory_efficiency"],
            strengths=model_info["strengths"],
            weaknesses=model_info["weaknesses"],
            recommended_alternatives=alternatives,
            confidence=confidence,
        )

    def _first_detected_gpu(self, context: ProjectContext) -> Optional[Dict[str, Any]]:
        """Return the first actually-detected GPU (if any) from the project context."""
        hw = context.hardware_information or {}
        gpus = hw.get("gpus") or []
        return gpus[0] if gpus else None

    def _lookup_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Look up a model by exact name."""
        # Try exact match
        if model_name in _MODEL_DATABASE:
            return dict(_MODEL_DATABASE[model_name])

        # Try just the last part of the name (e.g., "Llama-3-8B" from "meta-llama/llama-3-8b-instruct")
        short_name = model_name.split("/")[-1].lower() if "/" in model_name else model_name.lower()
        for key, info in _MODEL_DATABASE.items():
            if key.split("/")[-1].lower() == short_name:
                return dict(info)

        # Try common aliases
        aliases = {
            "llama3": "meta-llama/llama-3-8b-instruct",
            "llama3-8b": "meta-llama/llama-3-8b-instruct",
            "llama3-70b": "meta-llama/llama-3-70b-instruct",
            "llama-3": "meta-llama/llama-3-8b-instruct",
            "gemma-2b": "google/gemma-2b",
            "gemma-7b": "google/gemma-7b",
            "mistral-7b": "mistralai/mistral-7b-instruct",
            "qwen-7b": "qwen/qwen-7b-chat",
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "phi-2": "microsoft/phi-2",
            "deepseek-coder": "deepseek/deepseek-coder-7b-base",
        }
        if short_name in aliases:
            return dict(_MODEL_DATABASE[aliases[short_name]])

        return None

    def _partial_lookup(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Try fuzzy matching for model names."""
        name_lower = model_name.lower()
        for key, info in _MODEL_DATABASE.items():
            if name_lower in key.lower() or key.lower() in name_lower:
                return dict(info)
        return None

    def _infer_task(self, context: ProjectContext) -> str:
        """Infer the task type from project context."""
        project_name = (context.project_name or "").lower()

        if any(word in project_name for word in ["code", "program", "dev", "script"]):
            return "coding"
        if "math" in project_name or "reason" in project_name:
            return "reasoning"
        if "translate" in project_name or "multilingual" in project_name:
            return "multilingual"
        return "general"

    def _recommend_alternatives(self, task: str, model_info: Dict[str, Any]) -> List[str]:
        """Recommend alternative models based on task and current model."""
        alternatives = []

        if model_info["parameter_count"] == "unknown":
            alternatives.append("Consider using a well-known model like Llama 3, Gemma, or Mistral.")
            return alternatives

        current_params = model_info["parameter_count"]

        if task == "coding":
            if "7B" in current_params or "8B" in current_params:
                alternatives.append("DeepSeek Coder 7B — best-in-class for coding tasks")
            else:
                alternatives.append("DeepSeek Coder 7B — best-in-class for coding tasks")
            alternatives.append("Llama 3 8B Instruct — strong general and coding performance")
        elif task == "reasoning":
            alternatives.append("Llama 3 70B Instruct — excellent reasoning capabilities")
            alternatives.append("DeepSeek Coder 7B — strong reasoning for smaller size")
        elif task == "multilingual":
            alternatives.append("Qwen 7B Chat — excellent multilingual support")
            alternatives.append("Llama 3 70B Instruct — very high multilingual capability")
        else:
            alternatives.append("Llama 3 8B Instruct — balanced general-purpose model")
            alternatives.append("Gemma 7B — lightweight alternative for most tasks")

        return alternatives

    async def _llm_enhance(self, model_name: str, task: str, model_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use LLM to provide additional model analysis."""
        try:
            prompt = f"""Evaluate the selected base model for suitability to the intended task.

Model: {model_name}
Task: {task}

Known capabilities:
- Parameter count: {model_info.get('parameter_count', 'unknown')}
- Context length: {model_info.get('context_length', 'unknown')}
- VRAM: {model_info.get('estimated_vram', 'unknown')}
- Reasoning: {model_info.get('reasoning_capability', 'medium')}
- Coding: {model_info.get('coding_capability', 'medium')}
- Multilingual: {model_info.get('multilingual_capability', 'medium')}

Return JSON with: strengths (list), weaknesses (list), recommendedAlternatives (list).
Compare models objectively. Include trade-offs for alternatives."""
            return await self._llm.call_structured(prompt, temperature=0.3)
        except Exception as e:
            logger.debug(f"LLM model enhancement failed: {e}")
            return None
