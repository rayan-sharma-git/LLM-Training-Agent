"""Prompt quality analyzer — reads real prompt template files and evaluates them."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.schemas import ProjectContext, PromptAnalysisResult
from ai.llm import LLMHelper
from cleaning import dataset_io

logger = logging.getLogger(__name__)


class PromptAnalyzer:
    """Analyzes prompt template quality by reading actual template files."""

    def __init__(self):
        self._llm = LLMHelper()

    # Regex to detect template variables: {var}, {{var}}, <var>, %var%
    _PLACEHOLDER_PATTERNS = [
        re.compile(r"\{(\w+)\}"),       # {var}
        re.compile(r"{{(\w+)}}"),       # {{var}}
        re.compile(r"<(\w+)>"),         # <var>
    ]

    _AMBIGUITY_WORDS = [
        "etc", "and so on", "such as", "maybe", "perhaps",
        "sometimes", "often", "usually", "could be", "might be",
        "if applicable", "as appropriate", "depending on",
    ]

    _CONFLICTING_PATTERNS = [
        (r"do not.*but also", "Conflicting instruction: 'do not' vs 'but also'"),
        (r"never.*always", "Potential conflict: 'never' vs 'always'"),
        (r"must.*optional", "Conflicting instruction: 'must' vs 'optional'"),
    ]

    async def analyze(self, context: ProjectContext) -> PromptAnalysisResult:
        """Analyze prompt templates for quality, clarity, and consistency."""
        templates = context.prompt_templates or []

        if not templates:
            return PromptAnalysisResult(
                template_name="none",
                prompt_complexity="simple",
                ambiguity_score=0.0,
                clarity_score=0.0,
                formatting_score=0.0,
                instruction_quality_score=0.0,
                consistency_score=0.0,
                detected_issues=["No prompt templates found in project"],
                recommendations=["Add a prompt template file (e.g., prompt.txt)"],
                confidence="very_low",
            )

        # Resolve the first entry: it may be a file path OR the template
        # content itself (the /prompt/analyze endpoint accepts inline text).
        resolved = self._resolve_template(context, templates[0])
        template_name = str(resolved["name"] or "unknown")

        if resolved["content"] is None:
            return PromptAnalysisResult(
                template_name=template_name,
                prompt_complexity="simple",
                ambiguity_score=0.3,
                clarity_score=0.0,
                formatting_score=0.0,
                instruction_quality_score=0.0,
                consistency_score=0.0,
                detected_issues=[f"Template file not found: {template_name}"],
                recommendations=["Ensure the prompt template path is correct, or paste the template content"],
                confidence="very_low",
            )

        content = resolved["content"]
        logger.info(f"Analyzing prompt template: {template_name}")

        # Step 1: Deterministic analysis
        detected_issues: List[str] = []
        recommendations: List[str] = []

        # Check for missing placeholders
        placeholders = self._extract_placeholders(content)
        if not placeholders:
            detected_issues.append("No template variables found (e.g., {var})")
            recommendations.append("Add template variables for dynamic content")
        else:
            # Check if placeholders are used but not defined elsewhere
            undefined = self._check_undefined_placeholders(content, placeholders)
            if undefined:
                detected_issues.append(f"Potentially undefined placeholders: {', '.join(undefined[:5])}")

        # Check for conflicting instructions
        for pattern, issue in self._CONFLICTING_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                detected_issues.append(issue)

        # Check for prompt leakage
        if re.search(r"(system:|assistant:|user:)", content.strip()[:500], re.IGNORECASE):
            detected_issues.append("Possible prompt leakage: role tags found in template")

        # Compute scores
        ambiguity_score = self._compute_ambiguity(content)
        clarity_score = self._compute_clarity(content)
        formatting_score = self._compute_formatting(content)
        instruction_score = self._compute_instruction_quality(content)
        consistency_score = self._compute_consistency(content)

        if ambiguity_score > 0.4:
            recommendations.append("Clarify ambiguous instructions with concrete examples")
        if clarity_score < 0.7:
            recommendations.append("Rewrite unclear instructions for better clarity")
        if formatting_score < 0.8:
            recommendations.append("Fix formatting inconsistencies (spacing, capitalization)")

        # Determine complexity
        word_count = len(content.split())
        if word_count < 50:
            complexity = "simple"
        elif word_count < 150:
            complexity = "moderate"
        elif word_count < 300:
            complexity = "complex"
        else:
            complexity = "very_complex"

        confidence = "high" if len(content) > 50 else "low"

        # Step 2: Optional LLM enhancement
        if self._llm.is_available:
            llm_result = await self._llm_enhance(content)
            if llm_result:
                confidence = "high"
                # Merge LLM findings
                for issue in llm_result.get("detectedIssues", []):
                    if issue not in detected_issues:
                        detected_issues.append(issue)
                if "ambiguityScore" in llm_result:
                    ambiguity_score = float(llm_result["ambiguityScore"])
                if "clarityScore" in llm_result:
                    clarity_score = float(llm_result["clarityScore"])

        return PromptAnalysisResult(
            template_name=template_name,
            prompt_complexity=complexity,
            ambiguity_score=round(ambiguity_score, 2),
            clarity_score=round(clarity_score, 2),
            formatting_score=round(formatting_score, 2),
            instruction_quality_score=round(instruction_score, 2),
            consistency_score=round(consistency_score, 2),
            detected_issues=detected_issues,
            recommendations=recommendations,
            confidence=confidence,
        )

    def _extract_placeholders(self, content: str) -> List[str]:
        """Extract all template variables from the content."""
        placeholders = []
        for pattern in self._PLACEHOLDER_PATTERNS:
            placeholders.extend(pattern.findall(content))
        return placeholders

    def _check_undefined_placeholders(self, content: str, placeholders: List[str]) -> List[str]:
        """Check for potentially undefined placeholders."""
        # Simple heuristic: if a placeholder appears only once and isn't defined elsewhere
        undefined = []
        for ph in placeholders:
            count = content.count(f"{{{ph}}}") + content.count(f"{{{{{ph}}}}}")
            if count <= 1 and ph not in ("instruction", "input", "output", "response", "context"):
                undefined.append(ph)
        return list(set(undefined))

    def _resolve_template(self, context: ProjectContext, entry: str) -> Dict[str, Optional[str]]:
        """Resolve a prompt-template entry to ``(name, content)``.

        The entry may be either a file path or the raw template content (the
        ``/prompt/analyze`` endpoint accepts inline template text). A non-empty
        string that is not an existing path and contains a newline (or is long
        enough not to be a plausible path) is treated as inline content.
        ``name`` is always a string; ``content`` is None when the template
        could not be resolved.
        """
        entry = (entry or "").strip()
        if not entry:
            return {"name": "empty", "content": None}

        candidates = [Path(entry)]
        if context.project_path and not Path(entry).is_absolute():
            candidates.append(Path(context.project_path) / entry)

        for candidate in candidates:
            if candidate.is_file():
                try:
                    return {"name": candidate.name, "content": dataset_io.read_text(candidate)}
                except OSError as e:
                    logger.warning(f"Failed to read template {candidate}: {e}")
                    return {"name": candidate.name, "content": None}

        looks_like_path = ("\n" not in entry) and len(entry) < 260 and " " not in entry
        if looks_like_path:
            return {"name": entry, "content": None}
        # Inline template content
        return {"name": "inline template", "content": entry}

    def _compute_ambiguity(self, content: str) -> float:
        """Score ambiguity (0=low ambiguity, 1=high ambiguity)."""
        score = 0.0
        lower = content.lower()
        for word in self._AMBIGUITY_WORDS:
            if word in lower:
                score += 0.1
        if "?" in content:
            score += 0.05
        if "..." in content:
            score += 0.1
        return min(1.0, score)

    def _compute_clarity(self, content: str) -> float:
        """Score clarity (0=unclear, 1=clear)."""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return 0.0
        score = 1.0
        # Penalize very long lines (hard to follow)
        if any(len(line) > 200 for line in lines):
            score -= 0.2
        # Penalize missing structure (no clear instruction/response split)
        if "instruction" not in content.lower() and "###" not in content and "---" not in content:
            score -= 0.1
        return max(0.0, min(1.0, score))

    def _compute_formatting(self, content: str) -> float:
        """Score formatting consistency."""
        if not content.strip():
            return 0.0
        score = 1.0
        # Check consistent indentation
        lines = content.splitlines()
        indents = set()
        for line in lines:
            if line.strip():
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                indents.add(indent)
        if len(indents) > 4:
            score -= 0.2
        # Check consistent punctuation at end
        if content.rstrip()[-1:] not in (".", "!", "?", "}", '"', "'"):
            score -= 0.1
        return max(0.0, min(1.0, score))

    def _compute_instruction_quality(self, content: str) -> float:
        """Score instruction quality."""
        score = 1.0
        lower = content.lower()
        # Check for explicit instructions
        instruction_indicators = ["do", "don't", "write", "create", "analyze", "explain", "return", "format", "include"]
        found = sum(1 for ind in instruction_indicators if ind in lower)
        if found < 2:
            score -= 0.3
        if "as an" not in lower and "you are" not in lower:
            score -= 0.1
        return max(0.0, min(1.0, score))

    def _compute_consistency(self, content: str) -> float:
        """Score overall consistency."""
        score = 1.0
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return 0.0
        # Check if all lines start with similar formatting
        starts = [line[0] for line in lines if line]
        if len(set(starts)) > len(starts) * 0.7:
            score -= 0.3
        return max(0.0, min(1.0, score))

    async def _llm_enhance(self, content: str) -> Optional[Dict[str, Any]]:
        """Use LLM to enhance prompt analysis."""
        try:
            prompt = f"""Analyze the following prompt template for quality, clarity, and consistency.

Template:
{content[:2000]}

Return JSON with: ambiguityScore (0-1), clarityScore (0-1), formattingScore (0-1),
detectingIssues (list of strings), confidence (very_high|high|medium|low|very_low).
Base your analysis ONLY on the provided template text."""
            return await self._llm.call_structured(prompt, temperature=0.3)
        except Exception as e:
            logger.debug(f"LLM prompt enhancement failed: {e}")
            return None
