# Prompt - Prompt Analysis

Analyze the provided prompt template for quality, clarity, and consistency.

## Input
- Template text
- Variables/placeholders
- Usage context

## Output Schema
```json
{
  "promptComplexity": "simple|moderate|complex|very_complex",
  "ambiguityScore": 0.0-1.0,
  "clarityScore": 0.0-1.0,
  "formattingScore": 0.0-1.0,
  "instructionQualityScore": 0.0-1.0,
  "consistencyScore": 0.0-1.0,
  "detectedIssues": ["string"],
  "recommendations": ["string"],
  "confidence": "very_high|high|medium|low|very_low"
}
```

## Rules
- Flag ambiguous instructions
- Detect missing placeholders
- Identify formatting inconsistencies
- Never claim certainty