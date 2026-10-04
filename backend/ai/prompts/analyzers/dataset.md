# Prompt - Dataset Analysis

Analyze the provided dataset information and generate a structured assessment.

## Input
- Dataset paths
- Sample statistics
- Known quality indicators

## Output Schema
```json
{
  "quality_score": 0.0-1.0,
  "findings": ["string"],
  "warnings": ["string"],
  "recommendations": ["string"],
  "confidence": "very_high|high|medium|low|very_low"
}
```

## Rules
- Base analysis strictly on provided data
- Include numerical evidence where possible
- Never claim certainty; assign confidence based on data completeness
- Do not fabricate statistics