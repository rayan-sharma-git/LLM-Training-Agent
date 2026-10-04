# Prompt - Recommendation Generation

Merge all analyzer outputs into prioritized engineering recommendations.

## Input
- Dataset analysis results
- Prompt analysis results
- Hyperparameter analysis results
- Model analysis results
- Cost estimates
- Predictions

## Output Schema
```json
{
  "recommendations": [
    {
      "category": "dataset|prompt|hyperparameter|model|prediction|cost|general",
      "title": "string",
      "description": "string",
      "reasoning": "string",
      "evidence": "string",
      "severity": "critical|high|medium|low|info",
      "confidence": "very_high|high|medium|low|very_low",
      "estimated_benefit": "string",
      "implementation_difficulty": "easy|moderate|hard",
      "estimated_engineering_time": "string",
      "affected_files": ["string"],
      "suggested_actions": ["string"]
    }
  ]
}
```

## Rules
- Prioritize by severity and impact
- Every recommendation must include evidence
- Avoid duplicate recommendations
- Never claim certainty