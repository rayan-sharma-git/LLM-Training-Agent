# Prompt - Report Generation

Generate a final engineering report from all analysis results.

## Input
- ProjectContext
- All analyzer results
- Recommendations
- Predictions
- Cost estimates

## Output Schema
```json
{
  "executiveSummary": "string",
  "projectHealthScore": 0.0-1.0,
  "trainingReadinessScore": 0.0-1.0,
  "datasetSummary": {},
  "promptSummary": {},
  "hyperparameterSummary": {},
  "modelSummary": {},
  "predictionSummary": {},
  "costSummary": {},
  "prioritizedRecommendations": ["Recommendation"],
  "actionPlan": ["string"]
}
```

## Rules
- Summarize findings clearly
- Rank recommendations by severity and impact
- Include evidence for all claims
- Communicate confidence levels
- Never claim certainty