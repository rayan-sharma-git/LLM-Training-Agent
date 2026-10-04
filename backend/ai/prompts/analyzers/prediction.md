# Prompt - Training Outcome Prediction

Estimate likely outcomes of fine-tuning based on analysis results.

## Input
- Dataset quality metrics
- Hyperparameter configuration
- Model capabilities
- Cost estimates

## Output Schema
```json
{
  "instructionFollowingPrediction": "poor|fair|good|excellent",
  "hallucinationRisk": "low|medium|high|very_high",
  "reasoningPrediction": "poor|fair|good|excellent",
  "responseConsistencyPrediction": "poor|fair|good|excellent",
  "creativityPrediction": "low|medium|high",
  "formattingPrediction": "poor|fair|good|excellent",
  "likelyFailureModes": ["string"],
  "expectedStrengths": ["string"],
  "expectedWeaknesses": ["string"],
  "confidence": "very_high|high|medium|low|very_low"
}
```

## Rules
- Predictions are probabilistic, not deterministic
- Never claim exact model outputs
- Base predictions on provided analysis evidence
- Communicate uncertainty explicitly