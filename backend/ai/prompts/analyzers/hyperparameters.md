# Prompt - Hyperparameter Analysis

Analyze the training hyperparameters for potential issues and optimization opportunities.

## Input
- Learning rate, batch size, epochs
- Optimizer, scheduler, weight decay
- LoRA configuration if applicable

## Output Schema
```json
{
  "efficiency_score": 0.0-1.0,
  "overfitting_risk": "low|medium|high|very_high",
  "underfitting_risk": "low|medium|high|very_high",
  "recommendations": ["string"],
  "confidence": "very_high|high|medium|low|very_low"
}
```

## Rules
- Flag learning rates above 1e-3 as potentially unstable
- Flag batch sizes below 4 as potentially inefficient
- Flag epochs above 10 as overfitting risk
- Base analysis on training best practices
- Never claim certainty