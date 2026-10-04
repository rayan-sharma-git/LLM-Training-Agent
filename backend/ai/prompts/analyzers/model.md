# Prompt - Model Analysis

Evaluate the selected base model for suitability to the intended task.

## Input
- Model name/identifier
- Task description
- Known capabilities

## Output Schema
```json
{
  "reasoningCapability": "low|medium|high",
  "codingCapability": "low|medium|high",
  "multilingualCapability": "low|medium|high",
  "instructionFollowingCapability": "low|medium|high",
  "speedScore": "slow|medium|fast",
  "memoryEfficiency": "low|medium|high",
  "strengths": ["string"],
  "weaknesses": ["string"],
  "recommendedAlternatives": ["string"],
  "confidence": "very_high|high|medium|low|very_low"
}
```

## Rules
- Compare known models objectively
- Include trade-offs for alternatives
- Never recommend without justification
- Base analysis on model capabilities, not speculation