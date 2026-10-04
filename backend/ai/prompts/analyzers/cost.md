# Prompt - Cost Estimation

Estimate computational resources required for fine-tuning.

## Input
- Model size
- Dataset size
- Training configuration

## Output Schema
```json
{
  "estimatedTrainingTime": "string",
  "estimatedGPUHours": "number",
  "estimatedVRAMUsage": "string",
  "estimatedCheckpointSize": "string",
  "estimatedStorageRequirement": "string",
  "compatibleHardware": ["string"],
  "assumptions": ["string"],
  "confidence": "very_high|high|medium|low|very_low"
}
```

## Rules
- Base estimates on model size and dataset volume
- State assumptions clearly
- Never claim precision; use ranges where appropriate