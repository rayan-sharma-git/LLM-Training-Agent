# Prompt - Chat Context Building

Build context for the AI chat interface from project analysis results.

## Input
- ProjectContext
- EngineeringReport
- Recommendations
- Experiment History
- Conversation History

## Output Schema
```json
{
  "systemPrompt": "string",
  "userPrompt": "string",
  "contextSummary": "string",
  "relevantEvidence": ["string"],
  "confidence": "very_high|high|medium|low|very_low"
}
```

## Rules
- Prioritize project-specific information
- Do not rely on general LLM knowledge when project context exists
- Keep context within window limits
- Communicate uncertainty when context is insufficient