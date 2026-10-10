# User Manual

The LLM Training Agent reads your fine-tuning project and tells you whether training it is worth it, what it would cost, how long it would take, and what to fix first.

For installation see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md). For removal see [DELETION.md](DELETION.md).

## 1. Opening the panel

Click the **LLM Training Agent** icon in the VS Code activity bar. It has four views: **Overview** (project status), **Chat** (analysis and conversation), **Reports** (saved reports), and **Settings** (provider and API key setup).

## 2. Analyze Project

This is the main workflow. Open your project folder, then click **Analyze Project** in Overview or run `LLM Training Agent: Analyze Project` from the command palette (`Ctrl+Shift+P`).

The extension scans your datasets, prompts, configs, and scripts and posts one report to the **Chat** panel. Large datasets can take a while; you can cancel the progress notification. If one section fails, the rest still runs and the report is marked partial. Only the open folder is ever read.

## 3. What the report contains

Each value is labelled *measured* (from your files), *calculated* (from a formula), *heuristic* (rule of thumb), or *assumed*. The report covers: detected task, dataset size and quality, prompt clarity, hyperparameter review, model advice, detected GPUs, training-time and cost estimates with assumptions, risks, prioritised recommendations, and a concrete next experiment.

## 4. Analyze Dataset

Run `LLM Training Agent: Analyze Dataset` and pick a folder in the dialog for a dataset-only check. For the full picture (model, hardware, cost), use Analyze Project.

## 5. Chat

The **Chat** view holds the report and answers follow-up questions about your project. History is per project. If no AI provider is configured, chat says so instead of guessing.

## 6. Reports

The **Reports** view (or `LLM Training Agent: View Report`) opens the latest saved report for the current project. If no analysis has run yet, it says so.

## 7. Configuring an AI provider

In the **Settings** view pick a provider (Ollama is the default and needs no API key), enter the model and key if needed, then **Save** and **Test Connection**. **Remove Key** deletes the stored key. Keys are kept in VS Code SecretStorage, never in your project.

Supported providers: Ollama, OpenAI, Anthropic, Google Gemini, DeepSeek, Cohere, OpenRouter, and any OpenAI-compatible server (e.g. LM Studio).

## 8. View Changes

Run `LLM Training Agent: View Changes` to review file edits the agent proposed. Each opens in a diff editor with **Approve & Apply**, **Reject**, or **Dismiss**. Nothing is written without approval, conflicts with your newer edits are blocked, backups are kept for rollback, and proposals expire after 24 hours.

Dataset cleaning and experiment tracking have no dedicated views: ask the chat agent to clean a dataset (output goes to `.llm-training-agent/cleaned/`, originals untouched), and experiments are available through the backend API.

## 9. Settings and storage

Settings (`llmTrainingAgent.provider`, `llmTrainingAgent.model`, `llmTrainingAgent.pythonPath`, `llmTrainingAgent.backendUrl`) are user-level and never stored in your project. Analysis history lives in the extension's VS Code global storage (`llm_training_agent.db`); pending changes and backups live in `.llm_training_agent/` and `.llm_training_agent_backups/` inside your project. See [DELETION.md](DELETION.md) for cleanup.

## 10. Troubleshooting and limitations

- Backend will not start: open the **LLM Training Agent: Backend** output channel. Usually Python or its packages are missing — approving the install prompt fixes it.
- No folder open / path outside the open folder: open the project folder first; only it is ever read.
- Chat says no provider is available: configure one in Settings and test the connection.

The extension never invents accuracy scores, and model prices come from a bundled reference that may be outdated — verify costs before spending. GPU detection is best-effort. Developed and tested mainly on Windows.
