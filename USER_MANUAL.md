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

## 10. Backend runtime setup (automatic)

On first use the extension prepares its Python backend automatically — you
do not install anything by hand:

1. It finds a Python 3.10+ interpreter (`llmTrainingAgent.pythonPath` if you
   set one, otherwise a bundled interpreter, otherwise `python`/`py` on PATH).
2. It checks that every package in the bundled `backend/requirements.txt`
   imports (developer tools from the root `requirements.txt` are never
   installed for normal users).
3. If packages are missing it shows **Install Dependencies** (plus
   **Show Manual Command** / **Cancel**). Approving runs pip with progress
   in the notification area and streams output to the
   **LLM Training Agent: Backend** output channel.
4. After installing it re-checks the imports instead of assuming success,
   then starts the backend. A declined prompt is remembered for the session
   so you are not asked again on every click.

To check state yourself: open the **LLM Training Agent: Backend** output
channel — it logs the extension install dir, the backend entry point, each
Python candidate tried, and whether it was ready/missing/too old. The
interpreter actually used is the `Python interpreter : ... (ready ...)` line;
it is `llmTrainingAgent.pythonPath` when that setting is non-empty.

If setup fails: read the channel lines (missing packages, "too old", "could
not be started", or pip/network errors each get a distinct message), then
retry by running **Analyze Project** again. For a manual install, run the
exact command the channel printed under `Manual install command` — with the
extension's own backend it has the form below, using YOUR interpreter path
from the channel log and the bundled requirements file inside the installed
extension (on Windows the extension lives under
`C:\Users\<you>\.vscode\extensions\rayansharma.llm-training-agent-1.0.0`):

```powershell
& "<python-from-channel-log>" -m pip install -r "<extension-install-dir>\backend\requirements.txt"
```

Concrete example when the channel shows the interpreter
`C:\Python312\python.exe` and the install dir above:

```powershell
& "C:\Python312\python.exe" -m pip install -r "C:\Users\<you>\.vscode\extensions\rayansharma.llm-training-agent-1.0.0\backend\requirements.txt"
```

From a local repository checkout with that interpreter selected, the
equivalent is `python -m pip install -r backend/requirements.txt` run from
the repository root. Confirm with
`& "<python>" -c "import fastapi, uvicorn, sqlalchemy, httpx"` (no output =
success), then retry **Analyze Project**. Developer tooling
(`pytest`, `ruff`, `black`, `mypy`) lives only in the root
`requirements.txt` and is never needed to run the extension.

## 11. Troubleshooting and limitations

- Backend will not start: open the **LLM Training Agent: Backend** output channel (see §10). Usually Python or its packages are missing — approving the install prompt fixes it; the channel names the exact interpreter, missing packages, and manual command.
- Webview shows "Could not register service worker" or buttons do nothing: reload the window. If it persists, the output channel and the extension log (command names only, never keys) say which handler failed.
- No folder open / path outside the open folder: open the project folder first; only it is ever read.
- Chat says no provider is available: configure one in Settings and test the connection.

The extension never invents accuracy scores, and model prices come from a bundled reference that may be outdated — verify costs before spending. GPU detection is best-effort. Developed and tested mainly on Windows.
