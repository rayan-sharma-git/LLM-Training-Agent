# User Manual

The LLM Training Agent reads your fine-tuning project and tells you whether training it is worth it, what it would cost, how long it would take, and what to fix first.

For installation see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md). For removal see [DELETION.md](DELETION.md).

## 1. Opening the panel

Click the **LLM Training Agent** icon in the VS Code activity bar. It has four views: **Overview** (project status), **Chat** (analysis and conversation), **Reports** (saved reports), and **Settings** (provider and API key setup).

## 2. Analyze Project

This is the main workflow. Open your project folder, then click **Analyze Project** in Overview or run `LLM Training Agent: Analyze Project` from the command palette (`Ctrl+Shift+P`).

The extension scans your datasets, prompts, configs, and scripts and posts one report to the **Chat** panel. Large datasets can take a while; you can cancel the progress notification. If one section fails, the rest still runs and the report is marked partial. Only the open folder is ever read.

## 3. What the report contains

Each value is labelled *measured* (from your files), *calculated* (from a formula), *heuristic* (rule of thumb), or *assumed*. The report opens with what was found in your own project: the training script(s) (for example `fine_tune.py`), the base model, and the fine-tuning method and settings read from the code (LoRA/QLoRA, rank, learning rate, epochs, batch size and so on); anything not found is left out rather than guessed. It then covers: detected task, dataset size and quality, prompt clarity, hyperparameter review, model advice, detected GPUs, training-time and cost estimates with assumptions, risks, prioritised recommendations, and a concrete next experiment.

## 4. Analyze Dataset

Run `LLM Training Agent: Analyze Dataset` and pick a folder in the dialog for a dataset-only check. For the full picture (model, hardware, cost), use Analyze Project.

## 5. Chat

The **Chat** view opens on a list of your previous chats for the current project (title, date and time, newest first). **New chat** starts an empty one; a chat is also created automatically the moment you send your first message, and it gets a short title from that message. Open a chat to see the whole conversation again, including change viewers and answered cards; use the back arrow to return to the list and the ✕ on a row to delete a chat. Chats are stored by the backend, so they are still there after restarting VS Code. Every message has a **Copy Message** button.

Besides answering questions, the chat can carry out requests itself:

- **Change a setting** – for example "change epochs from 3 to 5". The agent finds the setting in your scripts and config files (under its different names and formats), edits the right file, re-parses the file to check it still validates, and shows the edit as a red/green change. If the setting appears in several places it asks which one. Say "revert that" (or press **Revert**) to undo it; edits you made elsewhere in the file in the meantime are kept, and if the edited lines themselves were changed it refuses instead of guessing.
- **Clean data** – the agent first asks how to clean (script or AI), then whether to write **separate output files** or **modify the existing files**. What it did is remembered in your project (`.llm-training-agent/cleaning_jobs/`), so "merge the cleaned data", "restore the original data" or "discard the cleaned data" work later, also after a restart or from another chat. A merge checks that the files were not changed in between, keeps backups, verifies the result and then removes the temporary cleaned folder. Records the AI could not process are kept unchanged and summarised in one short note instead of one warning per chunk.
- **Run a terminal command** – for example `git status`. See *Permissions* below.
- **Describe where you train** – for example "I train on a RunPod A100 at $2.00/hr". Estimates then use it (see *Analyze Project*).

If no AI provider is configured, ordinary questions say so instead of guessing; the requests above do not need one.

## 6. Reports

The **Reports** view (or `LLM Training Agent: View Report`) opens the latest saved report for the current project. If no analysis has run yet, it says so.

## 7. Configuring an AI provider

In the **Settings** view pick a provider (Ollama is the default and needs no API key), enter the model and key if needed, then **Save** and **Test Connection**. **Remove Key** deletes the stored key. Keys are kept in VS Code SecretStorage, never in your project.

Supported providers: Ollama, OpenAI, Anthropic, Google Gemini, DeepSeek, Cohere, OpenRouter, and any OpenAI-compatible server (e.g. LM Studio).

## 8. View Changes

Run `LLM Training Agent: View Changes` (or ask "show the changes") to see the latest edits **inside the chat**: the file path, then one bordered block per change. Removed lines are red and labelled **Removed**, replacement lines are green and labelled **Added**, with line numbers and unchanged context lines around them. Nothing opens in a separate editor. Conflicts with your newer edits are blocked, backups are kept for rollback, and proposals expire after 24 hours.

Dataset cleaning is done from the chat (see *Chat*), and experiments are available through the backend API.

## 9. Permissions

**Settings → Permissions** has four independent switches. They are enforced by the backend, not just hidden in the interface:

| Permission | Default | Allows |
| --- | --- | --- |
| Read files | on | Reading project files for analysis, datasets and finding settings |
| Write files | off | Editing settings, cleaning/merging data, reverting |
| Run safe terminal commands | off | Read-only commands such as `ls`, `git status`, `git diff`, `pip list`, `nvidia-smi` |
| Run unrestricted terminal commands | off | Any command except a short list of destructive ones |

When something needs a permission that is off, the chat shows what it wants to do with **Allow** / **Deny** buttons; Allow applies to that one action only. Without unrestricted commands, any command that is not on the safe list needs that approval each time. Commands that could destroy data (for example `rm -rf /`, formatting a disk) are always refused, even with everything turned on. Commands run inside the project folder with secret-looking environment variables removed, a time limit and a cap on output, and their output is scrubbed of keys. In an untrusted VS Code workspace, writing and commands are switched off regardless of these settings. The permission settings are user-level, so a project's own `.vscode/settings.json` cannot switch them on.

## 10. Settings and storage

**Training Hardware** (Settings view) tells Analyze Project where you train: *my own GPU*, *rented cloud GPU*, or detect automatically. A training-time estimate uses your project and hardware. A **cost** is only shown for a cloud GPU and only when you entered its hourly rate (cost = estimated runtime × that rate, with the assumptions listed); for your own GPU no rental cost is shown. Nothing about models, GPUs, settings, runtimes or prices is assumed when it cannot be found in your project or given by you.

Settings (`llmTrainingAgent.provider`, `llmTrainingAgent.model`, `llmTrainingAgent.pythonPath`, `llmTrainingAgent.backendUrl`, `llmTrainingAgent.permissions.*`, `llmTrainingAgent.hardware.*`) are user-level and never stored in your project. Analysis history lives in the extension's VS Code global storage (`llm_training_agent.db`); pending changes and backups live in `.llm_training_agent/` and `.llm_training_agent_backups/` inside your project. See [DELETION.md](DELETION.md) for cleanup.

## 11. Backend runtime setup (automatic)

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

## 12. Troubleshooting and limitations

- Backend will not start: open the **LLM Training Agent: Backend** output channel (see §10). Usually Python or its packages are missing — approving the install prompt fixes it; the channel names the exact interpreter, missing packages, and manual command.
- Webview shows "Could not register service worker" or buttons do nothing: reload the window. If it persists, the output channel and the extension log (command names only, never keys) say which handler failed.
- No folder open / path outside the open folder: open the project folder first; only it is ever read.
- Chat says no provider is available: configure one in Settings and test the connection.

The extension never invents accuracy scores, and model prices come from a bundled reference that may be outdated — verify costs before spending. GPU detection is best-effort. Developed and tested mainly on Windows.
