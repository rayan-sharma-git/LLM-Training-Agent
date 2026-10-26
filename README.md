# LLM Training Agent

A VS Code extension that looks at your LLM fine-tuning project and tells you three things: is fine-tuning worth it, what will it cost and how long will it take, and what should you fix first.

It reads your datasets, prompts, training config and hardware. It can also chat about your project, clean your dataset, and propose file changes that you review before anything is saved.

There are two parts below. **Basics** is for people who just want to use the extension. **Technical Details** is for contributors and anyone who wants to know how it works inside.

---

# Basics

## What you need

- VS Code 1.85 or newer.
- Python 3.10 or newer on your computer. Most computers already have it.
- Nothing else. No API key is needed to start, because the default AI provider (Ollama) runs on your own computer.

## Install it

Download `llm-training-agent-1.0.0.vsix` from the top level of this repository. In VS Code open Extensions, click the `⋯` menu, choose **Install from VSIX…** and pick the file. Full steps are in [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md).

The first time you open the extension, it checks that your Python has the packages it needs. If something is missing, it installs them for you automatically (progress shows in **View → Output → LLM Training Agent: Backend**). You never run pip by hand. If you would rather be asked first, turn off the setting `llmTrainingAgent.autoInstallDependencies`.

## Use it

Open your project folder, then click the **LLM Training Agent** icon in the left sidebar. It has four views: **Overview**, **Chat**, **Reports** and **Settings**.

### Analyze Project

The main feature. It scans your datasets, prompts, configs and scripts and writes one report into the Chat view. The report covers:

- what kind of task you are training for,
- dataset size and quality,
- problems in your prompts,
- a review of your hyperparameters,
- model advice that fits your GPU memory,
- the GPUs it found on your machine,
- training time and cost estimates, with the assumptions written down,
- risks, a prioritised list of fixes, and one concrete next experiment.

Every number is labelled *measured* (from your files), *calculated* (from a formula), *heuristic* (a rule of thumb) or *assumed*. If something cannot be known it says so, and it never makes up an accuracy score. Without an AI provider you still get all the measurements and the rule-based advice.

### Chat

Ask questions about your project in plain language. Answers are based on the report, so run **Analyze Project** first. Without it the chat only has general knowledge and tells you so.

- **Copy button.** Every answer has a **Copy answer** button, and every code block has its own **Copy** button that copies only that snippet.
- **Chat can do things.** Say "clean my dataset" or "remove duplicates from my dataset" and it does it instead of explaining how (see below). Questions such as "how do I remove duplicates?" are still answered normally.
- **Per project.** Each project has its own chat history.

### Clean Dataset

Cleans every dataset file in your project (`.jsonl`, `.json`, `.csv` and similar). Start it from the chat or from **LLM Training Agent: Clean Dataset** in the command palette. It asks you two things:

1. **How to clean:**
   - **Scripts (free).** Fast, nothing leaves your computer. Fixes stray whitespace and control characters.
   - **AI model.** Sends your data to the model in small chunks so it can also fix typos and formatting. With a paid API this uses your credits, so it asks before doing it.
2. **Remove duplicates?** Drops exact copies and keeps the first one. This is always a free script step.

Your files are never overwritten silently. The cleaned versions are shown as a change you can review and apply (see View Changes), and every record is accounted for: records out plus duplicates removed always equals records in.

### View Changes

Run **LLM Training Agent: View Changes** to see edits the agent proposed. Each one opens as a side-by-side diff. You choose **Approve & Apply**, **Reject** or **Dismiss**. Nothing is written without your approval, backups are kept so you can roll back, and proposals expire after 24 hours.

### Analyze Dataset

Run **LLM Training Agent: Analyze Dataset** and pick a folder for a dataset-only check. For the full picture (model, hardware, cost) use Analyze Project.

### Reports

The **Reports** view opens the latest saved report for the current project.

### Settings: choose an AI provider

In the **Settings** view pick a provider, type the model name, paste your key if it needs one, then press **Save** and **Test Connection**.

Supported providers: Ollama (default, local, no key), OpenAI, Anthropic, Google Gemini, DeepSeek, Cohere, OpenRouter, and any OpenAI-compatible server such as LM Studio.

Your key is stored in VS Code's secure storage, never in your project. After saving, the page only shows that a key is saved. It never shows the key again. You enter it once and it is remembered, including after restarts.

### If something goes wrong

Open **View → Output** and choose **LLM Training Agent: Backend**. It says what the extension tried and why it failed. More help is in [USER_MANUAL.md](USER_MANUAL.md).

## More documents

| Document | What it is for |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | Every feature and how to use it |
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | Installing the VSIX file |
| [DELETION.md](DELETION.md) | Uninstalling and cleaning up leftovers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | A map of the code for contributors |

---

# Technical Details

## How it works

Deterministic code measures whatever can be measured: dataset statistics, FLOPs-based time and VRAM estimates, token-based cost arithmetic. The LLM then interprets those measurements (judgement, prompt reading, trade-offs). With no provider configured, the pipeline falls back to measurements plus rules.

## Technology stack

- **Backend:** Python 3.10+, FastAPI, Pydantic v2, async SQLAlchemy 2 over SQLite, httpx. Every provider is called over plain REST, so there are no provider SDKs.
- **Extension:** TypeScript, the VS Code Extension API, axios. Tests use vitest and `tsc --noEmit`.
- **Packaging:** `vsce` builds the VSIX. `extension/scripts/sync-backend.js` bundles the backend into it.
- **Tests:** pytest (`backend/tests/`), vitest (`extension/tests/suite/`), the VSIX audit (`extension/scripts/verify_vsix.py`), the webview wiring check (`verify-webview-wiring.js`), and the two-project isolation check (`extension/scripts/e2e_two_projects.py`).

## Architecture

The VSIX contains the Python backend. The extension starts it as a local subprocess on `127.0.0.1:8000` (HTTP, plus a `/ws/analysis` WebSocket for progress) and prepares it as soon as the extension opens. Nothing is uploaded, and only the open folder is read.

Extension resources are always resolved from the install directory, never from the open project. Settings (`llmTrainingAgent.*`) are user-level, never per project. The UI and backend lifecycle live in `extension/`. Scanning, analysis, storage and LLM access live in `backend/`.

## Backend startup and dependencies

1. `backendManager.ts` tries Python candidates in order: the `llmTrainingAgent.pythonPath` setting, a bundled interpreter, then `python` or `py` on PATH. Each candidate must be Python 3.10+.
2. `backend/check_requirements.py` (standard library only) compares the installed packages with `backend/requirements.txt`. It honours environment markers (so `tomli` is not demanded on Python 3.11+) and minimum versions (so SQLAlchemy older than 2.0.30 counts as outdated). If it cannot run, an import check is used instead.
3. If packages are missing they are installed automatically (the setting `llmTrainingAgent.autoInstallDependencies`, on by default). With it off, a modal prompt offers **Install Dependencies** instead, and a declined prompt is remembered for the session. Either way pip runs on the bundled `backend/requirements.txt` only, never the root file, then the check runs again instead of assuming success. If the Python refuses system-wide installs (PEP 668), the output channel explains how to use a virtual environment and `llmTrainingAgent.pythonPath`.
4. The backend is started with uvicorn, and its output goes to the **LLM Training Agent: Backend** channel.
5. The backend keeps API keys in memory only. Every time a new backend process starts, `backendSync.ts` sends it the saved provider, model, URLs and keys from SecretStorage and settings. Health is re-checked on every action, so a crashed backend is restarted rather than reported as running.

Two requirements files, on purpose:

- `backend/requirements.txt` has only what is needed to run. It ships inside the VSIX and is what the extension installs for users. Developer tools are never installed for users.
- The root `requirements.txt` has the same runtime packages plus test and lint tools. It is for contributors.

## Analysis pipeline

`backend/scanner/` finds datasets, prompts, configs and scripts and builds the project context. The analyzers in `backend/analyzers/` cover datasets, prompts, hyperparameters, model advice, use case and cost. `backend/analysis/orchestrator.py` merges them into one report, which is marked partial if a section fails. After that, `backend/prediction/engine.py` adds risks and provenance, `backend/recommendation/engine.py` adds prioritised actions, `backend/reports/generator.py` writes the report and readiness score, and `backend/hardware/` does GPU detection and time/VRAM math.

## Dataset cleaning

`backend/cleaning/` (route: `POST /api/v1/dataset/clean`). Every dataset file is split into chunks of records (25 records and 12,000 characters by default). With the AI model, each chunk is its own request, so a whole dataset is never sent at once. Answers that break the contract (wrong count or shape) are rejected and the original records are kept. With `useLlm: false` only scripts run and no AI call is made. `removeDuplicates: true` drops exact duplicates after cleaning, keeping the first copy and the order, even across chunks.

Output goes to `<project>/.llm-training-agent/cleaned/` with a `cleaning_manifest.json`. The guarantee is `records_out + duplicates_removed == records_in` per file.

In the extension, `cleaningFlow.ts` asks the two questions, calls the route, then proposes each changed file through the file-editing system below. The flow is written against an interface, so it is tested without VS Code.

## Chat actions

`chatIntents.ts` recognises instructions such as "remove duplicates from my dataset" with plain keyword rules. This works with every provider, including small local models that cannot call tools. Questions ("how do I…") go to the AI as usual. When no analysis exists, the chat engine tells the model it has not seen the project, so it does not pretend to.

## AI providers

All providers sit behind the `AIProvider` interface (`backend/ai/provider.py`), with graceful degradation in `backend/ai/llm.py`. There are eight in `backend/ai/providers/`: Ollama (default, local, offline, no key), OpenAI, Anthropic, Gemini, DeepSeek, Cohere (v2 API), OpenRouter and OpenAI-compatible. Keys live in SecretStorage, travel only to the local backend, stay in memory, and are scrubbed from logs by `backend/core/redaction.py`.

Provider errors are classified from the HTTP status and the provider's response body, so the chat can say what is wrong (for example "run `ollama pull <model>`") without leaking secrets.

## Webviews

The Chat and Settings pages are built by pure functions in `extension/src/views/webviewHtml.ts`, with a nonce-based CSP and no inline scripts. Their behaviour lives in plain files, `extension/media/chat.js` and `settings.js`, which are syntax-checked by the tests. Copying uses the extension host's clipboard, because a webview may not be allowed to write one. Assistant text is only ever inserted as text, never as HTML.

## Chat, reports, experiments, storage

Chat (`backend/chat/`) answers over real results, with per-project history. Reports show in the Reports view. Experiments (`backend/experiments/service.py`, `/api/v1/experiments`) have no UI yet. Persistence is async SQLAlchemy over `llm_training_agent.db`, plus `allowed-roots.txt`, both in VS Code global storage. The API is `/api/v1` in `backend/api/routes.py`, with Pydantic models in `backend/models/schemas.py`.

## File editing

Proposals go PROPOSED → APPROVED → APPLIED (`backend/editing/file_editor.py`, the `/files/...` routes, and the diff UI in `extension/src/commands/changeCommands.ts`). Nothing applies without approval. Pending changes are kept in `<project>/.llm_training_agent/pending_changes/` and backups in `<project>/.llm_training_agent_backups/`. A proposal can hold at most 2,000,000 characters.

## Security

Reads are limited to the open workspace, with path-traversal protection (`backend/core/security.py`). The backend binds to loopback only. Credentials are scrubbed and tracebacks are hidden.

## Testing

```bash
cd backend   && python -m pytest tests -q
cd extension && npm test && npm run lint
```

`verify_vsix.py` rejects secrets, caches, test files and path assumptions, and it checks that the webview scripts and the requirements checker are inside the package.

## Packaging

`npm run package` in `extension/` rebuilds everything from source and copies the result to `llm-training-agent-1.0.0.vsix` in the repository root (the `postpackage` hook). The copy inside `extension/` is git-ignored and deleted on each build. Rebuild and run `npm run verify:vsix` whenever the contents of the VSIX change. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Development setup

Python 3.10+ with the root `requirements.txt`, and Node.js with `npm install` in `extension/`. Shared editor config is in the root `.vscode/` and `extension/.vscode/launch.json`. A root `pytest.ini` puts `backend/` on the import path so `pytest backend/tests` works from the repository root.

## Contributing and license

Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md), then follow [CONTRIBUTING.md](CONTRIBUTING.md): focused pull requests against `main`, passing tests, no new dependencies or redesigns, no made-up numbers, and docs updated in the same pull request.

License: MIT. See [LICENSE](LICENSE).
