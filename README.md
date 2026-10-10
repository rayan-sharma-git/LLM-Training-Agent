# LLM Training Agent

A VS Code extension that reads your LLM fine-tuning project — datasets, prompts, training config, hardware — and tells you whether fine-tuning is worth it, what it would cost, how long it would take, and what to fix first.

Install the single `llm-training-agent-1.0.0.vsix` file at the repository root, open your project folder, and work from the extension's panel in the activity bar.

## Basics

### What this is

A VS Code extension for anyone preparing or considering a fine-tuning run. It adds Overview, Chat, Reports, and Settings views plus commands such as Analyze Project, Analyze Dataset, Open Chat, View Report, Configure AI Provider, and View Changes.

Open your project folder, run Analyze Project, and read the report in the Chat panel. Ask follow-ups there, revisit saved reports, switch AI providers in Settings, and review proposed file edits in a diff before anything is applied.

### Who it is for

Engineers, researchers, students, and hobbyists with a fine-tuning project who need a grounded second opinion before spending GPU hours. No training expertise is needed to read the report.

### Main features

Analyze Project produces one report: detected task, dataset size and quality, prompt issues, hyperparameter review, model advice matched to your VRAM, real GPU detection, time and cost estimates with stated assumptions, risks, prioritised recommendations, and a next experiment. Values are labelled *measured*, *calculated*, *heuristic*, or *assumed*; unknown stays unknown, and accuracy scores are never invented. Without an AI provider you still get the measurements and rule-based guidance.

### Installing and learning more

You need VS Code 1.85.0 or newer and a Python 3.10+ interpreter; no API key is required to start (Ollama is the local default). See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for installation and [USER_MANUAL.md](USER_MANUAL.md) for every feature.

## Technical Details

### How it works

Deterministic code measures what can be measured (dataset statistics, FLOPs-based time/VRAM estimates, token-based cost arithmetic); the LLM interprets those measurements (judgement, prompt reading, trade-offs). The pipeline degrades to measurements plus rules when no provider is configured.


### Technology stack

- **Backend:** Python 3.10.6, FastAPI, Pydantic v2, SQLAlchemy 2 (async) over SQLite, httpx, optional `google-generativeai` / `cohere` SDKs.
- **Extension:** TypeScript, VS Code Extension API, axios; vitest and `tsc --noEmit`.
- **Packaging:** `vsce` builds the VSIX; `extension/scripts/sync-backend.js` bundles the backend into it.
- **Tests:** pytest (`backend/tests/`), vitest (`extension/tests/suite/`), VSIX audit (`extension/scripts/verify_vsix.py`), isolation check (`extension/scripts/e2e_two_projects.py`).

### Architecture

The extension bundles the Python backend inside the VSIX and starts it lazily as a local subprocess on `127.0.0.1:8000` (HTTP plus `/ws/analysis` WebSocket for progress). Nothing is uploaded; only the open folder is read. Extension resources resolve from the install directory only, and settings (`llmTrainingAgent.*`) are user-level, never per project. See `extension/` (UI, commands, backend lifecycle) and `backend/` (scanning, analysis, storage, LLM access).

### Analysis pipeline

`backend/scanner/` finds datasets, prompts, configs, and scripts and builds the project context. Analyzers in `backend/analyzers/` cover datasets, prompts, hyperparameters, model advice, use case, and cost; `backend/analysis/orchestrator.py` merges them into one report (partial if a section fails). Then `backend/prediction/engine.py` (risks, provenance), `backend/recommendation/engine.py` (prioritised actions), `backend/reports/generator.py` (report, readiness score), and `backend/hardware/` (GPU detection, time/VRAM math) complete it. Cleaning (`backend/cleaning/`, output `<project>/.llm-training-agent/cleaned/`) runs via the chat agent and API.

### AI providers

Behind the `AIProvider` interface (`backend/ai/provider.py`), with graceful degradation in `backend/ai/llm.py`. Eight providers in `backend/ai/providers/`: Ollama (default, local, offline, no key), OpenAI, Anthropic, Gemini, DeepSeek, Cohere, OpenRouter, OpenAI-compatible. Keys live in VS Code SecretStorage, travel to the local backend only, stay in memory, and are scrubbed from logs by `backend/core/redaction.py`.

### Chat, reports, experiments, storage

Chat (`backend/chat/`) answers over real results with per-project history. Reports render in the extension Reports view. Experiments (`backend/experiments/service.py`, `/api/v1/experiments`) have no dedicated UI yet. Persistence is async SQLAlchemy over `llm_training_agent.db` plus `allowed-roots.txt`, both in VS Code global storage. API: `/api/v1` in `backend/api/routes.py` with Pydantic models in `backend/models/schemas.py`.

### File editing

Proposals follow PROPOSED → APPROVED → APPLIED (`backend/editing/file_editor.py`, `/files/...` routes, diff UI in `extension/src/commands/changeCommands.ts`). Nothing applies without approval; staging lives under `<project>/.llm_training_agent/pending_changes/`, backups under `<project>/.llm_training_agent_backups/`.

### Security

Workspace-only reads with traversal defence (`backend/core/security.py`); loopback binding; credentials scrubbed, tracebacks hidden.

### Testing

`cd backend && python -m pytest tests -q`; `cd extension && npm test` plus `npm run lint`. `verify_vsix.py` rejects secrets, caches, test files, and path assumptions.

### Packaging and distribution

`npm run package` produces `llm-training-agent-1.0.0.vsix`; the released copy lives at the repository root (the `extension/*.vsix` build copy is git-ignored). Rebuild and re-audit whenever VSIX contents change (see [CONTRIBUTING.md](CONTRIBUTING.md)).

### Development setup

Python 3.10.6 with root `requirements.txt` (runtime subset `backend/requirements.txt`); Node.js with `npm install` in `extension/`. Shared config in root `.vscode/` and `extension/.vscode/launch.json`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the GitHub flow.

### Project structure

[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) maps the repository: where extension and backend code live and where to change what, including generated directories that must not be edited.

### Documentation

| Document | Covers |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | Every user-facing feature |
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | Installing the VSIX |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Repository layout |
| [DELETION.md](DELETION.md) | Uninstall and cleanup |

### Contributing

Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md), then follow [CONTRIBUTING.md](CONTRIBUTING.md): focused PRs against `main`, green tests, no new dependencies or redesigns, no fabricated values, docs updated in the same PR.

### License

MIT. See [LICENSE](LICENSE).
