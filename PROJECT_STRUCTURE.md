# Project Structure

A map of the repository for contributors: where things live, and where to look when you want to change something.

There are two parts, a VS Code extension (`extension/`, TypeScript) and a Python backend (`backend/`), plus the docs in the root. There is no top-level `src/`.

```
.
├── README.md                 # What it is (Basics) and how it works (Technical Details)
├── USER_MANUAL.md            # Every feature and how to use it
├── INSTALLATION_GUIDE.md     # Installing the VSIX
├── CONTRIBUTING.md           # How to contribute
├── PROJECT_STRUCTURE.md      # This file
├── DELETION.md               # Uninstall and cleanup
├── LICENSE                   # MIT
├── requirements.txt          # For contributors: runtime packages + test and lint tools
├── pytest.ini                # Lets `pytest backend/tests` find backend modules from the root
├── .github/PULL_REQUEST_TEMPLATE.md   # Pull request checklist
├── .vscode/                  # Shared editor settings, tasks, recommended extensions
├── llm-training-agent-1.0.0.vsix      # THE installable extension (the only one tracked by git)
├── backend/                  # Python service: all analysis logic
└── extension/                # VS Code extension: all UI logic
```

## Root files

- **The VSIX.** `llm-training-agent-1.0.0.vsix` in the root is the only file users download. `npm run package` in `extension/` rebuilds it from the latest source. The `prepackage` step copies `backend/` into `extension/backend/` and compiles `src/` into `extension/out/`. `vsce` builds the VSIX. The `postpackage` step (`extension/scripts/promote-vsix.js`) copies it to the root and deletes the build copy, so the root file cannot go stale. Any `extension/*.vsix` on disk is a temporary, git-ignored build leftover.
- **`requirements.txt`** is for contributors only. The runtime list that users get is `backend/requirements.txt`.
- **`.gitignore`** excludes `venv310/`, `node_modules/`, caches, `.db` files, `extension/out/`, `extension/backend/` and `extension/*.vsix`. The root VSIX stays tracked.
- **Docs** live only in the root. Never add copies under `docs/`.

## `backend/`: the analysis service

A FastAPI app that does all the analysis, storage and AI calls.

| Folder or file | What it does |
|---|---|
| `main.py` | App factory |
| `requirements.txt` | Packages needed to run (shipped in the VSIX) |
| `check_requirements.py` | Standard-library-only check the extension runs to see if the Python has the right packages |
| `core/` | Config, security, secret scrubbing, errors, logging |
| `api/` | Routes, WebSocket, result mapping |
| `scanner/` | Finds datasets, prompts, configs and scripts; detects the framework |
| `analyzers/` | Dataset, prompt, hyperparameter, model, use case and cost analysis |
| `analysis/orchestrator.py` | Merges the analyzers into one report |
| `prediction/`, `recommendation/`, `reports/` | Risks, prioritised actions, report text |
| `hardware/` | GPU detection, time and VRAM math |
| `ai/` | Provider interface, LLM helper, `providers/` (one file per provider), prompt templates |
| `cleaning/` | Dataset cleaning in chunks (scripts or AI), in place or into copies, merge/discard, background jobs (`jobs.py`) |
| `chat/` | Chat answers and per-project history |
| `editing/` | `file_editor.py` (PROPOSED → APPROVED → APPLIED) and `assistant.py` (asks the AI to rewrite one file) |
| `experiments/` | Experiment tracking (API only, no view yet) |
| `storage/` | Database, migrations, models |
| `models/schemas.py` | The API's data shapes |
| `tests/` | pytest suite |

## `extension/`: the VS Code extension

| Folder or file | What it does |
|---|---|
| `package.json` | Commands, views, settings, scripts |
| `.vscodeignore` | What is left out of the VSIX |
| `.vscode/launch.json` | `F5` debug setup |
| `LICENSE` | Copy of the root license. Needed here because `vsce` packs it as `LICENSE.txt`. |
| `resources/icon.svg` | Sidebar icon |
| `src/extension.ts` | Start-up: wires everything together |
| `src/commands/` | Analyze, chat, change-review and clean-dataset commands |
| `src/services/` | `backendManager` (starts Python), `dependencyManager` (checks and installs packages), `backendSync` (re-sends saved key and provider to a new backend), `apiClient`, `chatIntents` (spots "clean my dataset" in chat), `cleaningFlow` and `editFlow` (the step-by-step actions), `providers`, `settings`, `projectRoot`, `extensionPaths` |
| `src/views/` | Tree views, the Chat and Settings webviews, and `webviewHtml.ts` which builds their pages |
| `media/` | `chat.js` and `settings.js`, the scripts the webviews run. They are plain files so tests can check them. |
| `src/utils/` | Report rendering |
| `tests/suite/` | vitest tests |
| `scripts/` | `sync-backend.js`, `promote-vsix.js`, `verify_vsix.py`, `verify-webview-wiring.js`, `e2e_two_projects.py` |

`backend/`, `out/`, `node_modules/` and any `*.vsix` inside `extension/` are generated. Do not edit or commit them.

## Where do I look to change…

| I want to change | Look in |
|---|---|
| A view or command | `extension/src/views/`, `extension/src/commands/` |
| A new command, view or setting | `extension/package.json` (`contributes`), then register it in `extension/src/extension.ts` |
| What chat can do on request | `extension/src/services/chatIntents.ts`, `cleaningFlow.ts`, `editFlow.ts` |
| The chat or settings page | `extension/media/`, `extension/src/views/webviewHtml.ts` |
| Report rendering, icon | `extension/src/utils/index.ts`, `extension/resources/icon.svg` |
| An endpoint or its shape | `backend/api/routes.py`, `backend/models/schemas.py` |
| Dataset, prompt, hyperparameter, model or cost logic | The matching file in `backend/analyzers/` |
| How sections become one report | `backend/analysis/orchestrator.py` |
| Risks, recommendations, report text | `backend/prediction/engine.py`, `backend/recommendation/engine.py`, `backend/reports/generator.py` |
| GPU detection, time math | `backend/hardware/` |
| Dataset cleaning | `backend/cleaning/` |
| Chat answers | `backend/chat/` |
| Add or change an AI provider | `backend/ai/providers/` (and `backend/ai/llm.py`, `backend/ai/provider.py`) |
| Workspace rules, secret scrubbing | `backend/core/security.py`, `backend/core/redaction.py` |
| Database and saved runs | `backend/storage/` (the path is set in `extension/src/services/backendManager.ts`) |
| Reviewing and applying file changes | `backend/editing/file_editor.py`, `extension/src/commands/changeCommands.ts` |
| Python setup and package installing | `extension/src/services/backendManager.ts`, `dependencyManager.ts`, `backend/check_requirements.py` |
| Tests | `backend/tests/`, `extension/tests/suite/` |
| Packaging audit, two-project isolation check | `extension/scripts/verify_vsix.py`, `extension/scripts/e2e_two_projects.py` |
| What goes into the VSIX | `extension/.vscodeignore`, `extension/package.json` |

## Things that look duplicated but are not

- **The VSIX.** Only the root file is tracked. `extension/*.vsix` is a git-ignored build leftover that `postpackage` deletes.
- **`LICENSE`.** The root one is the repository license. `extension/LICENSE` is identical and must stay, because `vsce` packs it into the VSIX.
- **`requirements.txt` and `backend/requirements.txt`.** Different on purpose. The backend file is only what is needed to run. The root file is that plus test and lint tools.

## Where data is stored

- Settings (`llmTrainingAgent.*`): VS Code user settings.
- Database (`llm_training_agent.db`) and `allowed-roots.txt`: VS Code global storage for the extension.
- Pending changes and backups: `<project>/.llm_training_agent/` and `<project>/.llm_training_agent_backups/`.
- Cleaned copies (only until merged or discarded): `<project>/.llm-training-agent/cleaned/`.

## Conventions

- Backend imports are absolute from `backend/`, because the extension runs `main.py` as the entry point.
- The extension never reads its own resources from the open project.
- Every value carries a label: *measured*, *calculated*, *heuristic* or *assumed*. Keep this when adding an analyzer.
- Webview scripts are never written inside TypeScript strings. Put them in `extension/media/`.
