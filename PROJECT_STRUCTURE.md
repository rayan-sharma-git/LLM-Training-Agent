# Project Structure

Where everything lives, and where to look when you want to change something.

The repository is two components — a VS Code extension and a Python backend — plus root documentation. There is no `src/` directory: `backend/` and `extension/` are the two source roots.

```
.
├── README.md                 # What the project is and why it exists
├── USER_MANUAL.md            # Every user-facing feature, end to end
├── INSTALLATION_GUIDE.md     # Installing the VSIX
├── CONTRIBUTING.md           # How to contribute
├── PROJECT_STRUCTURE.md      # This file
├── DELETION.md               # Uninstall and cleanup
├── LICENSE                   # MIT
├── requirements.txt          # Full backend + dev dependency set
├── .github/PULL_REQUEST_TEMPLATE.md  # PR checklist
├── .vscode/                  # Shared editor config (settings, tasks, extensions)
├── llm-training-agent-1.0.0.vsix   # THE INSTALLABLE EXTENSION (root level)
├── backend/                  # Python FastAPI service — all analysis logic
└── extension/                # VS Code extension (TypeScript) — all UI logic
```

## Root files

The root `llm-training-agent-1.0.0.vsix` is the single authoritative distributable: it is the only VSIX tracked by git and the only file users download (see INSTALLATION_GUIDE.md). It is rebuilt from latest source by `npm run package` in `extension/` — the `prepackage` hook regenerates `extension/backend/` from `backend/` and `extension/out/` from `src/`, `vsce` builds a fresh VSIX in `extension/`, and the `postpackage` hook (`extension/scripts/promote-vsix.js`) copies it to the root and deletes the build-directory copy, so the root file can never silently stay stale. A leftover `extension/*.vsix` on disk is always a transient, git-ignored build artifact, never a second distributable.
`requirements.txt` is the full backend + dev set for developers only. `.gitignore` excludes `venv310/`, `node_modules/`, caches, `.db` files, `extension/out/`, `extension/backend/`, and `extension/*.vsix` — the root VSIX stays tracked. `.vscode/` holds shared settings, tasks, and recommended extensions; `.github/` holds the PR checklist.

Docs live only at the repository root — never add copies under `docs/`.

## `backend/` — the analysis service

FastAPI app owning all analysis logic: scanning, statistics, estimates, LLM access, storage.

`main.py` (app factory), `requirements.txt` (runtime deps only), `core/` (config, security, redaction, errors, logging), `api/` (routes, websocket, result mapping), `scanner/` (discovery, framework detection, context), `analyzers/` (dataset, prompt, hyperparameter, model, use case, cost), `analysis/orchestrator.py` (one merged report), `prediction/` + `recommendation/` + `reports/` (risks, prioritised actions, report text), `hardware/` (GPU detection, time/VRAM math), `ai/` (provider interface, helpers, `providers/`, prompt templates), `cleaning/`, `chat/`, `editing/file_editor.py` (PROPOSED → APPROVED → APPLIED), `experiments/`, `storage/` (database, migrations, models), `models/schemas.py` (API contract), `tests/` (pytest suite).

## `extension/` — the VS Code extension

TypeScript: everything the user sees, plus the bundled backend lifecycle.

`package.json` (commands, views, settings, scripts — `package` builds the VSIX and `postpackage` promotes it to the repository root), `.vscodeignore` (what ships in the VSIX), `.vscode/launch.json` (F5 debug), `LICENSE` (MIT — required at this location because `vsce` packages it as `LICENSE.txt`; the identical root `LICENSE` is the repository license), `resources/icon.svg`, `src/extension.ts` (activation), `src/commands/` (analyzer, chat, change commands), `src/services/` (backendManager, backendSync, apiClient, extensionPaths, projectRoot, providers, settings, dependencyManager), `src/views/` (tree views, chat and settings webviews, `webviewHtml.ts` page builders), `media/` (`chat.js` and `settings.js`, the scripts the webviews run — plain files, never inlined in TypeScript), `src/utils/` (report rendering). `tests/suite/` (vitest), `scripts/` (sync-backend.js, promote-vsix.js, verify_vsix.py, verify-webview-wiring.js, e2e_two_projects.py). `backend/`, `out/`, `node_modules/`, and any `*.vsix` in this folder are generated — do not edit or commit them.


## Where to look for what

| I want to change | Look in |
|---|---|
| A view or command | `extension/src/views/`, `extension/src/commands/` |
| A new command, view, or setting | `extension/package.json` (`contributes`), registered in `extension/src/extension.ts` |
| Report rendering, icon | `extension/src/utils/index.ts`, `extension/resources/icon.svg` |
| An endpoint or its shape | `backend/api/routes.py`, `backend/models/schemas.py` |
| Dataset, prompt, hyperparameter, model, cost logic | Matching file in `backend/analyzers/` |
| How sections become one report | `backend/analysis/orchestrator.py` |
| Risks, recommendations, report text | `backend/prediction/engine.py`, `backend/recommendation/engine.py`, `backend/reports/generator.py` |
| GPU detection, time math | `backend/hardware/` |
| Chat, cleaning, experiments | `backend/chat/`, `backend/cleaning/`, `backend/experiments/` |
| Add or change a provider | `backend/ai/providers/` (+ `backend/ai/llm.py`, `backend/ai/provider.py`) |
| Workspace rules, secret scrubbing | `backend/core/security.py`, `backend/core/redaction.py` |
| Database, schema, runs | `backend/storage/` (path set in `extension/src/services/backendManager.ts`) |
| File-edit lifecycle, diff UI | `backend/editing/file_editor.py`, `extension/src/commands/changeCommands.ts` |
| Backend / extension tests | `backend/tests/`, `extension/tests/suite/` |
| Packaging audit, isolation check | `extension/scripts/verify_vsix.py`, `extension/scripts/e2e_two_projects.py` |
| What ships in the VSIX | `extension/.vscodeignore`, `extension/package.json` (`npm run package` → `postpackage` promotes the root VSIX) |
| Where the single distributable lives | Repository root `llm-training-agent-1.0.0.vsix` — never `extension/*.vsix` (transient, git-ignored build artifact) |

## Duplicate-file audit (October 2026)

- **VSIX:** the root `llm-training-agent-1.0.0.vsix` and the on-disk `extension/llm-training-agent-1.0.0.vsix` were byte-identical (same SHA-256, size, and 388-entry archive listing), but only the root file is git-tracked — `git check-ignore` confirms `extension/*.vsix` is ignored and `git ls-files` lists only the root VSIX. The nested file was purely a leftover build copy, so it is not committed; the `postpackage` hook now deletes it on every build.
- **LICENSE:** root `LICENSE` and `extension/LICENSE` are byte-identical (same SHA-256) and both are retained deliberately: the root file is the repository license, while the nested file is required at its location because `vsce` packages it into the VSIX as `extension/LICENSE.txt` (verified in the archive). Neither is redundant.
- **`requirements.txt` vs `backend/requirements.txt`:** intentionally different by design — the root file is the full developer set (runtime + pytest/ruff/black/mypy/PyYAML), the backend file is the runtime-only subset the extension checks at startup. Both retained.

Runtime data: settings (`llmTrainingAgent.*`) in VS Code user settings; database (`llm_training_agent.db`) and `allowed-roots.txt` in VS Code global storage; pending changes and backups in `<project>/.llm_training_agent/` and `<project>/.llm_training_agent_backups/`; cleaned output in `<project>/.llm-training-agent/cleaned/`.

## Conventions

- **Backend imports are absolute from `backend/`.** The extension runs `main.py` as an entry point.
- **The extension never reads resources from the open project.** Extension files resolve from the install directory only.
- **Values carry provenance** (*measured*, *calculated*, *heuristic*, *assumed*). Preserve this when adding an analyzer.
