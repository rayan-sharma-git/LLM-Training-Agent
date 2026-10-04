# Project Structure

Where everything lives, and where to look when you want to change something.

The repository is two components — a VS Code extension and a Python backend — plus the
root documentation. There is no `src/` directory: `backend/` and `extension/` are the
two source roots, and that is the real architecture.

```
.
├── README.md                 # What the project is and why it exists
├── USER_MANUAL.md            # Every user-facing feature, end to end
├── INSTALLATION_GUIDE.md     # Installing the VSIX
├── CONTRIBUTING.md           # How to contribute
├── PROJECT_STRUCTURE.md      # This file
├── DELETION.md               # Safe uninstall and cleanup
├── LICENSE                   # MIT
├── requirements.txt          # Full backend + dev dependency set
├── .gitignore
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md  # Checklist shown when opening a pull request
├── .vscode/
│   ├── settings.json         # Project config: venv310 interpreter, pytest paths, excludes
│   ├── tasks.json            # Build / test / package tasks bound to venv310 + npm
│   └── extensions.json       # Recommended editor extensions
│
├── llm-training-agent-1.0.0.vsix   # THE INSTALLABLE EXTENSION (built artifact, root level)
│
├── backend/                  # Python FastAPI service — all analysis logic
└── extension/                # VS Code extension (TypeScript) — all UI logic
```

---

## Root files

| File | Purpose |
|---|---|
| `requirements.txt` | Full backend dependency set including tests and linting. **Developers only** — normal extension users do not need it. |
| `llm-training-agent-1.0.0.vsix` | The packaged, installable extension. Downloaded directly from the repository root; see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md). |
| `.gitignore` | Excludes `venv310/` (whole directory), `node_modules/`, Python caches and `.db` files, `extension/out/`, `extension/backend/` (generated), and the working VSIX inside `extension/`. The root VSIX is deliberately **not** ignored, because it is meant to be distributed. Editor-local state under `.vscode/` and `extension/.vscode/` is ignored, but the shared configuration `.vscode/settings.json`, `.vscode/tasks.json`, `.vscode/extensions.json` and `extension/.vscode/launch.json` are tracked. |
| `.vscode/settings.json` | Project-level editor config: points the Python interpreter at `venv310\Scripts\python.exe`, sets the pytest path to `backend/tests`, and keeps `venv310`, `node_modules` and the generated `extension/backend` + `extension/out` out of the file watcher and search index. |
| `.vscode/tasks.json` | Build/test/package tasks: install deps into `venv310`, run backend `pytest`, run `npm run compile` / `npm test` / `npm run package` in `extension/`. |
| `.vscode/extensions.json` | Recommended extensions for this repository. |
| `.github/PULL_REQUEST_TEMPLATE.md` | The checklist GitHub pre-fills on every new pull request. It repeats the invariants from [CONTRIBUTING.md](CONTRIBUTING.md) — tests, no architecture changes, no fabricated values, VSIX rebuilt if packaging changed, root documentation updated. |
| `LICENSE` | MIT. |

The six documentation files live **only** at the repository root. Do not add copies
under `docs/` or inside any component.

---

## `backend/` — the analysis service

A FastAPI application. It owns every piece of real logic: scanning files, computing
statistics, estimating time and cost, talking to LLM providers, and persisting results.

```
backend/
├── main.py                   # App factory, lifespan, router mounting
├── requirements.txt          # RUNTIME dependencies only (what the extension checks)
├── core/                     # Cross-cutting infrastructure
│   ├── config.py             # Pydantic settings, provider/model/key resolution
│   ├── runtime_config.py     # In-memory provider/API-key overrides from the extension
│   ├── security.py           # Workspace boundaries, path traversal defence, bounds
│   ├── redaction.py          # Secret scrubbing for logs, errors and LLM prompts
│   ├── errors.py             # Typed error hierarchy -> HTTP status codes
│   └── logging.py            # Logging setup
├── api/
│   ├── routes.py             # All REST endpoints
│   ├── websocket.py          # /ws/analysis — streaming analysis progress
│   └── result_mapping.py     # Safe reconstruction of typed results (shared by HTTP + WS)
├── scanner/                  # Project discovery
│   ├── scanner.py            # Finds datasets, prompts, configs, train/eval scripts
│   ├── framework_detector.py # Identifies the training framework
│   └── context_builder.py    # Builds the ProjectContext passed to analyzers
├── analyzers/                # Deterministic analysis of measurable facts
│   ├── dataset_analyzer.py       # Duplicates, missing fields, consistency, quality
│   ├── prompt_analyzer.py        # Clarity, ambiguity, formatting, placeholders
│   ├── hyperparameter_analyzer.py # Learning rate / batch / epochs vs. dataset size
│   ├── model_advisor.py          # Model fit, alternatives, VRAM suitability
│   ├── use_case_detector.py      # What task the project is actually for
│   └── cost_estimator.py         # Local vs. hosted cost, budget options
├── analysis/
│   └── orchestrator.py       # Composes every analyzer output into ONE report
├── prediction/
│   └── engine.py             # Training-outcome signals, risks, provenance labels
├── hardware/
│   ├── gpu_detector.py       # Real GPU detection (torch / nvidia-smi / env)
│   └── gpu_time_estimator.py # FLOPs/throughput math, VRAM estimate, calibration
├── ai/                       # LLM access
│   ├── provider.py           # AIProvider interface (chat_completion, health, models)
│   ├── llm.py                # LLMHelper: prompt loading + graceful degradation
│   ├── providers/            # ollama, openai, anthropic, gemini, deepseek,
│   │                         #   cohere, openrouter, openai_compatible
│   └── prompts/              # Prompt templates (.md), bundled into the VSIX
├── cleaning/
│   ├── dataset_cleaner.py    # Chunked LLM-assisted dataset cleaning
│   └── dataset_io.py         # Dataset discovery and read/write helpers
├── chat/
│   ├── engine.py             # Context-aware chat over real analysis results
│   └── store.py              # Per-project conversation history
├── recommendation/
│   └── engine.py             # Prioritised recommendations with reasoning
├── reports/
│   └── generator.py          # Engineering report, readiness score, action plan
├── editing/
│   └── file_editor.py        # ChangeStore: PROPOSED -> APPROVED -> APPLIED lifecycle
├── experiments/
│   └── service.py            # Experiment CRUD and comparison
├── storage/
│   ├── database.py           # Async SQLAlchemy engine, SQLite
│   ├── migrations.py         # Versioned schema
│   ├── models.py             # ORM models
│   ├── repositories.py       # Data access
│   └── analysis_store.py     # Analysis run lifecycle and status
├── models/
│   └── schemas.py            # Pydantic request/response models (the API contract)
└── tests/                    # pytest suite
```

---

## `extension/` — the VS Code extension

TypeScript. It owns everything the user sees and touches, plus the lifecycle of the
bundled Python backend.

```
extension/
├── package.json              # Extension manifest: commands, views, settings, scripts
├── tsconfig.json             # TypeScript configuration
├── vitest.config.ts          # Test configuration
├── .vscodeignore             # Controls what is and is not shipped inside the VSIX
├── .vscode/
│   └── launch.json           # F5 debug config: runs the extension in an Extension Development Host
├── resources/
│   └── icon.svg              # Activity bar icon
├── src/
│   ├── extension.ts          # activate() — registers views, commands, services
│   ├── commands/
│   │   ├── analyzerCommands.ts  # Analyze Project, dependency install prompt
│   │   ├── chatCommands.ts      # Open Chat
│   │   ├── changeCommands.ts    # View Changes: diff, approve, apply, rollback
│   │   └── index.ts
│   ├── services/
│   │   ├── backendManager.ts     # Starts/stops the bundled Python backend
│   │   ├── apiClient.ts          # Typed HTTP client for the backend
│   │   ├── extensionPaths.ts     # Path policy: extension dir, never the project
│   │   ├── projectRoot.ts        # Resolves the open workspace root per call
│   │   ├── settings.ts           # Reads/writes user-level settings
│   │   └── dependencyManager.ts  # Checks (never silently installs) Python deps
│   ├── views/
│   │   ├── simpleTreeView.ts         # Overview and Reports tree views
│   │   ├── chatWebviewProvider.ts    # Chat panel
│   │   ├── settingsWebviewProvider.ts# Provider / model / API key UI
│   │   └── viewIds.ts                # View identifier constants
│   └── utils/index.ts        # Report HTML rendering, error formatting
├── tests/suite/              # vitest suite
├── scripts/
│   ├── sync-backend.js       # Copies backend/ into the extension for packaging
│   ├── verify_vsix.py        # Audits the built VSIX (secrets, caches, path assumptions)
│   └── e2e_two_projects.py   # Installs the VSIX and proves project isolation
├── backend/                  # GENERATED by sync-backend.js — do not edit
├── out/                      # GENERATED by tsc — do not edit
├── node_modules/             # GENERATED by npm
└── llm-training-agent-1.0.0.vsix   # Build output; the released copy is at the repo root
```

### Generated directories — do not edit

`extension/backend/`, `extension/out/` and `extension/node_modules/` are build
artifacts.

- `extension/backend/` is overwritten from `../backend/` on every `npm run package`.
  Edit the real backend in `backend/`; your changes will be copied.
- `extension/out/` is TypeScript compiler output from `src/`.

Both are listed in `.gitignore`. Editing them is the fastest way to lose work.

---

## Where to look for what — "I want to fix X, look in Y"

### Extension UI

| I want to change | Look in |
|---|---|
| A view (tree, chat, settings) | `extension/src/views/` |
| A command's behaviour | `extension/src/commands/` |
| What appears in the sidebar | `extension/src/views/viewIds.ts`, then `extension/src/extension.ts` |
| A new command, view or setting | `extension/package.json` (`contributes`), then register it in `extension/src/extension.ts` |
| Report HTML rendering | `extension/src/utils/index.ts` |
| The activity bar icon | `extension/resources/icon.svg` |

### Backend API

| I want to change | Look in |
|---|---|
| A REST endpoint | `backend/api/routes.py` |
| The request/response shape | `backend/models/schemas.py` |
| Progress streaming | `backend/api/websocket.py` |
| Typed-result fallbacks | `backend/api/result_mapping.py` |
| How an error becomes an HTTP status | `backend/core/errors.py`, mapped in `backend/api/routes.py` |
| CORS, app startup, lifespan | `backend/main.py` |

### AI providers

| I want to change | Look in |
|---|---|
| Add a provider | A new file in `backend/ai/providers/`, then register it in `backend/ai/providers/__init__.py` |
| How a provider is called | `backend/ai/llm.py` (the `LLMHelper` degrades gracefully when none is available) |
| The provider interface | `backend/ai/provider.py` |
| Which provider/model is active | `backend/core/config.py` (`get_active_provider`, `get_active_model`) |
| API keys | `backend/core/runtime_config.py` (in memory only) |
| Prompt wording | `backend/ai/prompts/` (`.md` files, bundled into the VSIX) |

### Analysis logic

| I want to change | Look in |
|---|---|
| Dataset statistics | `backend/analyzers/dataset_analyzer.py` |
| Prompt analysis | `backend/analyzers/prompt_analyzer.py` |
| Hyperparameter assessment | `backend/analyzers/hyperparameter_analyzer.py` |
| Model selection advice | `backend/analyzers/model_advisor.py` (bundled model database) |
| Use-case inference | `backend/analyzers/use_case_detector.py` |
| Cost estimates | `backend/analyzers/cost_estimator.py` |
| How sections are combined into one report | `backend/analysis/orchestrator.py` |
| Training-outcome predictions and risks | `backend/prediction/engine.py` |
| GPU detection | `backend/hardware/gpu_detector.py` |
| Training-time and VRAM maths | `backend/hardware/gpu_time_estimator.py` |
| Recommendations | `backend/recommendation/engine.py` |
| What files get scanned | `backend/scanner/scanner.py` |
| Report generation | `backend/reports/generator.py` |
| Dataset cleaning | `backend/cleaning/dataset_cleaner.py` |
| Chat answers | `backend/chat/engine.py` |

### Security and privacy

| I want to change | Look in |
|---|---|
| Which directories the backend may touch | `backend/core/security.py` (`allowed_roots`, `resolve_workspace_root`) |
| How the extension publishes the open roots | `extension/src/services/extensionPaths.ts` (`publishAllowedRoots`) |
| Secret scrubbing | `backend/core/redaction.py` |
| Path policy (extension dir vs. project) | `extension/src/services/extensionPaths.ts` |
| Requiring a workspace root | `extension/src/services/projectRoot.ts` |

### Storage

| I want to change | Look in |
|---|---|
| Database engine / connection | `backend/storage/database.py` |
| Schema and migrations | `backend/storage/migrations.py` |
| ORM models | `backend/storage/models.py` |
| Analysis run lifecycle | `backend/storage/analysis_store.py` |
| Experiment CRUD | `backend/experiments/service.py` |
| Where the database file lives | Set by `extension/src/services/backendManager.ts` (VS Code global storage) |

### File editing and approval

| I want to change | Look in |
|---|---|
| The change lifecycle / safety rules | `backend/editing/file_editor.py` |
| Propose / approve / reject / apply / rollback | `backend/api/routes.py` (the `/files/...` routes) |
| The diff-and-approve UI | `extension/src/commands/changeCommands.ts` |

### Tests

| I want to change | Look in |
|---|---|
| Backend behaviour tests | `backend/tests/` (run: `cd backend && python -m pytest tests -q`) |
| Extension unit tests | `extension/tests/suite/` (run: `cd extension && npm test`) |
| Packaging correctness | `extension/scripts/verify_vsix.py` |
| Cross-project isolation | `extension/scripts/e2e_two_projects.py` |
| Shared test fixtures and setup | `backend/tests/conftest.py` |

### Packaging

| I want to change | Look in |
|---|---|
| Build the VSIX | `extension/package.json` → `npm run package` |
| How the backend is bundled | `extension/scripts/sync-backend.js` |
| What ships and what is excluded | `extension/.vscodeignore` |
| What the packaging audit enforces | `extension/scripts/verify_vsix.py` |
| Version, publisher, commands, settings | `extension/package.json` |
| The root VSIX | Copied from `extension/llm-training-agent-1.0.0.vsix` to the repository root |

### Configuration and runtime data locations

| What | Where |
|---|---|
| Extension settings | VS Code user settings, `llmTrainingAgent.*` — see `extension/package.json` → `contributes.configuration` |
| Backend environment settings | `backend/core/config.py`; overridable via `.env` in `backend/` |
| Runtime provider / API keys | `backend/core/runtime_config.py`, in memory only, never written to disk |
| SQLite database | VS Code global storage → `llm_training_agent.db` (path set in `extension/src/services/backendManager.ts`) |
| Authorized project roots | VS Code global storage → `allowed-roots.txt` |
| Pending file changes | `<project>/.llm_training_agent/pending_changes/` |
| Change backups | `<project>/.llm_training_agent_backups/` |
| Cleaned dataset output | `<project>/.llm-training-agent/cleaned/` |
| Local virtualenv | `venv310/` — git-ignored, not part of the project |

---

## Conventions

- **Backend imports are absolute from `backend/`.** `from core.config import ...`, not
  relative imports, because the extension runs `main.py` as an entry point.
- **Every backend package has an `__init__.py`.**
- **The extension never reads resources from the open project.** Extension-owned
  resources resolve from `context.extensionUri` only. This invariant is enforced by
  `extension/tests/suite/extensionPaths.test.ts` and by the VSIX audit.
- **Values carry provenance.** Analysis output is labelled *measured*, *calculated*,
  *heuristic* or *assumed*. Preserve this when adding an analyzer.
