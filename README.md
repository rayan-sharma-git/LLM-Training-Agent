# LLM Training Agent

A VS Code extension that reads your actual LLM project — datasets, prompts, training configuration, hardware — and tells you whether fine-tuning it is worth it, what it would cost, how long it would take, and what to fix first.

The question it answers is the one that is expensive to answer by hand: *"I have this project and this data. Should I fine-tune, and if so, exactly how?"*

You install the extension from a single file, open your project folder in VS Code, and work with it from a dedicated panel in the activity bar. No separate app to run and no service to deploy.

## Basics

### What this is

LLM Training Agent is a VS Code extension for people preparing or considering an LLM fine-tuning run. Once installed, it lives in the VS Code activity bar with four views — Overview, Chat, Reports, and Settings — and commands such as Analyze Project, Analyze Dataset, Open Chat, View Report, Configure AI Provider, and View Changes.

The main workflow is simple: open the folder containing your training project, run Analyze Project, and read the resulting report in the Chat panel. You can then ask follow-up questions, save and revisit reports, configure which AI provider gives interpretive advice, and review any file changes the agent proposes through a diff before anything is applied.

### Who it is for

It is for engineers, researchers, students, and hobbyists who have (or plan to have) a fine-tuning project — some data files, prompts, or training configs — and need a grounded second opinion before spending GPU hours. You do not need to be a training expert to read the report, but the report is written for someone who will actually run the training and wants specifics, not slogans.

### The problem it solves

Deciding whether to fine-tune today means stitching together scattered facts: row counts and duplicate rates from the dataset, hyperparameters from whatever config file happens to be lying around, a GPU you may or may not have, and pricing pages you have to keep refreshing. The engineer doing this is usually guessing, and the guess gets expensive precisely when it is least checked.

Existing options tend to fall into two camps. Dashboards show metrics without context — a row count is not a verdict. General-purpose LLM chat is fluent but has no measurements of your files, so it can sound confident about things it never checked. This extension takes the opposite stance: measure first, reason second. Every number in the report comes from your project; the AI interprets those numbers rather than inventing them.

### How the answer is produced

The extension scans the folder you have open — datasets, prompt files, training configs, and training or evaluation scripts — computes concrete statistics from what it finds, and combines those measurements with engineering estimates and AI interpretation into a single report. Each value is labelled by provenance — *measured*, *calculated*, *heuristic*, or *assumed* — so you always know which numbers to trust and which to verify. Where something genuinely cannot be determined, the system says so instead of producing a plausible-looking number. It deliberately does not fabricate accuracy scores.

If no AI provider is configured, the measurable half of the analysis still works: you get the statistics and rule-based guidance, just without the LLM-written interpretation.

### Main features

Running Analyze Project produces one report covering use-case detection with a confidence score, dataset size and quality (including duplicates, missing fields, and consistency), prompt clarity and formatting issues, hyperparameter review against your actual data, model advice matched to your task and available VRAM, real GPU and CUDA detection, training-time and VRAM estimates with stated assumptions, local-versus-hosted cost estimates with budget options, a risk register with mitigations, prioritised recommendations with reasoning, and a concrete next experiment with a success criterion.

Beyond the report, you can keep the conversation going in Chat, generate and revisit saved reports, switch AI providers or models from Settings (Ollama works locally with no API key), and review proposed file edits through the View Changes diff workflow, where nothing is applied until you approve it. Dataset cleaning and experiment tracking are available through the chat agent and the backend API.

### How it helps your fine-tuning work

Each feature maps to a decision you would otherwise make by gut feel. Dataset analysis tells you whether you have enough genuinely distinct examples before tuning anything. Prompt analysis catches ambiguity and formatting problems that silently degrade training. Hyperparameter review checks learning rate, batch size, and epochs against your data volume. Model advice narrows the base-model choice to what fits your task and your hardware. Time, VRAM, and cost estimates turn "can I afford this?" into arithmetic with visible assumptions. The risk register and the recommended next experiment give you an ordered list of what to fix first and how to tell whether the fix worked.

### Installing and learning more

The installable extension is the single `llm-training-agent-1.0.0.vsix` file at the repository root. You need VS Code 1.85.0 or newer; no API key is required to start because the default provider is local. For step-by-step installation, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md). For detailed instructions on every feature, see [USER_MANUAL.md](USER_MANUAL.md).

## Technical Details

### How it works: measure first, reason second

The system deliberately splits work by what each method is good at. Deterministic engineering handles everything measurable: dataset statistics (sample counts, exact and near-duplicate ratios, missing-field rates, length distributions, formatting and language consistency) are computed directly from the files; training-time and VRAM estimates come from a FLOPs/throughput model over the detected GPU; cost estimates come from arithmetic over real token counts. These values are reproducible — run the analysis twice on the same project and you get the same answer.

LLM reasoning handles everything interpretive: turning measurements into prioritised judgement, reading prompt files for ambiguity, weighing trade-offs, and explaining why a recommendation was made. The LLM is given real measured context, and every output value carries a provenance label (*measured*, *calculated*, *heuristic*, *assumed*). No accuracy or benchmark scores are fabricated; unknown results are reported as unknown. With no provider configured, the pipeline degrades to measurements plus rule-based recommendations instead of failing.

### Technology stack

- **Backend:** Python 3.10.6, FastAPI, Pydantic v2 (plus pydantic-settings and python-dotenv), SQLAlchemy 2 (async) with aiosqlite over SQLite, httpx for provider REST calls, and optional `google-generativeai` / `cohere` SDKs. Real YAML parsing via PyYAML when present, with a regex fallback otherwise.
- **Extension:** TypeScript, VS Code Extension API, axios; `vitest` for unit tests and `tsc --noEmit` as lint.
- **Packaging:** `vsce` produces the VSIX; `extension/scripts/sync-backend.js` bundles the Python backend into it.
- **Tests:** pytest (backend, `backend/tests/`), vitest (extension, `extension/tests/suite/`), a VSIX packaging audit (`extension/scripts/verify_vsix.py`), and a two-project isolation check (`extension/scripts/e2e_two_projects.py`).

### Architecture and components

| Component | Path | Responsibility |
|---|---|---|
| **VS Code extension** | `extension/` | UI, commands, bundled-backend lifecycle, VS Code SecretStorage |
| **Python backend** | `backend/` | FastAPI service: scanning, analysis, estimation, storage, LLM access |

The extension bundles the entire Python backend inside the VSIX. On first use it starts that backend as a local subprocess on `127.0.0.1:8000` and talks to it over HTTP, plus a WebSocket (`/ws/analysis`) for streaming analysis progress. Nothing is uploaded anywhere; the backend only ever reads the folder you have open in VS Code.

The extension resolves its backend and interpreter only from the extension install directory, never from your project — a project that happens to contain a `backend/` folder cannot replace the agent's backend, and switching projects never mixes state. Extension settings (`llmTrainingAgent.backendUrl`, `llmTrainingAgent.pythonPath`, `llmTrainingAgent.provider`, `llmTrainingAgent.model`) are user-level and are never stored inside a project.

### Analysis and recommendation pipeline

Project discovery lives in `backend/scanner/` (`scanner.py` finds datasets in `.jsonl`/`.json`/`.csv`/`.tsv`, prompt files, training configs in `.yaml`/`.yml`/`.toml`/`.json`, and training or evaluation scripts; `framework_detector.py` identifies the training framework; `context_builder.py` builds the `ProjectContext` passed to the analyzers). Deterministic analyzers in `backend/analyzers/` cover datasets (duplicates, missing fields, consistency, quality), prompts (clarity, ambiguity, formatting, placeholders), hyperparameters (learning rate, batch size, epochs versus dataset size), model advice (fit, alternatives, VRAM suitability), use-case detection, and cost estimation (local versus hosted, budget options). `backend/analysis/orchestrator.py` composes every analyzer output into one report; if one section fails, the rest still runs and the report is marked partial with the failed sections listed.

Downstream of the analyzers, `backend/prediction/engine.py` produces training-outcome signals, risks, and provenance labels; `backend/recommendation/engine.py` prioritises recommendations with reasoning; and `backend/reports/generator.py` produces the engineering report with a readiness score and action plan. Hardware estimation is split between `backend/hardware/gpu_detector.py` (real GPU detection via PyTorch when installed, otherwise `nvidia-smi`, otherwise environment variables) and `backend/hardware/gpu_time_estimator.py` (FLOPs/throughput math, VRAM estimates, and a calibrated benchmark mode alongside the quick published-TFLOPs mode). Dataset cleaning (`backend/cleaning/`) is chunked and LLM-assisted and is reachable through the chat agent and the API; cleaned output goes to `<project>/.llm-training-agent/cleaned/`.

### AI / LLM provider system

LLM access sits behind the `AIProvider` interface (`backend/ai/provider.py`: `chat_completion`, health checks, model listing), with `backend/ai/llm.py` (`LLMHelper`) loading prompt templates and degrading gracefully when no provider is available. Eight providers are implemented in `backend/ai/providers/`: Ollama (the default — local models, no API key, the only fully offline option), OpenAI, Anthropic, Google Gemini, DeepSeek, Cohere, OpenRouter, and any OpenAI-compatible server such as LM Studio. Prompt templates in `backend/ai/prompts/` (Markdown) are bundled into the VSIX.

Provider and model choice plus API keys are configured from the extension Settings view (dropdown or free-text model field, Save, Test Connection, Remove Key, Refresh Models). Keys are held in VS Code SecretStorage — never in a settings file or your project — sent to the local backend over `127.0.0.1` at runtime, kept only in memory (`backend/core/runtime_config.py`, never written to disk), and scrubbed from every log line, error message, and LLM prompt by `backend/core/redaction.py` before it leaves the process.

### Chat, reports, experiments, and storage

Context-aware chat (`backend/chat/engine.py`) answers follow-up questions over real analysis results, with per-project conversation history (`backend/chat/store.py`). Reports are generated by `backend/reports/generator.py` and surfaced in the extension Reports tree view alongside the Chat webview. Experiment tracking (`backend/experiments/service.py`) records training runs — model, hyperparameters, dataset, actual versus estimated results, and status including failures — and compares them; it is exposed through the `/api/v1/experiments` REST endpoints and is not yet a dedicated UI view.

Persistence is async SQLAlchemy over a local SQLite database (`backend/storage/`: engine, versioned migrations, ORM models, repositories, and the analysis-run lifecycle in `analysis_store.py`). The database file (`llm_training_agent.db`) and the authorized-roots list live in VS Code global storage, with the path set by `extension/src/services/backendManager.ts` — state is per user and per project, not inside your project folder. The API contract is defined by Pydantic request/response models in `backend/models/schemas.py`, served under `/api/v1` (`backend/api/routes.py`), with typed-error mapping (`backend/core/errors.py`) and CORS configured for the local client.

### File editing and approval workflow

Proposed edits follow a `PROPOSED -> APPROVED -> APPLIED` lifecycle managed by `backend/editing/file_editor.py` (`ChangeStore`), exposed through the `/files/...` routes (propose, approve, reject, apply, rollback) and reviewed in the extension through a diff UI (`extension/src/commands/changeCommands.ts`). Nothing is applied until approved. Pending changes stage under `<project>/.llm_training_agent/pending_changes/` and backups under `<project>/.llm_training_agent_backups/`, so every applied change can be rolled back.

### Progress communication

Long analyses stream progress over the WebSocket endpoint (`backend/api/websocket.py`, `/ws/analysis`), reconstructed into typed results shared with the HTTP path (`backend/api/result_mapping.py`). The extension shows a cancellable progress notification while analysis runs; partial results are reported as partial, never silently filled in.

### Security considerations

The backend enforces workspace boundaries: it only reads the folder currently open in VS Code and refuses any other directory, with path-traversal defence and bounds in `backend/core/security.py` and root resolution per call (`extension/src/services/projectRoot.ts`, `publishAllowedRoots` in `extensionPaths.ts`). Extension-owned resources resolve from `context.extensionUri` only — enforced by unit tests and the VSIX audit. Credentials are scrubbed from logs and errors (tracebacks are never shown to users), and the backend binds to loopback only.

### Testing

Backend behaviour is covered by the pytest suite in `backend/tests/` (run `cd backend && python -m pytest tests -q`); extension logic by vitest in `extension/tests/suite/` (`cd extension && npm test`, plus `npm run lint` for `tsc --noEmit`). `extension/scripts/verify_vsix.py` audits the built VSIX and fails on shipped secrets, caches, test files, or repository-relative path assumptions. `extension/scripts/e2e_two_projects.py` installs the VSIX and proves cross-project isolation.

### Packaging and distribution

`npm run package` (after `package:backend` copies `backend/` into the extension and `compile` builds TypeScript) produces `llm-training-agent-1.0.0.vsix` via `vsce`; what ships is controlled by `extension/.vscodeignore`. The released copy users download lives at the repository root next to this README — exactly one root VSIX; the `extension/*.vsix` build copy is git-ignored and removed after promotion. Rebuilding and re-auditing is required whenever anything inside the VSIX changes (see [CONTRIBUTING.md](CONTRIBUTING.md)).

### Development setup

Backend development uses Python 3.10.6 with the full dependency set in the root `requirements.txt` (the runtime-only subset the extension checks at startup is `backend/requirements.txt` — keep the two in sync). Extension development uses Node.js with `npm install` inside `extension/`. Shared editor configuration lives in the root `.vscode/` (`settings.json` points the interpreter at `venv310\Scripts\python.exe` and pytest at `backend/tests`; `tasks.json` wires Build/Test/Package tasks; `extensions.json` lists recommended editor extensions), and F5 debugging of the extension host is configured in `extension/.vscode/launch.json`. Contributions follow the GitHub flow with the checklist in `.github/PULL_REQUEST_TEMPLATE.md`.

### Project structure

[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) is the contributor-oriented map of the repository: where the extension and backend code live, what the major folders contain, which files own which responsibility, and where to look when changing a particular part of the project. It also documents generated directories (`extension/backend/`, `extension/out/`, `node_modules/`) that must not be edited directly. This README does not duplicate it — consult it before modifying code.

### Documentation

| Document | What it covers |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | Every user-facing feature, end to end |
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | Installing the VSIX |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Repository layout and where to change what |
| [DELETION.md](DELETION.md) | Safe uninstall and cleanup |

### Contributing

Contributions are welcome. Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) first, then follow [CONTRIBUTING.md](CONTRIBUTING.md): focused single-concern pull requests against `main`, existing tests kept green, no unnecessary dependencies, no architecture redesigns, no fabricated values, and documentation updated in the same PR. The pull-request template (`.github/PULL_REQUEST_TEMPLATE.md`) repeats these invariants as a checklist.

### License

MIT. See [LICENSE](LICENSE).
