# User Manual

Everything the LLM Training Agent extension can do, as it actually behaves.

For installation, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md). For uninstalling,
see [DELETION.md](DELETION.md).

---

## Contents

1. [Opening the panel](#1-opening-the-panel)
2. [Analyze Project — the main workflow](#2-analyze-project--the-main-workflow)
3. [What the report contains](#3-what-the-report-contains)
4. [Analyze Dataset](#4-analyze-dataset)
5. [Dataset cleaning](#5-dataset-cleaning)
6. [Chat](#6-chat)
7. [Reports](#7-reports)
8. [Configuring an AI provider](#8-configuring-an-ai-provider)
9. [Experiments](#9-experiments)
10. [View Changes — reviewing proposed edits](#10-view-changes--reviewing-proposed-edits)
11. [Settings](#11-settings)
12. [Where your data is stored](#12-where-your-data-is-stored)
13. [Errors and troubleshooting](#13-errors-and-troubleshooting)
14. [Known limitations](#14-known-limitations)

---

## 1. Opening the panel

Click the **LLM Training Agent** icon in the VS Code activity bar (left sidebar). It
contains four views:

| View | What it is |
|---|---|
| **Overview** | Project status, active provider and model |
| **Chat** | The full analysis and your conversation with the agent |
| **Reports** | Generated report entries |
| **Settings** | Provider, model and API key configuration |

The extension activates when you open one of these views or run one of its commands —
not on every VS Code startup.

**The backend starts lazily.** The first time you use a feature, the extension starts
its bundled Python backend and waits for it to become healthy. Later features in the
same window reuse that process. The backend shuts down when VS Code closes.

---

## 2. Analyze Project — the main workflow

**Where:** the ⚡ *Analyze Project* button at the top of the **Overview** view, the
command palette (`Ctrl+Shift+P` → *LLM Training Agent: Analyze Project*), or the
*Analyze Project* entry in the **Reports** view.

**Input:** the folder currently open in VS Code. Nothing else — the scanner finds the
datasets, prompts and configs itself.

**What happens:**

1. The extension confirms a folder is open and starts the backend.
2. The backend scans the project for datasets (`.jsonl`, `.json`, `.csv`, `.tsv`),
   prompt files, training configs (`.yaml`, `.yml`, `.toml`, `.json`) and training or
   evaluation scripts.
3. Every analyzer runs against what was found: dataset, prompt, hyperparameters, model,
   use case, cost, GPU.
4. The orchestrator assembles one report and posts it to **Chat**.

**Output:** a complete Markdown analysis in the Chat panel, plus a saved run you can
reopen later.

**Workflow:** open your project folder → ⚡ Analyze Project → read the report in Chat →
ask follow-ups in Chat → fix the top recommendation → re-run Analyze.

**Notes:**

- Analysis can take a while on large datasets; a progress notification is shown and can
  be cancelled.
- If one section fails, the rest still runs. The report is marked *partial* and the
  failed sections are listed — nothing is silently replaced with a made-up value.
- The backend only reads the folder that is open. It refuses any other directory, even
  if asked.

---

## 3. What the report contains

Each section states its **provenance** so you know how much to trust it:
*measured* (read from your files), *calculated* (derived by formula), *heuristic*
(engineering rule of thumb), or *assumed*.

| Section | What it reports |
|---|---|
| **Use case** | The task your project appears to target, with a confidence score and the evidence behind it |
| **Dataset** | Sample count, exact-duplicate and near-duplicate ratios, missing-field rate, length distribution, formatting/language/instruction consistency, quality score, findings |
| **Prompts** | Per-prompt clarity, ambiguity, formatting scores, and any placeholders that are used but never defined |
| **Model** | The base model found in your config, whether it fits your task and VRAM, and alternatives worth considering |
| **Hyperparameters** | Learning rate, batch size, epochs read from your config, assessed against your actual dataset size, with an overfitting risk |
| **Hardware** | GPUs actually detected on this machine, VRAM, CUDA version |
| **Training time** | Estimated duration and VRAM need, with a range, confidence, and every assumption listed |
| **Cost** | Local-compute cost versus hosted-API cost, with budget options |
| **Risks** | Overfitting, hallucination, no evaluation harness, no local GPU, ambiguous use case — each with why, severity, basis and mitigation |
| **Recommendations** | Prioritised actions, each with its reasoning |

---

## 4. Analyze Dataset

**Where:** command palette → *LLM Training Agent: Analyze Dataset*.

**Input:** a dataset directory you pick in a file dialog.

**Output:** the dataset analysis described above, for that directory.

**Limitations:** this runs the dataset pipeline on the chosen directory rather than the
full end-to-end project analysis, so it does not include model, hardware or cost
sections. Use **Analyze Project** for the complete picture.

---

## 5. Dataset cleaning

**Where:** backend API (`POST /api/v1/dataset/clean`). There is no dedicated button in
the current UI; the chat agent uses it when you ask it to clean a dataset.

**What it does:** reads your dataset in chunks, asks the configured LLM to clean each
chunk, and writes the results to a cleaned output directory alongside a manifest.

**Behaviour worth knowing:**

- Chunked, so an entire dataset is never sent in one request.
- Every record of every file is preserved. If a chunk fails validation, its original
  records are kept rather than dropped — you never lose data to a failed call.
- Output preserves the relative path and format of each source file.
- Requires a working LLM provider. Without one, nothing is written.

---

## 6. Chat

**Where:** the **Chat** view in the sidebar.

**What it does:** holds the analysis report and lets you ask follow-up questions. The
agent answers with the real analysis results for your project in context, not from
general knowledge alone.

**Actions:** the 💬 *Open Chat* button in the Chat view header, or the command palette.

**Notes:**

- History is per-project. Switching projects starts a clean context.
- A response includes a confidence level and references to the underlying results.
- If no provider is configured, chat reports that clearly rather than pretending.

---

## 7. Reports

**Where:** the **Reports** view, or *LLM Training Agent: View Report*.

**What it does:** fetches the latest saved report for the current project and renders it
as a full HTML page in an editor tab.

**Inputs:** none — it always shows the current project's most recent report.

**Notes:** if no analysis has been run for this project, it says so instead of showing
an empty page.

---

## 8. Configuring an AI provider

**Where:** the **Settings** view, or ⚙ *Configure AI Provider* from the command palette.

### Supported providers

---

## 10. View Changes — reviewing proposed edits

**Where:** the ⇄ *View Changes* button in the **Overview** view, or the command palette.

**What it does:** lists file modifications the agent has proposed, and walks you through
reviewing and applying them.

**Workflow:**

1. Open **View Changes**. A picker lists pending changes with status, operation and
   line counts.
2. Pick one. It opens in VS Code's native diff editor: current file on the left,
   proposed version on the right.
3. Choose an action:
   - **Approve & Apply** — writes the change to disk and verifies the result.
   - **Reject** — discards it; the file is not modified.
   - **Dismiss** — look, decide later.

**Safety guarantees:**

- Nothing is written without explicit approval. The backend refuses to apply a proposal
  that was not approved.
- If the file changed on disk since the proposal was created, the change is flagged as
  **conflicted** and will not be applied over your newer edits.
- A backup of the previous content is kept, and any applied change can be rolled back.
- Proposals expire after 24 hours and can no longer be approved.
- Delete operations require the same explicit approval as any other edit, and warn more
  loudly.

**Note:** changes are created via the backend API (`POST /api/v1/files/propose`). The
review-and-approve flow in the UI is what you use once a change exists.

---

## 11. Settings

All settings are **user-level or machine-level**. They apply to every project and are
never written into a project's `.vscode/settings.json`. Installing once is enough;
switching projects never requires reconfiguring.

| Setting | Default | Scope | Meaning |
|---|---|---|---|
| `llmTrainingAgent.provider` | `ollama` | User | Which LLM provider to use |
| `llmTrainingAgent.model` | `llama3.2` | User | Model name for that provider |
| `llmTrainingAgent.pythonPath` | *(empty)* | Machine | Absolute path to the Python interpreter for the backend. Empty means "use the bundled interpreter, or `python` from PATH" |
| `llmTrainingAgent.backendUrl` | `http://127.0.0.1:8000` | Window | Only change this to point at an externally managed backend |

Edit them in **Settings UI** (`Ctrl+,` → search `llmTrainingAgent`) or in `settings.json`
under `llmTrainingAgent`.

---

## 12. Where your data is stored

The agent keeps its own state in VS Code's global storage for the extension, not in your
project and not in the extension install folder.

| What | Location |
|---|---|
| SQLite database (analysis runs, projects, experiments, recommendations) | VS Code global storage → `llm_training_agent.db` |
| Authorized project roots | VS Code global storage → `allowed-roots.txt` |
| Pending file changes | Inside the analyzed project → `.llm_training_agent/pending_changes/` |
| Change backups | Inside the analyzed project → `.llm_training_agent_backups/` |
| Cleaned dataset output | Inside the analyzed project → `.llm-training-agent/cleaned/` |
| API keys | VS Code SecretStorage (encrypted by VS Code) |

Nothing is written to your project except those three directories: the first two
only when you approve a file change, and the third only when you run dataset
cleaning (which never modifies your original dataset files). See
[DELETION.md](DELETION.md) for cleanup.

---

## 13. Errors and troubleshooting

| Symptom | What it means | What to do |
|---|---|---|
| "the Python backend could not be started" | The bundled backend failed to launch | Open the **LLM Training Agent: Backend** output channel for the real error. Usually a missing Python or missing packages. |
| "backend dependencies are not available" | The interpreter lacks the runtime packages | Run **Analyze Project** — the agent offers to install them once you approve. Or run `pip install -r backend/requirements.txt` yourself with the interpreter from `llmTrainingAgent.pythonPath`. |
| "no folder is open" | Commands need a project | Open the project folder, then run the command. |
| "outside the authorized workspace" | A path outside the open folder was requested | Expected behaviour. Only the open folder is ever read. |
| Chat says no provider is available | No API key, or Ollama is not running | Open **Settings**, configure a provider, and use **Test Connection**. |
| "Database unavailable" (HTTP 503) | The SQLite file could not be opened or migrated | Check write permissions on the extension's global storage folder. |
| Analysis is slow | Large dataset or a slow local model | Let it run, or analyse a smaller directory with **Analyze Dataset**. |

Every backend error message is written for a human, names the actual problem, and is
scrubbed of API keys and absolute paths. Tracebacks are never shown to you.

---

## 14. Known limitations

Honest list of what is *not* finished or not verifiable:

- **No fabricated accuracy predictions.** The agent deliberately does not estimate a
  benchmark score or expected quality gain, because this project holds no empirical
  evaluation data to justify one. It reports what it can measure and labels the rest.
- **Model specs and pricing come from a bundled reference database**, not a live
  provider lookup. Treat pricing and availability as possibly outdated, and verify
  against the provider's official documentation before committing budget. The report
  says so explicitly.
- **GPU detection is best-effort.** It uses PyTorch if installed, otherwise
  `nvidia-smi`, otherwise environment variables. On a machine with no NVIDIA GPU it
  correctly reports zero GPUs, and training-time estimates degrade accordingly.
- **Dataset cleaning has no button in the UI** — it is reachable through the chat agent
  and the API only.
- **Experiments have no dedicated UI view** — API only.
- **GPU time estimates are estimates.** The quick mode uses published TFLOPs figures;
  the calibrated mode runs a small benchmark. Neither is a guarantee. Assumptions are
  always listed alongside the number.
- **Windows and macOS/Linux paths are handled, but the project was developed and tested
  on Windows.**


| Provider | Needs an API key | Notes |
|---|---|---|
| **Ollama** (default) | No | Local models. Also the only provider that works fully offline. |
| **OpenAI** | Yes | |
| **Anthropic** | Yes | |
| **Google Gemini** | Yes | |
| **DeepSeek** | Yes | |
| **Cohere** | Yes | |
| **OpenRouter** | Yes | Free and paid models |
| **OpenAI-compatible** | Usually | Any local or hosted server that speaks the OpenAI API (e.g. LM Studio) |

### Actions in the Settings view

- **Provider dropdown** — switch provider. The model list and API key field update.
- **Model** — a dropdown for providers that publish a model list, or a free-text field
  (Ollama, OpenAI-compatible) where you type the model name.
- **Save** — stores the provider and model choice, and the API key if one was typed.
- **Test Connection** — verifies the provider is reachable and the credentials work.
  Reports success or a specific failure.
- **Remove Key** — deletes the stored key for the selected provider.
- **Refresh Models** — re-fetches the model list (Ollama, OpenRouter, OpenAI-compatible).

### Where your API key goes

API keys are held in **VS Code SecretStorage**, not in a settings file and not in your
project. They are sent to the local backend over `127.0.0.1` at runtime and held in
memory only — the backend never writes them to disk, and every log line, error message
and LLM prompt is scrubbed of credential-shaped strings before it leaves the process.

Ollama needs no key. If you use it, the extension works with no account and no network.

---

## 9. Experiments

**Where:** backend API (`/api/v1/experiments`). Not yet surfaced as a dedicated view in
the extension UI.

**What it does:** records training runs — model, hyperparameters, dataset, actual
versus estimated results, status including failures — and compares them. All state lives
in the local SQLite database, not in your project.

| **Next experiment** | A concrete first step with a success criterion |

Sections that could not be produced appear under **unavailable**, with the reason.
The agent never fabricates a value to fill a gap.
