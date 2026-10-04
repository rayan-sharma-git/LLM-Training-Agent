# LLM Training Agent

A VS Code extension that reads your actual LLM project — datasets, prompts, training
configuration, hardware — and tells you whether fine-tuning it is worth it, what it
would cost, how long it would take, and what to fix first.

The question it answers is the one that is expensive to answer by hand: *"I have this
project and this data. Should I fine-tune, and if so, exactly how?"*

## The problem

Deciding whether to fine-tune a model today means stitching together scattered facts:
row counts and duplicate rates from the dataset, hyperparameters from whatever config
file happens to be lying around, a GPU you may or may not have, and pricing pages you
have to keep refreshing. The engineer doing this is usually guessing, and the guess
gets expensive precisely when it is least checked.

Existing tools split cleanly into two camps, and both fail:

- **Dashboards** show metrics without context. A row count is not a verdict.
- **General-purpose LLM chat** is fluent and confidently wrong about your files. It has
  no measurements, so it invents them.

This project takes the opposite stance: **measure first, reason second**. Every number
in the report comes from your project. The AI interprets those numbers; it never
invents them.

## The hybrid approach

The system is deliberately *not* a pile of `if/else` rules, and *not* a thin wrapper
around an LLM prompt. It splits the work by what each method is actually good at.

**Deterministic engineering** handles everything measurable. Dataset statistics —
sample counts, exact and near-duplicate ratios, missing-field rates, length
distributions, formatting and language consistency — are computed directly from the
files. Training-time and VRAM estimates come from a FLOPs/throughput model over the
detected GPU. Cost estimates come from arithmetic over real token counts. These values
are reproducible: run the analysis twice on the same project and you get the same
answer, and you can check the work.

**LLM reasoning** handles everything interpretive. Turning "17% near-duplicates, 240
samples, 3 epochs at 2e-4" into *"your dataset is too small to learn from and too
repetitive to help — get to a few hundred genuinely distinct examples before tuning
anything"* is judgement, not arithmetic. The same applies to reading prompt files for
ambiguity, weighing trade-offs, and explaining *why* a recommendation was made. Rules
can enumerate failure modes; they cannot prioritise them for your situation.

**Your project is the source of truth.** The LLM is given real measured context, not a
guess at it. Where something genuinely cannot be determined, the system says so rather
than producing a plausible number. Every value is labelled by provenance — *measured*,
*calculated*, *heuristic*, or *assumed* — so you always know which numbers to trust and
which to verify. There are no fabricated accuracy scores.

If no LLM provider is configured, the deterministic half still works. The analysis
degrades to measurements and rule-based recommendations instead of disappearing.

## What it does

Open a project folder, run **Analyze Project**, and get a single report in the chat
panel covering:

- **Use-case detection** — what your project is actually trying to do, with a confidence score
- **Dataset analysis** — size, quality, duplicates, missing fields, consistency
- **Prompt analysis** — clarity, ambiguity, formatting, undefined placeholders
- **Model advice** — which base model fits your task and your VRAM, and why
- **Hyperparameter review** — learning rate, batch size, epochs against your actual data
- **Hardware detection** — real GPUs, VRAM, CUDA version on this machine
- **Training-time and VRAM estimation** — with ranges and stated assumptions
- **Cost estimation** — local vs. hosted, with budget options
- **Risk register** — overfitting, hallucination risk, no evaluation harness, with mitigations
- **Recommendations** — prioritised, each with its reasoning
- **A concrete next experiment** — with a success criterion

Then keep going: ask follow-up questions in **Chat**, generate a **Report**, track
**Experiments**, and review or apply **proposed file changes** through a diff.

## Architecture

Two components:

| Component | Path | Responsibility |
|---|---|---|
| **VS Code extension** | `extension/` | UI, commands, bundled-backend lifecycle, VS Code SecretStorage |
| **Python backend** | `backend/` | FastAPI service: scanning, analysis, estimation, storage, LLM access |

The extension bundles the entire Python backend inside the VSIX. On first use it starts
that backend as a local subprocess on `127.0.0.1:8000` and talks to it over HTTP (plus a
WebSocket for analysis progress). Nothing is uploaded anywhere; the backend only ever
reads the folder you have open in VS Code.

This split is not incidental. The analysis logic is numeric, file-parsing and
database-backed — comfortable in Python. The interaction logic is panels, tree views,
diff editors and secret storage — comfortable in the VS Code API. Packaging the Python
side *inside* the VSIX means users install one file and get a working tool, while
developers still work on two cleanly separated codebases.

The extension resolves its backend and interpreter **only** from the extension install
directory, never from your project. A project that happens to contain a `backend/`
folder cannot replace the agent's own backend, and switching projects never mixes
state between them.

## Technology

- **Backend:** Python 3.10.6, FastAPI, Pydantic v2, SQLAlchemy 2 (async) + SQLite, httpx
- **Extension:** TypeScript, VS Code Extension API, axios, vitest
- **Packaging:** `vsce` → VSIX
- **Tests:** pytest (backend), vitest (extension), plus a VSIX packaging audit

Tested with Python 3.10.6.

## Documentation

| Document | What it covers |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | Every user-facing feature, end to end |
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | Installing the VSIX |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Repository layout and where to change what |
| [DELETION.md](DELETION.md) | Safe uninstall and cleanup |

## License

MIT. See [LICENSE](LICENSE).
