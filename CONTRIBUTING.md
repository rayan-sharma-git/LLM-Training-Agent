# Contributing

Contributions are welcome. Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) first: it shows which file does what.

## Ground rules

- **Look before you change.** Understand why the code is built the way it is. Some rules are deliberate and tests enforce them. For example, the extension reads its own files only from its install folder, never from the open project. Do not weaken them.
- **Keep changes small.** One concern per pull request. A refactor mixed with a feature is hard to review and hard to undo.
- **Do not redesign.** Adding an analyzer, a provider or an endpoint fits the current patterns. Rewriting the structure does not belong in a contribution.
- **Avoid new dependencies.** Every package is a cost for every user. Prefer the standard library or code already in the project.
- **Do not break what works.** If behaviour changes, keep the tests passing or update them in the same commit.
- **Keep the code clean.** No commented-out code, no stray debug prints, no duplicated logic. Match the style around it.
- **Never invent numbers.** Results must come from the user's project or from a documented formula. If something cannot be known, say it is unknown, and never add an AI-generated accuracy score.

## Setup

- Python 3.10+. Install the root `requirements.txt` (it has the runtime packages plus the test and lint tools).
- Node.js. Run `npm install` in `extension/`.
- Press `F5` in VS Code to launch a test copy of the extension.

There are two requirements files on purpose. `backend/requirements.txt` has only what is needed to **run** the extension. It is built into the VSIX and is what users get. The root `requirements.txt` is for **contributors**. When you add or change a dependency, update **both**.

## Before you open a pull request

Run the tests for the part you changed.

```bash
cd backend   && python -m pytest tests -q
cd extension && npm test && npm run lint
```

The shared VS Code tasks (**Terminal → Run Task**) do the same: **Backend: run tests**, **Extension: test**, **Extension: compile**, **Extension: package VSIX**.

If your change affects anything inside the VSIX, rebuild and audit it:

```bash
cd extension
npm run package
npm run verify:vsix
```

Both must pass. The audit fails if the package would ship secrets, caches, test files or paths that only work on your machine, or if a required file is missing.

`npm run package` builds the VSIX in `extension/`, then the `postpackage` hook (`scripts/promote-vsix.js`) copies it to the repository root as `llm-training-agent-1.0.0.vsix` and deletes the build copy. Keep exactly one copy, the root one. `extension/*.vsix` is git-ignored. Never commit it or copy it by hand.

## Update the docs

Change the matching document in the same pull request. All docs live at the repository root. Do not add copies elsewhere.

| You changed | Update |
|---|---|
| A feature or command users see | [USER_MANUAL.md](USER_MANUAL.md) (and the Basics part of [README.md](README.md)) |
| How it works inside | The Technical Details part of [README.md](README.md) |
| Installing or packaging | [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md), and rebuild the root VSIX |
| The layout of files and folders | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |
| What the extension stores or leaves behind | [DELETION.md](DELETION.md) |
| A dependency | `requirements.txt` **and** `backend/requirements.txt` |

## Workflow

1. **Fork** the repository.
2. **Branch** from `main` with a clear name, like `fix/allowed-roots-cache` or `add-anthropic-provider`.
3. **Commit** focused changes with clear messages.
4. **Push** to your fork.
5. **Open a pull request** that says what changed, why, and how you checked it. Update your branch from `main` first.

## Reporting bugs

Open an issue with:

- what you did, what you expected, and what happened,
- your OS and VS Code version,
- the text of the **LLM Training Agent: Backend** output channel, if the backend was involved (secrets are removed from it, so it is safe to share),
- the relevant file names in your project (not your data).
