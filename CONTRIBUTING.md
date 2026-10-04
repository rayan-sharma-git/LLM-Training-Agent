# Contributing

Contributions are welcome.

Before changing anything, read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) so you know
which file owns which responsibility.

## Ground rules

- **Inspect before you change.** Understand how the existing code works and why it is
  structured that way. This project has deliberate invariants (for example: extension
  resources are resolved only from the extension install directory, never from the open
  project) that tests enforce. Do not weaken them.
- **Make focused changes.** One concern per pull request. A refactor bundled with a
  feature is hard to review and hard to revert.
- **Do not redesign the architecture.** Adding an analyzer, a provider or an endpoint
  fits the existing patterns. Rewriting the layering does not belong in a contribution.
- **Avoid unnecessary dependencies.** Every new package is a cost for every user. Prefer
  the standard library or code already in the project.
- **Do not break existing functionality.** If you change behaviour, keep the existing
  tests passing or update them deliberately in the same commit.
- **Keep the code clean.** No commented-out code, no stray debug prints, no duplicated
  logic. Match the surrounding style.
- **No fabricated numbers.** Analysis output must come from the user's project or from
  a documented formula. If something cannot be determined, report it as unknown — do
  not invent a plausible value, and do not add an LLM-generated accuracy score.

## Before you open a pull request

Run the tests for the part you touched.

```bash
# Backend
cd backend
python -m pytest tests -q

# Extension
cd extension
npm test          # vitest
npm run lint      # tsc --noEmit
```

Or use the shared VS Code tasks (`Terminal > Run Task`), which are wired to
`venv310\Scripts\python.exe` and to `npm` inside `extension/`: **Backend: run tests**,
**Extension: test**, **Extension: compile** and **Extension: package VSIX**.

If you changed anything that ends up inside the VSIX, rebuild and re-audit it:

```bash
cd extension
npm run package
python scripts/verify_vsix.py llm-training-agent-1.0.0.vsix
```

Both commands must pass. The audit script exits non-zero if the package would ship
secrets, caches, test files, or repository-relative path assumptions.

`npm run package` writes the VSIX into `extension/`. The released copy that users
download lives at the **repository root**, so after verifying, promote it:

```bash
# from the repository root
cp extension/llm-training-agent-1.0.0.vsix ./llm-training-agent-1.0.0.vsix
rm extension/llm-training-agent-1.0.0.vsix
```

Keep exactly one copy: the root one. `extension/*.vsix` is git-ignored precisely
because it is a build-directory artifact.

## Update the documentation

If your change alters user-facing behaviour, update the matching document in the same
pull request:

| You changed | Update |
|---|---|
| A user-facing feature or command | [USER_MANUAL.md](USER_MANUAL.md) |
| Installation or packaging | [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md), and the root VSIX |
| File or folder layout | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |
| Uninstall behaviour or stored data | [DELETION.md](DELETION.md) |
| Dependencies | `requirements.txt` **and** `backend/requirements.txt` |

Do not create a second copy of any of these documents in a subdirectory. They live at
the repository root.

## Workflow

The normal GitHub flow:

1. **Fork** the repository.
2. **Branch** off `main` with a descriptive name (`fix/allowed-roots-cache`,
   `add-anthropic-provider`).
3. **Commit** focused changes with clear messages.
4. **Push** the branch to your fork.
5. **Open a pull request** describing what changed, why, and how you verified it.

Keep the branch up to date with `main` before opening the pull request.

## Reporting bugs

Open an issue with:

- What you did, what you expected, and what happened instead.
- Your OS and VS Code version.
- The contents of the **LLM Training Agent: Backend** output channel, if the backend
  was involved. It is safe to share: credentials are scrubbed from log output.
- The relevant part of your project structure (file names, not your data).
