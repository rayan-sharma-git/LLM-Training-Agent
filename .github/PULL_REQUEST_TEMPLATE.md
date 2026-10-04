# Pull Request

## What and why

<!-- What changed, and what problem it solves. One concern per PR. -->

## Checklist

- [ ] I have read and followed [CONTRIBUTING.md](../CONTRIBUTING.md) and know which file owns the responsibility I touched.
- [ ] My branch is based on `main` and contains a single, focused concern.
- [ ] I inspected the existing code first and matched the surrounding style.
- [ ] I did not weaken existing invariants (for example, extension resources are still resolved only from the extension install directory, never from the open project).
- [ ] I have tested my changes: `cd backend && python -m pytest tests -q` and/or `cd extension && npm test && npm run lint` for the part I touched.
- [ ] Existing functionality and tests still pass, or I updated them deliberately in this PR.
- [ ] I added no unnecessary dependencies; `requirements.txt` and `backend/requirements.txt` are in sync if I changed any.
- [ ] I did not redesign the architecture, add commented-out code, stray debug prints, or duplicated logic.
- [ ] All reported values come from the user's project or a documented formula — nothing is fabricated, and unknown results are reported as unknown.
- [ ] If this change affects the VSIX, I rebuilt it and `python scripts/verify_vsix.py llm-training-agent-1.0.0.vsix` passes, and there is exactly one VSIX at the repository root.
- [ ] I updated the relevant root documentation (`USER_MANUAL.md`, `INSTALLATION_GUIDE.md`, `PROJECT_STRUCTURE.md`, `DELETION.md`) and did not add duplicate copies elsewhere.
- [ ] I included no secrets, API keys, passwords, tokens, or other sensitive information or user data.
- [ ] I included no copyrighted or proprietary code I do not have the right to use.
- [ ] I reviewed the final diff for unintended changes, generated files, or caches.

## How I verified it

<!-- Commands run and what the output showed. -->
