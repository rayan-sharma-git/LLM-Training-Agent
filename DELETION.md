# Deletion & Cleanup Guide

How to remove the LLM Training Agent extension safely, what survives an
uninstall, and how to reinstall with a completely fresh state.

This guide is written against the actual implementation in this repository, not
against generic VS Code conventions. Where a physical path is chosen by VS Code
rather than by this extension, that is stated explicitly instead of guessing.

---

## 1. Uninstall the Extension

### Supported method (UI)

1. Open the Extensions view (`Ctrl+Shift+X` on Windows/Linux, `Cmd+Shift+X` on
   macOS).
2. Find **LLM Training Agent** in the installed list.
3. Click the **⚙ (gear)** icon → **Uninstall**.
4. Choose **Reload Window** when prompted.

This is the only method that is guaranteed available on this machine.

### CLI method (optional, may not be installed)

The supported command form is:

```bash
code --uninstall-extension rayansharma.llm-training-agent
```

`rayansharma.llm-training-agent` is the real extension identifier, taken from
`"publisher"` and `"name"` in `extension/package.json`.

However, the CLI directory observed on the development machine
(`C:\Users\Ryan\.VS Code\CLI\`) is **empty**, which means no VS Code CLI shims
were installed. If `code` is not on your `PATH`, this command will simply fail
with "command not found" — use the UI method above instead. It is not necessary
to install the CLI just to uninstall the extension.

---

## 2. What Uninstalling Removes — and What Survives

### Removed by VS Code (extension installation files)

The installed extension directory is deleted:

```
C:\Users\Ryan\.VS Code\extensions\rayansharma.llm-training-agent-1.0.0\
```

Observed contents of that directory: `out`, `resources`, `.vsixmanifest`,
`CHANGELOG`, `LICENSE`, `package.json`, `README`.

Also removed: the **LLM Training Agent** activity-bar icon, the four sidebar
views (Overview, Chat, Reports, Settings), and the six contributed commands
(`analyzeProject`, `analyzeDataset`, `openChat`, `viewReport`,
`configureProvider`, `viewChanges`).

### Survives the uninstall

| What | Why it survives |
|---|---|
| SQLite database + WAL/SHM files | VS Code global storage is not owned by the extension code |
| `allowed-roots.txt` | Same |
| API keys in SecretStorage | This extension never deletes them; see section 4 |
| `.llm_training_agent/` and `.llm_training_agent_backups/` in projects | They are ordinary files inside your projects |
| `llmTrainingAgent.*` settings | VS Code keeps user settings after uninstall |
| Python packages installed by `pip` | Installed into your Python environment, not the extension |
| An externally-started backend process | Not managed by the extension once uninstalled |

**The extension itself contains no uninstall handler.** `deactivate()` in
`extension/src/extension.ts` is an empty function, and there is no code anywhere
that deletes global-storage files, project folders, or secrets on uninstall. Any
removal beyond the install directory is done by VS Code or left to you.

---
## 3. Remove Extension Persistent Data

### What the extension writes into global storage

Verified in code — the extension writes exactly two things into
`context.globalStorageUri`:

| File | Written by | Contents |
|---|---|---|
| `llm_training_agent.db` | `BackendManager.databaseUrl()` (`extension/src/services/backendManager.ts:302-308`), which passes it to the backend as `DATABASE_URL` | SQLite database: analysis runs, projects, experiments, recommendations |
| `llm_training_agent.db-wal`, `llm_training_agent.db-shm` | SQLite, from `PRAGMA journal_mode=WAL` (`backend/storage/database.py:169`) | Write-ahead log and shared-memory index. Present only while the backend has the database open |
| `allowed-roots.txt` | `publishAllowedRoots()` (`extension/src/services/extensionPaths.ts:141,153-164`) | The folders currently open in VS Code, used by the backend to authorise project access. Rewritten on every project switch |

### Where that folder physically is

The extension calls `context.globalStorageUri` and never hardcodes a path — this
is a deliberate design decision (`backendManager.ts:16-19`).

**The physical location is therefore decided by VS Code, not by this extension**,
and it varies with the VS Code distribution and profile: stable vs Insiders,
local vs Remote-SSH / WSL / dev-container, portable vs installed, and per user
profile.

A specific caveat for this machine: the installation observed on the development
machine has **no** `C:\Users\Ryan\.VS Code\User\` folder and **no**
`%APPDATA%\Code\User\` folder. So the commonly cited path
`%APPDATA%\Code\User\globalStorage\rayansharma.llm-training-agent\` is
**not applicable here** and must not be used as a deletion target.

### How to locate the folder safely

The most reliable method is to let the extension create it and then find it:

1. Open a project folder in VS Code.
2. Run **LLM Training Agent: Analyze Project**. This starts the backend, which
   creates the global storage folder and the database.
3. Search your user profile for the two filenames:

```powershell
Get-ChildItem -Path "$env:USERPROFILE" -Filter 'allowed-roots.txt' -Recurse -Force -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty FullName
```

The extension's global storage folder is the **parent** of the `allowed-roots.txt`
that is found. Verify by confirming that `llm_training_agent.db` sits beside it.

### Deleting it (optional)

This step is entirely optional. Deleting it discards your entire analysis
history — every past report, run, project record and experiment — for **all**
projects, not just one. Nothing else is lost, and nothing outside this folder is
affected.

Stop the backend first (close the VS Code window, or reload it after
uninstalling), otherwise the `-wal` / `-shm` files may be held open.

```powershell
# Replace <globalStorageFolder> with the folder found in the step above.
Remove-Item -LiteralPath '<globalStorageFolder>\llm_training_agent.db'    -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath '<globalStorageFolder>\llm_training_agent.db-wal' -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath '<globalStorageFolder>\llm_training_agent.db-shm' -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath '<globalStorageFolder>\allowed-roots.txt'        -Force -ErrorAction SilentlyContinue
```

Deleting the whole folder is also fine if you confirmed it is the extension's
own folder and contains nothing else. **Never delete the global storage folder
itself** — many other extensions keep state in it.

### What is recreated automatically

Both files are recreated automatically on the next use; no reinstall is needed:

- `allowed-roots.txt` is rewritten by `refreshAllowedRoots()` whenever the open
  folders change (`backendManager.ts:120-141`), and on every backend start.
- `llm_training_agent.db` is created and migrated lazily by
  `Database.init_schema()` (`backend/storage/database.py:181-195`) during backend
  startup. Its parent directory is created first (`database.py:147-153`).

---

## 4. Remove API Keys / Credentials

### How keys are stored

Keys are stored in **VS Code SecretStorage** under these keys:

```
llmTrainingAgent.apiKey.ollama
llmTrainingAgent.apiKey.openai
llmTrainingAgent.apiKey.anthropic
llmTrainingAgent.apiKey.gemini
llmTrainingAgent.apiKey.deepseek
llmTrainingAgent.apiKey.cohere
llmTrainingAgent.apiKey.openrouter
llmTrainingAgent.apiKey.openai_compatible
```

The pattern is `llmTrainingAgent.apiKey.<provider>`
(`extension/src/services/apiClient.ts:75,147-148`). Only providers you actually
configured will exist.

The backend receives keys at runtime over `127.0.0.1` and keeps them **in memory
only** — `backend/core/runtime_config.py` and `backend/api/routes.py:260-262` are
explicit that keys are never written to disk. There is no `.env` file and no key
file to delete.

### Supported removal method

1. Open the **LLM Training Agent → Settings** view in the sidebar (or run
   **LLM Training Agent: Configure AI Provider**).
2. Select the provider in the dropdown.
3. Click **Remove API Key** under *Actions*
   (`extension/src/views/settingsWebviewProvider.ts:378,517-522,85-89`).

This deletes the SecretStorage entry **and** clears the key from the running
backend's memory (`removeApiKey()` and `removeApiKeyBackend()`).

Repeat for every provider you configured. This is the only supported cleanup
route.

### Does uninstalling delete them?

**Not demonstrably.** The extension contains no code that deletes secrets on
uninstall — there is no cleanup handler in `extension.ts`, and no `secrets.delete`
call outside the Remove API Key button. Whether VS Code itself prunes a secret
store when an extension is uninstalled is **VS Code's own behaviour and is
controlled by VS Code and the operating system's credential store**; it cannot be
verified from this codebase and is not guaranteed here.

Therefore: **remove your keys through the extension UI before uninstalling.**
That is deterministic and guaranteed. After removing them, uninstalling leaves
nothing behind to worry about.

### Do not do this

Do **not** manually edit or delete entries in the Windows Credential Manager, the
macOS Keychain, or any encrypted secrets store. Those are shared with every other
application and extension, they are easy to corrupt, and the extension UI already
provides a correct way to remove exactly what it stored.

---

## 5. Remove Project-Specific Data

The agent creates up to **three** directories at the **root of each analyzed
project**, and only for the feature that produced them:

| Directory | Created for | Contents |
|---|---|---|
| `.llm_training_agent/pending_changes/` | Proposed file edits (`file_editor.py:58,216`) | One JSON record per proposal: target path, operation, original hash, and the proposed content |
| `.llm_training_agent_backups/` | Applied file edits (`file_editor.py:60`) | Pre-change copies of each modified file, named `<flattened-path>.<changeId>.bak` (`file_editor.py:645-651`) |
| `.llm-training-agent/cleaned/` | Dataset cleaning (`dataset_cleaner.py:43`) | Rewritten copies of your dataset files, mirroring their original paths, plus `cleaning_manifest.json` |

The first two exist only when a file change has actually been
proposed/applied. The third exists only when you run dataset cleaning, and
never modifies your original dataset files.

Nothing else is ever written into your project. The agent reads your files and
only writes to a project through an explicit, individually approved change
(`file_editor.py`: PROPOSED → APPROVED → APPLIED), or through the dataset
cleaning output directory when you explicitly run cleaning.

Note the two different spellings: `.llm_training_agent` (underscores) is used by
the **file-change** workflow, and `.llm-training-agent` (hyphens) by the
**dataset-cleaning** workflow. They are separate directories; neither contains
the other.

### Deleting them

```powershell
# Run from inside the project folder, one project at a time.
Remove-Item -LiteralPath '.llm_training_agent'         -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath '.llm_training_agent_backups' -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath '.llm-training-agent'         -Recurse -Force -ErrorAction SilentlyContinue
```

```bash
# Linux / macOS
rm -rf .llm_training_agent .llm_training_agent_backups .llm-training-agent
```

### What you lose

- **`.llm_training_agent/pending_changes/`** — pending proposals become
  unapprovable. This is low-risk: proposals already expire after 24 hours
  (`PROPOSAL_TTL`, `file_editor.py:55`).
- **`.llm_training_agent_backups/`** — **rollback is permanently disabled.**
  `rollback()` (`file_editor.py:611-643`) restores the pre-change content from
  the `.bak` file; with the backup gone, an already-applied edit can only be
  undone by hand. Delete this directory **only** once you are certain you do not
  want to roll back any applied change.
- **`.llm-training-agent/cleaned/`** — the cleaned dataset output is gone. Your
  original dataset files are never modified by cleaning, so re-running it
  regenerates the output.

Both deletions are optional and neither is recreated unless you apply another
approved change.

---
## 6. Remove Settings

The extension declares exactly four settings in `extension/package.json`, all
under the `llmTrainingAgent` namespace. There is no `.env` file and no extension-
owned settings file.

| Setting | Default | Scope |
|---|---|---|
| `llmTrainingAgent.backendUrl` | `http://127.0.0.1:8000` | window |
| `llmTrainingAgent.pythonPath` | `""` | machine-overridable |
| `llmTrainingAgent.provider` | `ollama` | application |
| `llmTrainingAgent.model` | `llama3.2` | application |

These are **never written into a project**. `SettingsManager.updateSettings()`
always writes to `vscode.ConfigurationTarget.Global`
(`extension/src/services/settings.ts:45-51`), so no `.vscode/settings.json` in
your project contains `llmTrainingAgent` entries.

To remove them:

1. Open the Settings UI (`Ctrl+,`) and search `llmTrainingAgent`, then use
   **Reset Setting** on each — or
2. Open `settings.json` via **Preferences: Open User Settings (JSON)** and delete
   only the `"llmTrainingAgent": { ... }` block.

Uninstalling does **not** remove these values; they simply stop having any
effect, and are reused unchanged on reinstall. Resetting them is optional.

---

## 7. Complete Clean Reinstall

Use this only if you want a genuinely fresh start, or if the extension is
misbehaving after an update.

1. **Remove credentials first** — Settings view → **Remove API Key** for each
   configured provider (section 4). Do this *before* uninstalling, while the
   extension is still running.
2. **Uninstall** the extension via the Extensions view (section 1).
3. **Reload the window.**
4. *(Optional)* **Delete the persistent data**: `llm_training_agent.db`,
   `llm_training_agent.db-wal`, `llm_training_agent.db-shm` and
   `allowed-roots.txt` from the extension's global storage folder (section 3).
   Skip this step if you want to keep your analysis history.
5. *(Optional)* **Delete project-local data** in each project:
   `.llm_training_agent/` and `.llm_training_agent_backups/` (section 5). Keep
   `.llm_training_agent_backups/` if you still need to roll anything back.
6. *(Optional)* **Reset settings** in the Settings UI (section 6).
7. **Reinstall** `llm-training-agent-1.0.0.vsix` via
   Extensions → **Views and More Actions** → **Install from VSIX…**, per
   [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#3-install-the-vsix).
8. **Reload** the VS Code window if prompted. The backend is started lazily on
   first use, not at activation, so nothing needs to be started manually.
9. **Open a project folder** and run **LLM Training Agent: Analyze Project** to
   verify. Check the **LLM Training Agent: Backend** output channel if it does
   not respond.
10. If the backend cannot start, the likely cause is a missing Python or
    missing packages — the extension offers to install them once you approve.
    Note that installing them modifies your Python environment, and they remain
    after the extension is uninstalled.

---
## 8. What NOT to Delete

Do not delete any of the following when cleaning up this extension:

- ❌ **The VS Code installation or the entire user/configuration directory.**
  Uninstalling this extension never requires touching them.
- ❌ **The whole global storage folder, or any `globalStorage` directory.**
  Many other extensions keep state there. Delete only the four files listed in
  section 3.
- ❌ **Other extensions' folders**, including other versions of this extension
  that VS Code may still be holding for rollback.
- ❌ **Unrelated VS Code settings**, keybindings, snippets, tasks or workspace
  files. Only the `llmTrainingAgent` block is this extension's.
- ❌ **Your project source code.** The agent only ever writes to a project file
  through an individually approved change, and stores its own records in the two
  dot-directories in section 5.
- ❌ **Your Python installation, virtual environments, or site-packages.**
  `pip`-installed backend dependencies are shared with other tools and are not
  removed by uninstalling. Use `pip uninstall` selectively if you truly want the
  disk space.
- ❌ **The repository's `backend/`, `extension/`, `requirements.txt` or the
  VSIX.** These are source/release files, not installed state; deleting them
  breaks nothing about an already-installed extension and prevents reinstalling.
- ❌ **Windows Credential Manager, macOS Keychain, or any secrets database.**
  See section 4.
- ❌ **Anything you did not see listed in this document.** If you are unsure
  whether something belongs to this extension, leave it alone.

### A note on logs and chat history

The backend writes its log to standard output only
(`backend/core/logging.py:16-21`, `stream=sys.stdout`). The extension captures
that stream into the in-memory **LLM Training Agent: Backend** output channel
(`backendManager.ts:64,245-251`). No log files are written to disk by this
extension, so there is nothing to delete. Chat conversation history is likewise
**in-memory only** (`backend/chat/store.py:1-14`) and disappears when the backend
process stops — it is never persisted to the database.
