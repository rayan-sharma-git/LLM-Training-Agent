# Uninstall and Cleanup

How to remove the LLM Training Agent extension and its leftover files.

## 1. Uninstall from VS Code

1. Open the Extensions view (`Ctrl+Shift+X` on Windows/Linux, `Cmd+Shift+X` on macOS).
2. Find **LLM Training Agent**, click the gear icon, choose **Uninstall**.
3. Choose **Reload Window** when prompted.

Alternatively, run `code --uninstall-extension rayansharma.llm-training-agent` if the `code` command is on your PATH.

## 2. Delete leftover files

Uninstalling removes the extension itself, but not the files below. Delete only what you no longer need.

| File or folder | Location | Safe to delete? |
|---|---|---|
| `llm_training_agent.db`, `llm_training_agent.db-wal`, `llm_training_agent.db-shm`, `allowed-roots.txt` | Extension global storage (the exact path is chosen by VS Code; search your user profile for `allowed-roots.txt` — its folder is the location, with `llm_training_agent.db` beside it). Close VS Code first. | Yes, if you no longer need your analysis history. They are recreated on next use. |
| `.llm_training_agent/`, `.llm_training_agent_backups/` | Inside each analyzed project folder (proposed changes and backups of applied changes). | Yes, once you no longer need pending proposals or rollback of applied changes. |
| `.llm-training-agent/` | Inside each analyzed project folder (dataset-cleaning output; your original files are never modified). | Yes, if you no longer need the cleaned copies. |
| API keys | Settings view → **Remove API Key** for each provider, before uninstalling. | Yes. Do not edit the OS credential store by hand. |
| `llmTrainingAgent.*` settings | VS Code user settings (`Ctrl+,`, search `llmTrainingAgent`, Reset Setting). | Yes, optional. They do nothing once uninstalled. |

## 3. Optional permanent cleanup

If you will never use the extension again, also delete the downloaded `llm-training-agent-1.0.0.vsix` file. Python packages installed from `backend/requirements.txt` stay in your Python environment after uninstall — remove them with `pip uninstall` only if nothing else uses them.

## 4. Files to keep

- **Your project source files.** The extension only changes them after your explicit approval.
- **The repository's `backend/`, `extension/`, and `requirements.txt`.** These are source files, not installed state; keep them if you may reinstall.
- **Everything belonging to other extensions or VS Code itself.** Touch only the files listed above.

Chat history and logs live in memory only and disappear when VS Code closes — there is nothing to delete.
