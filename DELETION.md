# Uninstall and Cleanup

How to remove the extension and the files it leaves behind.

## 1. Uninstall the extension

1. Open Extensions (`Ctrl+Shift+X`).
2. Find **LLM Training Agent**, click the gear icon, and choose **Uninstall**.
3. Choose **Reload Window**.

Or, from a terminal: `code --uninstall-extension rayansharma.llm-training-agent`

## 2. Delete leftover extension files (Windows)

VS Code normally removes the extension's folder. If it is still there, delete **only** that one folder:

```
C:\Users\<your-name>\.vscode\extensions\
```

1. Open File Explorer, then `C:` → `Users` → your account folder.
2. Open `.vscode` → `extensions`.
3. Delete the folder whose name starts with `rayansharma.llm-training-agent`.

Do not delete the whole `.vscode` folder or other extensions' folders. If you cannot find it, there is nothing to delete. (On macOS and Linux the folder is `~/.vscode/extensions/`.)

## 3. Delete the data it created

Delete only what you no longer need.

| What | Where | Notes |
|---|---|---|
| Analysis history: `llm_training_agent.db` (and `-wal`, `-shm`) and `allowed-roots.txt` | VS Code's storage for the extension. Search your user profile for `allowed-roots.txt`; its folder is the one. | Close VS Code first. They are recreated if you use it again. |
| Pending changes and backups | `.llm_training_agent/` and `.llm_training_agent_backups/` inside each project you analyzed | Deleting the backups removes your ability to roll back applied changes. |
| Leftover cleaned copies | `.llm-training-agent/` inside each project | Only exists if you saved AI-cleaned copies and never merged or discarded them. Your dataset files are not affected. |
| API keys | In the extension's **Settings**, press **Remove API Key** for each provider, before uninstalling | Do not edit the system credential store by hand. |
| Settings | `Ctrl+,`, search `llmTrainingAgent`, **Reset Setting** | Optional. They do nothing once uninstalled. |

## 4. Optional

- Delete the downloaded `llm-training-agent-1.0.0.vsix`.
- The Python packages the extension installed stay in your Python. Remove them with `pip uninstall` only if nothing else uses them.

## Keep these

- Your own project files. The extension only changes them after you approve.
- The repository's `backend/`, `extension/` and `requirements.txt`, if you may reinstall.
- Anything that belongs to VS Code or other extensions.

Chat history is kept in memory only and disappears when VS Code closes, so there is nothing to delete.
