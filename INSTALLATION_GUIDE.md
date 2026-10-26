# Installation Guide

How to install the LLM Training Agent in VS Code. For what it does, see [USER_MANUAL.md](USER_MANUAL.md). To remove it, see [DELETION.md](DELETION.md).

## 1. What you need

- **VS Code 1.85 or newer.**
- **Python 3.10 or newer** on your computer. Most computers already have it. The extension uses it to run its built-in backend. To use a specific Python, set `llmTrainingAgent.pythonPath`.

You do **not** need:

- an API key. The default provider, Ollama, runs on your own computer. You only need a key if you choose an online provider, and the extension tells you when.
- Node.js or npm.
- to install any Python packages yourself. The extension installs the ones it needs (see step 4).

## 2. Get the file

The installer is one file in the top folder of this repository, next to `README.md`:

```
llm-training-agent-1.0.0.vsix
```

- **On GitHub:** open that file and click **View Raw** (or the download button). Save it anywhere, for example **Downloads**. Downloading does not install it.
- **If you cloned the repository:** the file is already there.

## 3. Install it

1. Open VS Code.
2. Open Extensions (`Ctrl+Shift+X` on Windows and Linux, `Cmd+Shift+X` on macOS).
3. Click the **⋯** menu at the top right of the Extensions view.
4. Choose **Install from VSIX…** and select `llm-training-agent-1.0.0.vsix`.
5. When it finishes, choose **Reload Window** if VS Code asks.

## 4. First start

Open a project folder. The extension then prepares its backend by itself:

1. It looks for Python 3.10 or newer.
2. It checks the packages listed in `backend/requirements.txt`, the list that is built into the extension. If any are missing, it installs them automatically. This can take a minute or two the first time.
3. It starts the backend.

Progress is shown in **View → Output → LLM Training Agent: Backend**. If you would rather be asked first, turn off the setting `llmTrainingAgent.autoInstallDependencies`.

If your Python refuses system-wide installs (common on macOS with Homebrew, and on Debian or Ubuntu), the Output channel says so. Create a virtual environment (`python -m venv .venv`), install `backend/requirements.txt` into it, and set `llmTrainingAgent.pythonPath` to that environment's Python.

## 5. Check that it works

1. Click the **LLM Training Agent** icon in the left sidebar. You should see **Overview**, **Chat**, **Reports** and **Settings**.
2. Run **LLM Training Agent: Analyze Project** from the command palette (`Ctrl+Shift+P`).

If something goes wrong, a message names the problem. The Output channel from step 4 has the details.

## 6. Choose an AI provider (optional)

The extension works straight away with Ollama (run `ollama pull llama3.2` once to download a model). To use an online provider, open **Settings** in the extension, pick a provider, paste your key, press **Save**, then **Test Connection**. See [USER_MANUAL.md](USER_MANUAL.md#10-choosing-an-ai-provider).

Your key is stored in VS Code's secure storage, never in your project.
