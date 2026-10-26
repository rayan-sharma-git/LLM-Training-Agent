# User Manual

Everything the LLM Training Agent can do, and how to use it.

- Installing it: [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- Removing it: [DELETION.md](DELETION.md)

## 1. Opening the extension

Click the **LLM Training Agent** icon in the VS Code sidebar. It has four views:

| View | What it is |
|---|---|
| **Overview** | Status of your project, plus quick buttons |
| **Chat** | The analysis report and your conversation with the agent |
| **Reports** | Saved reports |
| **Settings** | Choose an AI provider and save its key |

Open your project folder first. The extension only ever reads that folder.

## 2. First run

The first time you open the extension, it gets its Python backend ready on its own:

1. It looks for Python 3.10 or newer: your `llmTrainingAgent.pythonPath` setting first, then a bundled one, then `python` or `py` on your PATH.
2. It checks that the packages in the bundled `backend/requirements.txt` are installed and new enough. It never installs the developer tools from the root `requirements.txt`.
3. If something is missing, it installs the packages for you automatically. Progress shows in the **LLM Training Agent: Backend** output channel. This can take a minute or two.
4. It checks again after installing, then starts the backend.

If you would rather be asked first, turn off the setting `llmTrainingAgent.autoInstallDependencies`. A dialog then offers **Install Dependencies**, **Show Manual Command** or **Cancel**, and if you cancel you are not asked again until you restart VS Code.

If your Python refuses system-wide installs (common with Homebrew, Debian and Ubuntu), the output channel says so. Create a virtual environment (`python -m venv .venv`), install `backend/requirements.txt` into it, and set `llmTrainingAgent.pythonPath` to its Python.

## 3. Analyze Project

This is the main feature. Click **Analyze Project** in Overview, or run `LLM Training Agent: Analyze Project` (`Ctrl+Shift+P`). You can also type "analyze my project" in the chat.

It scans your datasets, prompts, configs and scripts and posts one report in the **Chat** view. Large datasets take time, and you can cancel from the progress notification. If one section fails the rest still runs and the report is marked partial.

## 4. What the report contains

The report covers:

- the detected task,
- dataset size and quality,
- prompt clarity,
- a hyperparameter review,
- model advice,
- the GPUs it detected,
- training time and cost estimates, with their assumptions,
- risks,
- prioritised recommendations,
- one concrete next experiment.

Each value is labelled *measured* (from your files), *calculated* (from a formula), *heuristic* (a rule of thumb) or *assumed*.

## 5. Chat

The **Chat** view shows the report and answers follow-up questions about your project. History is kept per project.

**Run Analyze Project first.** Chat answers from the analysis. Without one, the agent only has general knowledge and tells you to run it.

**Copying.** Every answer has a **Copy answer** button. Every code block has its own **Copy** button at the top. It copies only that snippet. The button shows **Copied ✓** when it worked.

**Asking it to do things.** The chat can act, not only explain. Type an instruction such as:

- "clean my dataset"
- "remove duplicates from my dataset"
- "analyze my project"

A question like "how do I remove duplicates?" is answered as a normal question.

If no AI provider is set up, or the provider fails, the chat tells you why. For example, with Ollama it tells you to run `ollama pull <model>`.

## 6. Clean Dataset

Cleans every dataset file in your project (`.jsonl`, `.json`, `.csv` and similar). Start it from the chat, or run `LLM Training Agent: Clean Dataset`, or use the button in Overview.

The agent asks you:

**How should I clean it?**

| Choice | What happens | Cost |
|---|---|---|
| **Use scripts** | Fixes extra whitespace and control characters. Nothing leaves your computer. | Free |
| **Use the AI model** | Sends the data to your provider in small chunks, so it can also fix typos and formatting. | Free with Ollama. With a paid API it uses your credits, so use it with care on big datasets. |

**Remove duplicate records?** Removes exact copies and keeps the first. This is a script step, so it is free in both modes. (If you already said "duplicates" in your message, it does not ask again.)

Then the agent:

1. cleans the data and shows a summary, for example `10 in → 8 out (2 duplicates removed)`,
2. puts the cleaned versions in `.llm-training-agent/cleaned/`,
3. prepares each changed file as a proposal. Choose **Review changes** to see the difference (section 8) and apply it.

Your original files are never overwritten without your approval. No record is lost: records out plus duplicates removed always equals records in. A file that is already clean is left alone. A file too big to review (over 2,000,000 characters) stays as a cleaned copy for you to copy over yourself.

## 7. Analyze Dataset

Run `LLM Training Agent: Analyze Dataset` and pick a folder. This checks the dataset only. For the full picture (model, hardware, cost) use Analyze Project.

## 8. View Changes

Run `LLM Training Agent: View Changes` to see edits the agent proposed. Each one opens as a side-by-side diff with three choices: **Approve & Apply**, **Reject** or **Dismiss**.

- Nothing is written without your approval.
- If you edited the file yourself in the meantime, the change is blocked instead of overwriting your work.
- Backups are kept so you can roll back.
- Proposals expire after 24 hours.

## 9. Reports

The **Reports** view, or `LLM Training Agent: View Report`, opens the latest saved report for the current project. If no analysis has run yet, it says so.

## 10. Choosing an AI provider

In the **Settings** view:

1. Pick a provider.
2. Type the model name (the box suggests some, and **Refresh Models** loads the list from the provider).
3. Paste your API key if the provider needs one. Ollama does not.
4. For Ollama or an OpenAI-compatible server, check the URL.
5. Press **Save**, wait for "Ready to use", then press **Test Connection**.

Supported: Ollama, OpenAI, Anthropic, Google Gemini, DeepSeek, Cohere, OpenRouter, and any OpenAI-compatible server (for example LM Studio).

About your key:

- It is stored in VS Code's secure storage, never in your project.
- After saving, the page only says a key is saved. It never shows the key again.
- You enter it once. It is remembered, including after you restart VS Code.
- **Test Connection** uses the saved key when the box is empty.
- **Remove API Key** deletes it.

## 11. Settings and stored data

Settings are user-level, so they are never saved inside your project: `llmTrainingAgent.provider`, `.model`, `.pythonPath`, `.backendUrl`, `.autoInstallDependencies`, `.ollamaBaseUrl` and `.openaiCompatibleBaseUrl`.

| What | Where |
|---|---|
| Analysis history (`llm_training_agent.db`) | VS Code global storage for the extension |
| Pending changes and backups | `.llm_training_agent/` and `.llm_training_agent_backups/` inside your project |
| Cleaned datasets | `.llm-training-agent/cleaned/` inside your project |

See [DELETION.md](DELETION.md) to clean these up.

## 12. Troubleshooting

**The backend will not start.** Open **View → Output** and choose **LLM Training Agent: Backend**. It lists the Python it tried and whether it was ready, missing packages, or too old. Usually the extension fixes this by itself by installing the missing packages. Developer tools (testing and linting) live only in the root `requirements.txt` and are never needed to run the extension.

To install by hand, run the command the channel prints under `Manual install command`. On Windows it looks like this (use your own Python path and extension folder):

```powershell
& "C:\Python312\python.exe" -m pip install -r "C:\Users\<you>\.vscode\extensions\rayansharma.llm-training-agent-1.0.0\backend\requirements.txt"
```

Then restart VS Code. To check that it worked, run `<your python> -c "import fastapi, uvicorn, sqlalchemy, httpx"`. No output means success. If you are working from a repository checkout, run `python -m pip install -r backend/requirements.txt` from the repository root.

**Buttons do nothing, or you see "Could not register service worker".** That message comes from VS Code itself and usually clears after **Developer: Reload Window**. If it does not, run **Developer: Open Webview Developer Tools** on that view. The Output channels name the command that failed (never your keys).

**Chat says the provider failed.** The message now says why. Common causes: the Ollama model is not downloaded (`ollama pull <model>`), Ollama is not running, the API key was rejected, or the model name is wrong. Check it in Settings.

**Chat says no provider is configured.** Choose one in Settings and press Test Connection.

**"No folder open" or path outside the folder.** Open your project folder first. Only that folder is read.

## 13. Limits

- The extension never invents accuracy scores.
- Model prices come from a bundled reference that may be out of date. Check costs before you spend.
- GPU detection is best-effort.
- Developed and tested mainly on Windows.
- Experiment tracking is available through the backend API only, and has no view yet.
