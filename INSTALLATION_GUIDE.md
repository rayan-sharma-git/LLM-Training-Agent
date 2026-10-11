# Installation Guide

Installing the LLM Training Agent VS Code extension.

This guide covers **normal extension installation only**. For what the extension does
once installed, see [USER_MANUAL.md](USER_MANUAL.md). For removing it, see
[DELETION.md](DELETION.md).

---

## 1. Where the VSIX is

The installable file sits **directly in the project root**:

```
llm-training-agent-1.0.0.vsix
```

**Download it from the project root.** You do not need to look inside `backend/`,
`extension/`, `build/`, `dist/`, or any other folder — the VSIX is the single file at
the top level of the repository, next to `README.md`.

If you are viewing the repository on GitHub in your browser (no cloning):

1. Open the `llm-training-agent-1.0.0.vsix` file in the repository's root directory.
2. Click **View Raw** to download the file.
3. Save it in any folder you prefer, such as **Downloads**. It can be on the C: drive, D: drive, or another location. **There is no required download folder or special path.** Just remember where you saved it so you can select the file during installation.
4. Downloading the file does not install the extension. You must still follow the VS Code **Extensions → three dots → Install from VSIX...** steps below and select the downloaded file.

If you are cloning the repository, the file is already there after cloning.

---

## 2. Prerequisites

To install, you need only:

- **VS Code 1.85.0 or newer.**

To run the extension, you also need:

- **A Python 3.10 or newer interpreter** on your machine. The extension runs its bundled backend with it, and most systems already have one. If the backend fails to start, the extension says so and offers to install the missing Python packages. You can point it at a specific interpreter with the `llmTrainingAgent.pythonPath` setting.

You do **not** need anything else to get started:

- You do **not** need to install the repository's `requirements.txt`. That file exists
  for backend development, not for using the extension.
- You do **not** need an API key. The default provider is **Ollama**, which runs models
  locally. An API key is only needed if you choose a hosted provider, and the
  extension tells you when it needs one.
- You do **not** need Node.js or npm.

---

## 3. Install the VSIX

1. Open **VS Code**.
2. Open the Extensions view (`Ctrl+Shift+X` on Windows/Linux, `Cmd+Shift+X` on macOS).
3. Click the **⋯** (Views and More Actions) button at the top-right of the Extensions
   view.
4. Choose **Install from VSIX…**
5. Select `llm-training-agent-1.0.0.vsix` from the project root.
6. Confirm, then wait for the installation to finish. If prompted, choose
   **Reload Window**.

---

## 4. Verify the installation

1. Look for the **LLM Training Agent** icon in the activity bar (left sidebar). It
   appears once the extension is installed and active.
2. Click it. You should see four views: **Overview**, **Chat**, **Reports** and
   **Settings**.
3. Open a project folder in VS Code, then run **LLM Training Agent: Analyze Project**
   from the command palette (`Ctrl+Shift+P`).

The first run starts the bundled backend and can take a little longer than usual. If a
notification appears, follow it — it names the actual problem.

---

## 5. First-time configuration (optional)

The extension works immediately with **Ollama**, the default provider. To use a hosted
provider instead, open the **Settings** view in the LLM Training Agent panel, choose a
provider, enter your API key, and press **Test Connection**. See
[USER_MANUAL.md](USER_MANUAL.md#7-configuring-an-ai-provider).

API keys are stored in VS Code SecretStorage, never in your project.
