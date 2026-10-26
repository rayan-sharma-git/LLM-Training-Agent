import * as vscode from 'vscode';
import { SettingsManager } from './services/settings';
import { ApiClient } from './services/apiClient';
import { BackendManager } from './services/backendManager';
import { syncBackendConfig } from './services/backendSync';
import { registerAnalyzerCommands } from './commands/analyzerCommands';
import { registerChatCommands } from './commands/chatCommands';
import { registerChangeCommands } from './commands/changeCommands';
import { registerCleaningCommands } from './commands/cleaningCommands';
import { registerTreeView } from './views/simpleTreeView';
import { ChatWebviewProvider } from './views/chatWebviewProvider';
import { SettingsWebviewProvider } from './views/settingsWebviewProvider';
import { generateReportHtml, formatError } from './utils';
import { requireProjectRoot } from './services/projectRoot';
import { OVERVIEW_VIEW_ID, CHAT_VIEW_ID, REPORTS_VIEW_ID, SETTINGS_VIEW_ID } from './views/viewIds';

export async function activate(context: vscode.ExtensionContext) {
  try {
    // --- Services ---
    const settings = new SettingsManager(context);
    const config = settings.getSettings();

    // BackendManager resolves the bundled backend from the *extension install
    // directory*; the open workspace is only ever the project context.
    const backendManager = new BackendManager(context);
    const apiClient = new ApiClient(config.backendUrl, () => backendManager.getBaseUrl(), context);

    // Lazy startup: the backend is started on first use, not on activation.
    // This keeps activation cheap and avoids spawning Python for users who
    // never invoke the agent in a given window.
    //
    // Every call re-checks health (one cheap GET) instead of caching a past
    // success: a backend that crashed since the last click must be restarted,
    // not reported as running. Concurrent callers share one in-flight attempt.
    //
    // The backend keeps API keys in memory only, so whenever it is a new
    // process (or one this window has not configured yet) the saved provider,
    // model, URLs and keys are pushed to it from SecretStorage/settings.
    let inflight: Promise<boolean> | null = null;
    let syncedGeneration = -1;
    const ensureBackend = (): Promise<boolean> => {
      if (!inflight) {
        inflight = (async () => {
          try {
            const ok = await backendManager.ensureRunning();
            if (!ok) {
              vscode.window.showWarningMessage(
                'LLM Training Agent: the Python backend could not be started. ' +
                  'Check the "LLM Training Agent: Backend" output channel. ' +
                  'If Python dependencies are missing, run "Analyze Project" to be ' +
                  'offered the dependency install.'
              );
              return false;
            }
            if (syncedGeneration !== backendManager.generation) {
              const current = settings.getSettings();
              const result = await syncBackendConfig(current, {
                getApiKey: (provider) => apiClient.getApiKey(provider),
                updateConfig: (payload) => apiClient.updateConfig(payload),
                selectProvider: (provider, model) => apiClient.selectProvider(provider, model),
                log: (line) => backendManager.logLine(line),
              });
              syncedGeneration = backendManager.generation;
              if (!result.ok) {
                vscode.window.showWarningMessage(
                  `LLM Training Agent: could not apply your saved AI settings to the backend (${result.error}). ` +
                    'Open the AI settings and press Save to retry.'
                );
              }
            }
            return true;
          } finally {
            inflight = null;
          }
        })();
      }
      return inflight;
    };

    // Register shutdown so the child process is cleaned up when VS Code closes.
    context.subscriptions.push({
      dispose: () => {
        backendManager.shutdown();
      },
    });

    // When the user switches projects (or adds/removes folders), the set of
    // authorized roots changes. Re-publishing them lets the already-running
    // backend follow the switch immediately — no reload, no reinstall, and no
    // chance of Project A's data being served while Project B is open.
    context.subscriptions.push(
      vscode.workspace.onDidChangeWorkspaceFolders(() => {
        backendManager.refreshAllowedRoots();
      })
    );

    // --- Overview tree view ---
    const overviewProvider = registerTreeView(context, OVERVIEW_VIEW_ID, [
      { label: 'Project: No project analyzed yet' },
      { label: 'Provider: ' + config.provider },
      { label: 'Model: ' + config.model },
    ]);

    // --- Reports tree view ---
    registerTreeView(context, REPORTS_VIEW_ID, [
      { label: 'No report available' },
      {
        label: 'Generate a report',
        command: {
          command: 'llmTrainingAgent.analyzeProject',
          title: 'Analyze Project',
        },
      },
    ]);

    // --- Chat webview view ---
    const chatProvider = new ChatWebviewProvider(context.extensionUri, apiClient, ensureBackend);
    context.subscriptions.push(
      vscode.window.registerWebviewViewProvider(ChatWebviewProvider.viewType, chatProvider, {
        webviewOptions: { retainContextWhenHidden: true },
      })
    );

    // --- Settings webview view ---
    // Pass `ensureBackend` so saving settings starts the bundled backend on
    // first use, exactly like the chat view already does.
    const settingsProvider = new SettingsWebviewProvider(context.extensionUri, apiClient, settings, ensureBackend);
    context.subscriptions.push(
      vscode.window.registerWebviewViewProvider(SettingsWebviewProvider.viewType, settingsProvider, {
        webviewOptions: { retainContextWhenHidden: true },
      })
    );

    // --- Commands ---
    registerAnalyzerCommands(context, apiClient, settings, overviewProvider, chatProvider, ensureBackend, backendManager);
    registerChatCommands(context, apiClient, settings, ensureBackend);
    registerChangeCommands(context, apiClient, ensureBackend);
    registerCleaningCommands(context, apiClient, settings, chatProvider, ensureBackend);
    context.subscriptions.push(
      vscode.commands.registerCommand('llmTrainingAgent.configureProvider', async () => {
        await settingsProvider.reveal();
      })
    );

    // Analyze Dataset command.
    const analyzeDatasetCmd = vscode.commands.registerCommand(
      'llmTrainingAgent.analyzeDataset',
      async () => {
        if (!vscode.workspace.workspaceFolders || vscode.workspace.workspaceFolders.length === 0) {
          vscode.window.showErrorMessage(
            'LLM Training Agent: no folder is open. Open the project containing the dataset and run the command again.'
          );
          return;
        }

        // Start the backend on demand.
        await ensureBackend();

        const datasetPath = await vscode.window.showOpenDialog({
          canSelectFiles: false,
          canSelectFolders: true,
          canSelectMany: false,
          title: 'Select Dataset Directory',
          openLabel: 'Analyze',
        });

        if (!datasetPath || datasetPath.length === 0) {
          return;
        }

        await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: 'Analyzing dataset...',
            cancellable: true,
          },
          async (_progress, token) => {
            try {
              _progress.report({ increment: 0, message: 'Starting analysis' });
              if (token.isCancellationRequested) {
                return;
              }
              await apiClient.analyzeDataset(datasetPath[0].fsPath);
              _progress.report({ increment: 80, message: 'Analysis complete' });
              vscode.window.showInformationMessage('Dataset analysis complete');
            } catch (error) {
              vscode.window.showErrorMessage(`Dataset analysis failed: ${formatError(error)}`);
            }
          }
        );
      }
    );
    context.subscriptions.push(analyzeDatasetCmd);

    // View Report command.
    const viewReportCmd = vscode.commands.registerCommand(
      'llmTrainingAgent.viewReport',
      async () => {
        // The report belongs to the currently open project, so the root is
        // resolved per invocation — never cached across projects.
        const projectRoot = await requireProjectRoot();
        if (!projectRoot) {
          return;
        }
        try {
          await ensureBackend();
          const report = await apiClient.getReport(projectRoot);
          if (!report) {
            vscode.window.showInformationMessage('No report available for this project');
            return;
          }
          const panel = vscode.window.createWebviewPanel(
            'llmTrainingAgent.report',
            'Report',
            vscode.ViewColumn.One,
            { enableScripts: true }
          );
          panel.webview.html = generateReportHtml(report);
        } catch (error) {
          vscode.window.showErrorMessage(`Failed to load report: ${formatError(error)}`);
        }
      }
    );
    context.subscriptions.push(viewReportCmd);
    // Prepare the backend as soon as the user opens the extension, so the
    // one-time "install the Python packages?" prompt appears right away rather
    // than on their first message. Runs in the background; failures are
    // reported by ensureBackend itself.
    void ensureBackend().catch((error) => {
      console.error('Backend preparation failed:', error instanceof Error ? error.message : String(error));
    });
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to activate LLM Training Agent: ${formatError(error)}`);
    console.error('Extension activation failed:', error);
  }
}

export function deactivate() {}