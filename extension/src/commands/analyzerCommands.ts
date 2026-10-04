import * as vscode from 'vscode';
import { ApiClient } from '../services/apiClient';
import { SettingsManager } from '../services/settings';
import { SimpleTreeDataProvider } from '../views/simpleTreeView';
import { ChatWebviewProvider } from '../views/chatWebviewProvider';
import { BackendManager } from '../services/backendManager';
import { requireProjectRoot } from '../services/projectRoot';
import { checkDependencies, installDependencies, DependencyCheckResult } from '../services/dependencyManager';

/**
 * Formats and displays GPU training estimate in the chat.
 */
function formatGpuEstimateForChat(gpuEstimate: any): string {
  if (!gpuEstimate) {
    return '';
  }
  const lines: string[] = [];
  lines.push('## 🎮 GPU TRAINING ESTIMATE');
  lines.push('');

  const gpuInfo = gpuEstimate.gpu_info || {};
  const gpus = gpuInfo.gpus || [];
  if (gpus.length > 0) {
    for (const gpu of gpus) {
      lines.push(`**GPU ${gpu.index}:** ${gpu.name || 'Unknown'}`);
      if (gpu.vram_gb) {
        lines.push(`**VRAM:** ${gpu.vram_gb} GB`);
      }
      if (gpu.cuda_version) {
        lines.push(`**CUDA:** ${gpu.cuda_version}`);
      }
    }
  } else {
    lines.push('⚠ No supported GPU detected. GPU time estimation may be unavailable or less accurate.');
  }

  lines.push('');
  lines.push(`**Estimated training time:** ~${gpuEstimate.estimated_time || 'unknown'}`);
  if (gpuEstimate.range) {
    lines.push(`**Likely range:** ${gpuEstimate.range}`);
  }
  lines.push(`**Confidence:** ${gpuEstimate.confidence || 'low'}`);
  if (gpuEstimate.mode) {
    lines.push(`**Mode:** ${gpuEstimate.mode === 'calibrated' ? 'Calibrated' : 'Quick'} estimate`);
  }
  if (gpuEstimate.throughput_steps_per_sec) {
    lines.push(`**Throughput:** ${gpuEstimate.throughput_steps_per_sec.toFixed(2)} steps/sec`);
  }
  if (gpuEstimate.total_steps) {
    lines.push(`**Total steps:** ${gpuEstimate.total_steps}`);
  }
  lines.push(`**VRAM:** ${gpuEstimate.vram_feasible ? '✓ Likely sufficient' : '⚠ May exceed available VRAM'}`);

  if (gpuEstimate.warnings && gpuEstimate.warnings.length > 0) {
    lines.push('');
    lines.push('**Warnings:**');
    for (const w of gpuEstimate.warnings) {
      lines.push(`- ${w}`);
    }
  }

  if (gpuEstimate.assumptions && gpuEstimate.assumptions.length > 0) {
    lines.push('');
    lines.push('**Assumptions:**');
    for (const a of gpuEstimate.assumptions) {
      lines.push(`- ${a}`);
    }
  }

  return lines.join('\n');
}

/**
 * Build the human-readable message shown to the user when dependencies are missing.
 */
function buildMissingDepsMessage(result: DependencyCheckResult): string {
  const missingList = result.missing.length > 0
    ? result.missing.join(', ')
    : 'one or more packages';
  return [
    '⚠️ Required backend dependencies are missing.',
    '',
    `Missing: ${missingList}`,
    '',
    'Analyze Project cannot start until they are installed.',
    '',
    'Would you like me to install the required dependencies?',
    '',
    'Manual command:',
    '```',
    result.manualCommand,
    '```',
  ].join('\n');
}

/**
 * Ask the user for permission to install missing dependencies.
 *
 * Prefers the chat window (action card with [Install Dependencies] [Cancel]).
 * Falls back to a modal VS Code dialog if the chat view is not available.
 *
 * Returns 'install' | 'cancel' | '' (no decision).
 */
async function askPermissionToInstall(
  chatProvider: ChatWebviewProvider | null,
  result: DependencyCheckResult
): Promise<'install' | 'cancel' | ''> {
  const message = buildMissingDepsMessage(result);

  // Prefer the chat window action card.
  if (chatProvider) {
    const actionId = `dep-install-${Date.now()}`;
    const choice = await chatProvider.postActionCard({
      text: message,
      actionId,
      buttons: [
        { label: 'Install Dependencies', action: 'install' },
        { label: 'Cancel', action: 'cancel' },
      ],
    });
    if (choice === 'install') {
      return 'install';
    }
    if (choice === 'cancel') {
      return 'cancel';
    }
    // No decision from chat (view not available) — fall through to modal.
  }

  // Fallback: modal dialog.
  const selection = await vscode.window.showWarningMessage(
    message,
    { modal: true },
    'Install Dependencies',
    'Cancel'
  );
  if (selection === 'Install Dependencies') {
    return 'install';
  }
  if (selection === 'Cancel') {
    return 'cancel';
  }
  return '';
}

/**
 * Registers the project analysis command.
 * - llmTrainingAgent.analyzeProject : checks backend dependencies, then scans
 *   the currently open project and refreshes the Overview view.
 */
export function registerAnalyzerCommands(
  context: vscode.ExtensionContext,
  apiClient: ApiClient,
  settings: SettingsManager,
  overviewProvider: SimpleTreeDataProvider,
  chatProvider: ChatWebviewProvider | null = null,
  ensureBackend?: () => Promise<boolean>,
  backendManager?: BackendManager
): void {
  const analyzeCommand = vscode.commands.registerCommand(
    'llmTrainingAgent.analyzeProject',
    async () => {
      // The project to analyze is resolved dynamically on every invocation:
      // no folder -> explicit message, multi-root -> explicit user choice.
      const projectPath = await requireProjectRoot();
      if (!projectPath) {
        return;
      }

      // Ensure the path exists before sending to the backend.
      const fs = require('fs') as typeof import('fs');
      if (!fs.existsSync(projectPath)) {
        vscode.window.showErrorMessage(
          `Project path does not exist: ${projectPath}`
        );
        return;
      }

      // ------------------------------------------------------------------
      // STEP 1: Check backend dependencies (read-only).
      // ------------------------------------------------------------------
      if (backendManager) {
        const backendPath = backendManager.getBackendPath();
        const pythonExe = backendManager.getPythonExecutable();

        if (backendPath) {
          const depResult = await checkDependencies({
            pythonExe,
            backendMainPath: backendPath,
          });

          if (!depResult.available) {
            // STOP — do not start analysis. Ask for permission.
            const choice = await askPermissionToInstall(chatProvider, depResult);

            if (choice === 'install') {
              // User explicitly approved installation.
              const outputChannel = vscode.window.createOutputChannel('LLM Training Agent: Dependencies');
              const backendDir = require('path').dirname(backendPath) as string;

              const installOk = await new Promise<boolean>((resolve) => {
                installDependencies({
                  pythonExe,
                  requirementsTxtPath: depResult.requirementsTxtPath || '',
                  backendDir,
                  outputChannel,
                  onDone: (success) => resolve(success),
                });
              });

              if (!installOk) {
                vscode.window.showErrorMessage(
                  'Dependency installation failed. Check the "LLM Training Agent: Dependencies" output channel.'
                );
                return;
              }

              // Verify again after installation.
              const recheck = await checkDependencies({
                pythonExe,
                backendMainPath: backendPath,
              });
              if (!recheck.available) {
                vscode.window.showErrorMessage(
                  `Dependencies were installed but are still not importable: ${recheck.missing.join(', ')}`
                );
                return;
              }

              if (chatProvider) {
                chatProvider.postAssistantMessage(
                  '✅ Backend dependencies are now available. Starting analysis...'
                );
              }
            } else {
              // Cancel (or no decision) — do nothing, do not start analysis.
              if (chatProvider) {
                chatProvider.postAssistantMessage(
                  'Analyze Project was cancelled because the required backend dependencies are missing.\n\n' +
                  `You can install them manually with:\n\`\`\`\n${depResult.manualCommand}\n\`\`\``
                );
              }
              vscode.window.showWarningMessage(
                'Analyze Project was cancelled because the required backend dependencies are missing.'
              );
              return;
            }
          }
        }
      }

      // ------------------------------------------------------------------
      // STEP 2: Wait for the backend to be ready before sending the request.
      // ------------------------------------------------------------------
      if (ensureBackend) {
        const ok = await ensureBackend();
        if (!ok) {
          vscode.window.showErrorMessage(
            'The Python backend could not be started. ' +
              'Check the "LLM Training Agent: Backend" output channel for details.'
          );
          return;
        }
      }

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'Analyzing project...',
          cancellable: true,
        },
        async (progress, token) => {
          try {
            progress.report({ increment: 0, message: 'Contacting backend' });
            if (token.isCancellationRequested) {
              return;
            }
            progress.report({ increment: 20, message: 'Scanning project' });
            const result = await apiClient.analyzeProject(projectPath);
            progress.report({ increment: 80, message: 'Generating report' });
            vscode.window.showInformationMessage('Analysis complete');

            // Refresh the overview tree view with analysis results.
            overviewProvider.refresh([
              { label: 'Project: ' + projectPath },
              { label: 'Provider: ' + settings.getSettings().provider },
              { label: 'Model: ' + settings.getSettings().model },
            ]);

            // Store the result in WORKSPACE state, scoped to this project.
            // Global state would leak Project A's report into Project B.
            await context.workspaceState.update('lastAnalysisResult', {
              projectPath,
              result,
            });

            // Post the COMPLETE end-to-end analysis to the chat window.
            // The backend composes every real analyzer output into one
            // structured report, so the user receives the full answer in a
            // single click — no follow-up questions required.
            if (chatProvider && result) {
              const analysis = result.analysis;
              if (analysis && analysis.markdown) {
                chatProvider.postAssistantMessage(analysis.markdown);
              } else if (result.report && result.report.gpu_time_estimate) {
                // Graceful degradation: an older backend without the composed
                // analysis still shows the GPU estimate rather than nothing.
                chatProvider.postAssistantMessage(
                  formatGpuEstimateForChat(result.report.gpu_time_estimate)
                );
              }
            }
          } catch (error) {
            // Use the ApiError message (already human-friendly).
            const err = error instanceof Error ? error : new Error(String(error));
            vscode.window.showErrorMessage(`Analysis failed: ${err.message}`);
            console.error('Analyze project error:', err);
            // Surface the failure in the chat too, so the user is not left
            // waiting on a notification they may have missed.
            if (chatProvider) {
              chatProvider.postAssistantMessage(
                `⚠️ Analysis failed: ${err.message}`
              );
            }
          }
        }
      );
    }
  );

  context.subscriptions.push(analyzeCommand);
}