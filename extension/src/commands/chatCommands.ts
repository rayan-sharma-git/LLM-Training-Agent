import * as vscode from 'vscode';
import { ApiClient } from '../services/apiClient';
import { SettingsManager } from '../services/settings';
import { CHAT_VIEW_ID } from '../views/viewIds';

/**
 * Registers the chat-related commands.
 * - llmTrainingAgent.openChat  : opens the Chat sidebar view.
 */
export function registerChatCommands(
  context: vscode.ExtensionContext,
  _apiClient: ApiClient,
  _settings: SettingsManager,
  _ensureBackend?: () => Promise<boolean>
): void {
  const openChatCommand = vscode.commands.registerCommand('llmTrainingAgent.openChat', async () => {
    // Reveal the chat view.
    await vscode.commands.executeCommand(`${CHAT_VIEW_ID}.focus`);
    // Fallback: reveal the container so the sidebar appears even if the
    // view command is not available yet.
    await vscode.commands.executeCommand('workbench.view.extension.llm-training-agent');
  });

  context.subscriptions.push(openChatCommand);
}