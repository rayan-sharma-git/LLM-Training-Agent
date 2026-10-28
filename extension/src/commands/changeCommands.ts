import * as vscode from 'vscode';
import { ChatWebviewProvider } from '../views/chatWebviewProvider';

/**
 * View Changes.
 *
 * Opens the chat and posts the latest changes there as an inline viewer: the file path, then
 * one bordered block per change with the removed lines in red (labelled "Removed") and the
 * replacement lines in green (labelled "Added"). Nothing is opened in a separate editor and
 * nothing is applied from here; applying and reverting happen through the chat.
 */
export function registerChangeCommands(
  context: vscode.ExtensionContext,
  chatProvider: ChatWebviewProvider
): void {
  const viewChanges = vscode.commands.registerCommand('llmTrainingAgent.viewChanges', async () => {
    await vscode.commands.executeCommand('workbench.view.extension.llm-training-agent');
    await chatProvider.showChanges();
  });
  context.subscriptions.push(viewChanges);
}
