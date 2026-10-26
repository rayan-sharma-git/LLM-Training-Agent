import * as vscode from 'vscode';
import * as fs from 'fs';
import { ApiClient } from '../services/apiClient';
import { SettingsManager } from '../services/settings';
import { ChatWebviewProvider } from '../views/chatWebviewProvider';
import { requireProjectRoot } from '../services/projectRoot';
import { runCleaningFlow, CleaningDeps } from '../services/cleaningFlow';
import { runEditFlow, EditDeps } from '../services/editFlow';

async function readText(file: string): Promise<string | undefined> {
  try {
    return await fs.promises.readFile(file, 'utf8');
  } catch {
    return undefined;
  }
}

/** Wire the cleaning flow to VS Code and the chat view. */
export function createCleaningDeps(
  apiClient: ApiClient,
  settings: SettingsManager,
  chat: ChatWebviewProvider,
  ensureBackend: () => Promise<boolean>
): CleaningDeps {
  return {
    getProjectRoot: () => requireProjectRoot(),
    ensureBackend,
    ask: async (text, buttons) => {
      const fromChat = await chat.postActionCard({ text, actionId: `ask-${Date.now()}`, buttons });
      if (fromChat) {
        return fromChat;
      }
      // The chat view is not available: fall back to a dialog.
      const pick = await vscode.window.showInformationMessage(text, { modal: true }, ...buttons.map((b) => b.label));
      return buttons.find((b) => b.label === pick)?.action ?? '';
    },
    say: (text) => chat.postAssistantMessage(text),
    withProgress: (title, task) =>
      Promise.resolve(vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title }, () => task())),
    clean: (options) => apiClient.cleanDatasets(options),
    readText,
    propose: (relativePath, content, root, reason) =>
      apiClient.proposeFileChange(relativePath, content, root, 'modify', reason),
    openChanges: async () => {
      await vscode.commands.executeCommand('llmTrainingAgent.viewChanges');
    },
    providerLabel: () => {
      const current = settings.getSettings();
      return `${current.provider} / ${current.model}`;
    },
    isLocalProvider: () => settings.getSettings().provider === 'ollama',
  };
}

/** Wire the edit flow to VS Code and the chat view. */
export function createEditDeps(
  apiClient: ApiClient,
  settings: SettingsManager,
  chat: ChatWebviewProvider,
  ensureBackend: () => Promise<boolean>,
  base: CleaningDeps
): EditDeps {
  return {
    getProjectRoot: base.getProjectRoot,
    ensureBackend,
    findFile: async (name) => {
      const normalized = name.replace(/\\/g, '/');
      const found = await vscode.workspace.findFiles(`**/${normalized}`, '**/{node_modules,.git,venv,.venv,__pycache__,out,dist}/**', 10);
      return found.map((uri) => vscode.workspace.asRelativePath(uri, false));
    },
    suggest: (relativePath, instruction, root) => apiClient.suggestFileEdit(relativePath, instruction, root),
    propose: (relativePath, content, root, reason) => apiClient.proposeFileChange(relativePath, content, root, 'modify', reason),
    ask: base.ask,
    say: base.say,
    withProgress: base.withProgress,
    openChanges: base.openChanges,
    providerLabel: base.providerLabel,
  };
}

/** Registers `llmTrainingAgent.cleanDataset` and lets the chat trigger the same flow. */
export function registerCleaningCommands(
  context: vscode.ExtensionContext,
  apiClient: ApiClient,
  settings: SettingsManager,
  chat: ChatWebviewProvider,
  ensureBackend: () => Promise<boolean>
): void {
  const deps = createCleaningDeps(apiClient, settings, chat, ensureBackend);
  const editDeps = createEditDeps(apiClient, settings, chat, ensureBackend, deps);

  context.subscriptions.push(
    vscode.commands.registerCommand('llmTrainingAgent.cleanDataset', async () => {
      await chat.reveal().then(undefined, () => undefined);
      await runCleaningFlow(deps);
    })
  );

  chat.setIntentHandler(async (intent) => {
    if (intent.kind === 'clean') {
      await runCleaningFlow(deps, { removeDuplicates: intent.removeDuplicates ? true : undefined });
    } else if (intent.kind === 'edit') {
      await runEditFlow(editDeps, { file: intent.file, instruction: intent.instruction });
    } else if (intent.kind === 'analyze') {
      await vscode.commands.executeCommand('llmTrainingAgent.analyzeProject');
    }
  });
}
