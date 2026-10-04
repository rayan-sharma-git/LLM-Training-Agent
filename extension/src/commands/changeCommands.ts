import * as vscode from 'vscode';
import { ApiClient } from '../services/apiClient';
import { requireProjectRoot } from '../services/projectRoot';
import { formatError } from '../utils';

const VIEW_CHANGES_SCHEME = 'llm-training-agent-changes';

type ChangeStatus =
  | 'PROPOSED'
  | 'APPROVED'
  | 'REJECTED'
  | 'APPLIED'
  | 'FAILED'
  | 'CONFLICTED'
  | 'CANCELLED';

interface PendingChangeEntry {
  changeId: string;
  filePath: string;
  operation: string;
  status: ChangeStatus;
  additions?: number;
  deletions?: number;
  reason?: string;
  createdAt: string;
}

const STATUS_ICON: Record<ChangeStatus, string> = {
  PROPOSED: '$(git-compare)',
  APPROVED: '$(check)',
  APPLIED: '$(check)',
  REJECTED: '$(close)',
  FAILED: '$(error)',
  CONFLICTED: '$(warning)',
  CANCELLED: '$(circle-slash)',
};

/**
 * Virtual-document provider that surfaces the original and proposed file
 * contents in VS Code's native diff editor.
 */
class ChangeContentProvider implements vscode.TextDocumentContentProvider {
  private readonly contents = new Map<string, string>();

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.toString()) ?? '';
  }

  set(uri: vscode.Uri, content: string): void {
    this.contents.set(uri.toString(), content);
  }
}

/**
 * Registers the "View Changes" workflow:
 * list pending agent-made changes → inspect each as a native diff →
 * apply, discard, or roll back.
 */
export function registerChangeCommands(
  context: vscode.ExtensionContext,
  apiClient: ApiClient,
  ensureBackend?: () => Promise<boolean>
): void {
  const provider = new ChangeContentProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(VIEW_CHANGES_SCHEME, provider)
  );

  const viewChanges = vscode.commands.registerCommand(
    'llmTrainingAgent.viewChanges',
    async () => {
      // Workspace identity is resolved per invocation, never cached globally,
      // so switching folders cannot act on another workspace's proposals.
      const root = await requireProjectRoot();
      if (!root) {
        return;
      }

      if (ensureBackend) {
        await ensureBackend();
      }

      let changes: PendingChangeEntry[];
      try {
        changes = (await apiClient.listFileChanges(root)).changes ?? [];
      } catch (error) {
        vscode.window.showErrorMessage(`Could not list pending changes: ${formatError(error)}`);
        return;
      }

      if (changes.length === 0) {
        vscode.window.showInformationMessage(
          'No pending changes. The agent has not proposed any file modifications.'
        );
        return;
      }

      const picked = await vscode.window.showQuickPick(
        changes.map((change) => {
          const stats = [
            change.additions !== undefined ? `+${change.additions}` : '',
            change.deletions !== undefined ? `-${change.deletions}` : '',
          ]
            .filter(Boolean)
            .join(' ');
          return {
            label: `${STATUS_ICON[change.status] ?? ''} ${change.filePath}`.trim(),
            description: `${change.status} · ${change.operation}${stats ? ` · ${stats}` : ''}`,
            detail: change.reason || change.createdAt,
            change,
          };
        }),
        { placeHolder: 'View Changes — select a proposed change to inspect' }
      );
      if (!picked) {
        return;
      }

      const change = picked.change;
      let detail: any;
      try {
        detail = await apiClient.getFileChange(change.changeId, root);
      } catch (error) {
        vscode.window.showErrorMessage(`Could not load the change: ${formatError(error)}`);
        return;
      }

      // Show the proposal in VS Code's native diff editor (current vs. proposed).
      const originalUri = vscode.Uri.parse(
        `${VIEW_CHANGES_SCHEME}:${change.filePath}.original?${encodeURIComponent(change.changeId + ':original')}`
      );
      const proposedUri = vscode.Uri.parse(
        `${VIEW_CHANGES_SCHEME}:${change.filePath}.proposed?${encodeURIComponent(change.changeId + ':proposed')}`
      );
      provider.set(originalUri, detail.originalContent ?? '');
      // A delete proposal has no new content; show it as an empty right pane.
      provider.set(proposedUri, detail.proposedContent ?? '');

      await vscode.commands.executeCommand(
        'vscode.diff',
        originalUri,
        proposedUri,
        `View Changes — ${change.filePath} (current vs. proposed)`,
        { preview: true }
      );

      // A stale proposal must never be applied over newer user edits.
      if (detail.conflicted) {
        const regenerate = await vscode.window.showWarningMessage(
          `${change.filePath} changed on disk after this proposal was created. ` +
            'Applying it would overwrite those newer changes.',
          'Ask Agent to Regenerate',
          'Dismiss'
        );
        if (regenerate === 'Ask Agent to Regenerate') {
          await vscode.commands.executeCommand('llmTrainingAgent.openChat');
        }
        return;
      }

      // Only PROPOSED and APPROVED changes are actionable; anything else is
      // reported without offering a write path.
      if (change.status !== 'PROPOSED' && change.status !== 'APPROVED') {
        vscode.window.showInformationMessage(
          `${change.filePath} is ${change.status} and cannot be applied.`
        );
        return;
      }

      const isDestructive = change.operation === 'delete';
      const confirm = isDestructive
        ? `Permanently DELETE ${change.filePath}? A backup is kept for rollback.`
        : `Approve and apply this change to ${change.filePath}?`;

      const action = await vscode.window.showWarningMessage(
        confirm,
        { modal: true },
        'Approve & Apply',
        'Reject',
        change.status === 'APPROVED' ? 'Apply' : 'Dismiss'
      );

      try {
        if (action === 'Reject') {
          await apiClient.rejectFileChange(change.changeId, root);
          vscode.window.showInformationMessage(
            `Change rejected. ${change.filePath} was not modified.`
          );
          return;
        }
        if (action !== 'Approve & Apply' && action !== 'Apply') {
          return;
        }

        // Approve first when needed: the backend refuses to apply a proposal
        // that was not explicitly approved, so the UI must do this step.
        if (change.status === 'PROPOSED') {
          await apiClient.approveFileChange(change.changeId, root);
        }

        const result = await apiClient.applyFileChange(change.changeId, root);
        if (result.verified === false) {
          vscode.window.showErrorMessage(
            `The change was written to ${change.filePath} but could not be verified. ` +
              'Please check the file contents.'
          );
          return;
        }
        vscode.window.showInformationMessage(
          `Applied to ${change.filePath}. You can roll back from View Changes.`
        );
      } catch (error) {
        vscode.window.showErrorMessage(`Operation failed: ${formatError(error)}`);
      }
    }
  );

  context.subscriptions.push(viewChanges);
}
