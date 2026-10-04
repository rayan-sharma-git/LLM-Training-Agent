/**
 * Project-root selection.
 *
 * The extension is installed once and shared by every project, so the project
 * to operate on must be resolved *dynamically* on each operation and must never
 * be cached in global state (that would leak Project A's identity into
 * Project B).
 */
import * as vscode from 'vscode';
import * as path from 'path';
import {
  resolveWorkspaceState,
  WorkspaceFolderLike,
  WorkspaceState,
} from './extensionPaths';

export type ProjectRootResult =
  | { status: 'ok'; root: string }
  | { status: 'no-workspace' }
  | { status: 'cancelled' };

/** Pickable entry for a multi-root workspace. */
interface RootQuickPickItem extends vscode.QuickPickItem {
  root: string;
}

/**
 * Resolve the project root for a project-scoped operation.
 *
 * Behaviour:
 *  - No folder open  -> explicit `no-workspace` result (never a silent guess).
 *  - Single folder   -> that folder.
 *  - Multi-root      -> the user picks explicitly, so the agent can never
 *                       silently analyze the wrong folder.
 */
export async function selectProjectRoot(
  workspaceFolders: readonly WorkspaceFolderLike[] | undefined = vscode.workspace.workspaceFolders
): Promise<ProjectRootResult> {
  const state: WorkspaceState = resolveWorkspaceState(workspaceFolders);

  if (state.kind === 'none') {
    return { status: 'no-workspace' };
  }

  if (state.kind === 'single') {
    return { status: 'ok', root: state.folders[0] };
  }

  const items: RootQuickPickItem[] = state.folders.map((root) => ({
    label: path.basename(root) || root,
    description: root,
    root,
  }));

  const picked = await vscode.window.showQuickPick(items, {
    title: 'Select the project to analyze',
    placeHolder: 'This workspace contains multiple folders. Choose one.',
    ignoreFocusOut: true,
  });

  if (!picked) {
    return { status: 'cancelled' };
  }
  return { status: 'ok', root: picked.root };
}

/**
 * Resolve the project root, showing a user-facing message for the states the
 * user must act on. Returns undefined when the operation should be aborted.
 */
export async function requireProjectRoot(
  workspaceFolders: readonly WorkspaceFolderLike[] | undefined = vscode.workspace.workspaceFolders
): Promise<string | undefined> {
  const result = await selectProjectRoot(workspaceFolders);

  if (result.status === 'ok') {
    return result.root;
  }

  if (result.status === 'no-workspace') {
    vscode.window.showErrorMessage(
      'LLM Training Agent: no folder is open. Open the project you want to analyze and run the command again.'
    );
  }

  // 'cancelled' is a deliberate user choice — stay silent.
  return undefined;
}
