/**
 * Pure path/state resolution helpers.
 *
 * These functions deliberately contain no `vscode` import so they can be unit
 * tested directly. They encode the single most important invariant of this
 * extension:
 *
 *   EXTENSION INSTALL PATH  !=  CURRENT WORKSPACE PATH
 *
 * Extension-owned resources (the bundled Python backend, the bundled prompts,
 * the packaged runtime) are only ever resolved from the VS Code extension
 * installation directory. The user's workspace is a completely separate input
 * that is resolved fresh on every operation and is never cached globally.
 */
import * as path from 'path';

/** Filesystem probe, injected so resolution can be tested without touching disk. */
export type ExistsSync = (candidate: string) => boolean;

/** Shape of the workspace-folder information we need, decoupled from `vscode`. */
export interface WorkspaceFolderLike {
  uri: { fsPath: string };
  name?: string;
}

/** Directory name of the Python backend bundled inside the extension. */
export const BACKEND_DIR_NAME = 'backend';

/** Entry point of the bundled backend, relative to the extension root. */
export const BACKEND_ENTRY = 'main.py';

/**
 * Resolve the bundled backend entry point.
 *
 * ONLY the extension installation directory is consulted. The user's open
 * project is never a candidate: a project that happens to contain
 * `backend/main.py` must not be able to replace the agent's own backend.
 */
export function resolveBackendEntry(extensionPath: string, exists: ExistsSync): string | null {
  const candidates = [path.join(extensionPath, BACKEND_DIR_NAME, BACKEND_ENTRY)];
  for (const candidate of candidates) {
    if (exists(candidate)) {
      return candidate;
    }
  }
  return null;
}

/** Directory holding the bundled backend (its parent), or null when absent. */
export function resolveBackendDir(extensionPath: string, exists: ExistsSync): string | null {
  const entry = resolveBackendEntry(extensionPath, exists);
  return entry ? path.dirname(entry) : null;
}

/** Platform-specific executables searched for inside a bundled Python folder. */
function bundledPythonCandidates(platform: NodeJS.Platform): string[] {
  return platform === 'win32'
    ? ['python.exe', 'Scripts/python.exe', 'bin/python.exe']
    : ['bin/python3', 'bin/python'];
}

/**
 * Resolve the Python interpreter used to run the bundled backend.
 *
 * Order:
 *   1. The `llmTrainingAgent.pythonPath` setting (user-level, global).
 *   2. A Python interpreter bundled *inside the extension* — used by
 *      self-contained builds. Never looked up relative to the repository.
 *   3. `python` / `python3` from `PATH`.
 *
 * There is deliberately no search of `..`, of the repository root, or of the
 * open workspace: the extension must behave identically no matter which project
 * happens to be open.
 */
export function resolvePythonExecutable(options: {
  extensionPath: string;
  configuredPath?: string;
  exists: ExistsSync;
  platform: NodeJS.Platform;
}): string {
  const { extensionPath, configuredPath, exists, platform } = options;

  const configured = configuredPath?.trim();
  if (configured) {
    return configured;
  }

  for (const relative of bundledPythonCandidates(platform)) {
    const candidate = path.join(extensionPath, 'python', relative);
    if (exists(candidate)) {
      return candidate;
    }
  }

  return platform === 'win32' ? 'python' : 'python3';
}

/** Result of inspecting the current workspace. */
export type WorkspaceState =
  | { kind: 'none'; folders: [] }
  | { kind: 'single'; folders: [string] }
  | { kind: 'multi'; folders: string[] };

/**
 * Classify the currently open workspace.
 *
 * Resolved fresh on every call so that switching from Project A to Project B
 * (or closing all folders) is always observed immediately.
 */
export function resolveWorkspaceState(
  workspaceFolders: readonly WorkspaceFolderLike[] | undefined
): WorkspaceState {
  const folders = (workspaceFolders ?? [])
    .map((folder) => folder.uri.fsPath)
    .filter((fsPath) => typeof fsPath === 'string' && fsPath.length > 0);

  if (folders.length === 0) {
    return { kind: 'none', folders: [] };
  }
  if (folders.length === 1) {
    return { kind: 'single', folders: [folders[0]] };
  }
  return { kind: 'multi', folders };
}

/** All open workspace folders as absolute paths (empty when none are open). */
export function listWorkspaceRoots(
  workspaceFolders: readonly WorkspaceFolderLike[] | undefined
): string[] {
  return resolveWorkspaceState(workspaceFolders).folders as string[];
}

/** True when the workspace holds more than one folder (multi-root). */
export function isMultiRoot(
  workspaceFolders: readonly WorkspaceFolderLike[] | undefined
): boolean {
  return resolveWorkspaceState(workspaceFolders).kind === 'multi';
}

/** File name used to share the authorized roots with the backend process. */
export const ALLOWED_ROOTS_FILE = 'allowed-roots.txt';

/**
 * Publish the currently open project roots for the backend to read.
 *
 * The backend authorizes exactly the folders that are open right now. Writing
 * them to a file (rather than only to the process environment) means a
 * Project A -> Project B switch is picked up by the running backend, so the
 * user never has to reload the window or reinstall anything.
 *
 * The file lives in the extension's *global storage*, not in any project.
 */
export function publishAllowedRoots(options: {
  globalStoragePath: string;
  workspaceFolders: readonly WorkspaceFolderLike[] | undefined;
  writeFile: (filePath: string, contents: string) => void;
  delimiter: string;
}): void {
  const { globalStoragePath, workspaceFolders, writeFile, delimiter } = options;
  const roots = listWorkspaceRoots(workspaceFolders);
  const filePath = path.join(globalStoragePath, ALLOWED_ROOTS_FILE);
  // An empty file denies every project, which is the correct behaviour when no
  // folder is open: the agent must not fall back to its own directory.
  writeFile(filePath, roots.join(delimiter));
}
