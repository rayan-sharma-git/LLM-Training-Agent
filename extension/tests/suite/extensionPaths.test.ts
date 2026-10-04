/**
 * Tests for the extension-install-path vs workspace-path separation.
 *
 * These are the highest-value regression guards for the "install once, use in
 * every project" requirement: extension resources must resolve from the
 * installation directory, and the project context must resolve dynamically.
 */
import { describe, it, expect, vi } from 'vitest';
import * as path from 'path';
import {
  resolveBackendEntry,
  resolveBackendDir,
  resolvePythonExecutable,
  resolveWorkspaceState,
  listWorkspaceRoots,
  isMultiRoot,
  publishAllowedRoots,
  ALLOWED_ROOTS_FILE,
  WorkspaceFolderLike,
} from '../../src/services/extensionPaths';

/** Simulate a VS Code installation directory that is NOT the repo. */
const INSTALL_DIR = path.join(
  'C:', 'Users', 'dev', '.vscode', 'extensions',
  'rayansharma.llm-training-agent-1.0.0'
);

/** Two unrelated user projects. */
const PROJECT_A = path.join('C:', 'Projects', 'ProjectA');
const PROJECT_B = path.join('C:', 'Projects', 'ProjectB');

const folder = (fsPath: string): WorkspaceFolderLike => ({ uri: { fsPath } });

/** existsSync that only knows about the paths present in `existing`. */
const existsIn = (existing: string[]) => (p: string) => existing.includes(p);

describe('extension install path resolution', () => {
  it('resolves the bundled backend from the installation directory', () => {
    const backendEntry = path.join(INSTALL_DIR, 'backend', 'main.py');
    expect(resolveBackendEntry(INSTALL_DIR, existsIn([backendEntry]))).toBe(backendEntry);
  });

  it('resolves the backend identically regardless of the open project', () => {
    // The resolver takes no workspace argument at all, which is the invariant.
    const backendEntry = path.join(INSTALL_DIR, 'backend', 'main.py');
    const exists = existsIn([backendEntry]);
    expect(resolveBackendEntry(INSTALL_DIR, exists)).toBe(
      resolveBackendEntry(INSTALL_DIR, exists)
    );
  });

  it('never falls back to a backend inside the open project', () => {
    // A project that happens to contain backend/main.py must NOT be used.
    const projectBackend = path.join(PROJECT_A, 'backend', 'main.py');
    expect(resolveBackendEntry(INSTALL_DIR, existsIn([projectBackend]))).toBeNull();
  });

  it('returns null when the installation is missing the backend', () => {
    expect(resolveBackendEntry(INSTALL_DIR, existsIn([]))).toBeNull();
  });

  it('exposes the backend directory, not the entry file', () => {
    const backendEntry = path.join(INSTALL_DIR, 'backend', 'main.py');
    expect(resolveBackendDir(INSTALL_DIR, existsIn([backendEntry]))).toBe(
      path.join(INSTALL_DIR, 'backend')
    );
  });
});

describe('python interpreter resolution', () => {
  it('prefers the configured user-level interpreter', () => {
    expect(
      resolvePythonExecutable({
        extensionPath: INSTALL_DIR,
        configuredPath: 'C:\\Python312\\python.exe',
        exists: existsIn([]),
        platform: 'win32',
      })
    ).toBe('C:\\Python312\\python.exe');
  });

  it('ignores a blank configured value', () => {
    expect(
      resolvePythonExecutable({
        extensionPath: INSTALL_DIR,
        configuredPath: '   ',
        exists: existsIn([]),
        platform: 'win32',
      })
    ).toBe('python');
  });

  it('uses an interpreter bundled inside the extension when present', () => {
    const bundled = path.join(INSTALL_DIR, 'python', 'python.exe');
    expect(
      resolvePythonExecutable({
        extensionPath: INSTALL_DIR,
        exists: existsIn([bundled]),
        platform: 'win32',
      })
    ).toBe(bundled);
  });

  it('never searches repository-relative or workspace-relative locations', () => {
    const repoVenv = path.join('C:', 'dev', 'repo', 'venv310', 'Scripts', 'python.exe');
    const projectVenv = path.join(PROJECT_A, 'venv310', 'Scripts', 'python.exe');
    const resolved = resolvePythonExecutable({
      extensionPath: INSTALL_DIR,
      exists: existsIn([repoVenv, projectVenv]),
      platform: 'win32',
    });
    expect(resolved).not.toBe(repoVenv);
    expect(resolved).not.toBe(projectVenv);
    expect(resolved).toBe('python');
  });

  it('falls back to python3 on non-Windows platforms', () => {
    expect(
      resolvePythonExecutable({
        extensionPath: INSTALL_DIR,
        exists: existsIn([]),
        platform: 'linux',
      })
    ).toBe('python3');
  });
});

describe('workspace resolution', () => {
  it('reports no workspace when nothing is open', () => {
    expect(resolveWorkspaceState(undefined)).toEqual({ kind: 'none', folders: [] });
    expect(resolveWorkspaceState([])).toEqual({ kind: 'none', folders: [] });
  });

  it('reports a single folder', () => {
    expect(resolveWorkspaceState([folder(PROJECT_A)])).toEqual({
      kind: 'single',
      folders: [PROJECT_A],
    });
  });

  it('reports a multi-root workspace', () => {
    expect(resolveWorkspaceState([folder(PROJECT_A), folder(PROJECT_B)]).kind).toBe('multi');
    expect(isMultiRoot([folder(PROJECT_A), folder(PROJECT_B)])).toBe(true);
    expect(isMultiRoot([folder(PROJECT_A)])).toBe(false);
  });

  it('switches from Project A to Project B without stale state', () => {
    expect(listWorkspaceRoots([folder(PROJECT_A)])).toEqual([PROJECT_A]);
    // Simulate closing Project A and opening Project B.
    expect(listWorkspaceRoots([folder(PROJECT_B)])).toEqual([PROJECT_B]);
    expect(listWorkspaceRoots(undefined)).toEqual([]);
  });

  it('ignores folders with an empty path', () => {
    expect(listWorkspaceRoots([folder('')])).toEqual([]);
  });
});

describe('allowed roots publication', () => {
  const globalStorage = path.join(
    'C:', 'Users', 'dev', 'AppData', 'Roaming', 'Code', 'User',
    'globalStorage', 'rayansharma.llm-training-agent'
  );

  it('writes the open project roots to global storage', () => {
    const writeFile = vi.fn();
    publishAllowedRoots({
      globalStoragePath: globalStorage,
      workspaceFolders: [folder(PROJECT_A)],
      writeFile,
      delimiter: ';',
    });
    expect(writeFile).toHaveBeenCalledWith(
      path.join(globalStorage, ALLOWED_ROOTS_FILE),
      PROJECT_A
    );
  });

  it('rewrites the file when switching from Project A to Project B', () => {
    const writeFile = vi.fn();
    publishAllowedRoots({
      globalStoragePath: globalStorage,
      workspaceFolders: [folder(PROJECT_A)],
      writeFile,
      delimiter: ';',
    });
    publishAllowedRoots({
      globalStoragePath: globalStorage,
      workspaceFolders: [folder(PROJECT_B)],
      writeFile,
      delimiter: ';',
    });
    expect(writeFile).toHaveBeenLastCalledWith(
      path.join(globalStorage, ALLOWED_ROOTS_FILE),
      PROJECT_B
    );
  });

  it('writes an empty list when no folder is open', () => {
    const writeFile = vi.fn();
    publishAllowedRoots({
      globalStoragePath: globalStorage,
      workspaceFolders: undefined,
      writeFile,
      delimiter: ';',
    });
    expect(writeFile).toHaveBeenCalledWith(
      path.join(globalStorage, ALLOWED_ROOTS_FILE),
      ''
    );
  });

  it('joins multiple roots with the platform delimiter', () => {
    const writeFile = vi.fn();
    publishAllowedRoots({
      globalStoragePath: globalStorage,
      workspaceFolders: [folder(PROJECT_A), folder(PROJECT_B)],
      writeFile,
      delimiter: ';',
    });
    expect(writeFile).toHaveBeenCalledWith(
      path.join(globalStorage, ALLOWED_ROOTS_FILE),
      `${PROJECT_A};${PROJECT_B}`
    );
  });

  it('writes into global storage, never into a project', () => {
    const writeFile = vi.fn();
    publishAllowedRoots({
      globalStoragePath: globalStorage,
      workspaceFolders: [folder(PROJECT_A)],
      writeFile,
      delimiter: ';',
    });
    const writtenPath = writeFile.mock.calls[0][0] as string;
    expect(writtenPath.startsWith(globalStorage)).toBe(true);
    expect(writtenPath.startsWith(PROJECT_A)).toBe(false);
  });
});

