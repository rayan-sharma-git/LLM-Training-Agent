import { describe, it, expect, vi, beforeEach } from 'vitest';

// Hoist the mock functions so the vi.mock factories can reference them.
// Vitest hoists vi.mock calls to the top of the file, so we must create the
// mock functions with vi.hoisted to avoid `Cannot access before initialization`.
const { mockExistsSync, mockReadFileSync } = vi.hoisted(() => ({
  mockExistsSync: vi.fn((_p: string) => false) as (p: string) => boolean,
  mockReadFileSync: vi.fn((_p: string, _enc: string) => '') as (p: string, enc: string) => string,
}));

// Mock child_process.execFile so we never spawn real Python during tests.
vi.mock('child_process', () => ({
  execFile: vi.fn(),
  spawn: vi.fn(),
}));

// Mock the fs module so we control existsSync/readFileSync deterministically.
vi.mock('fs', () => ({
  existsSync: (p: string) => mockExistsSync(p),
  readFileSync: (p: string, encoding: string) => mockReadFileSync(p, encoding),
}));

// dependencyManager imports vscode only for OutputChannel/env types at the
// value level (openExternal on the no-interpreter path). Stub it so the pure
// Node suite never needs the Electron host.
vi.mock('vscode', () => ({
  env: { openExternal: vi.fn() },
  Uri: { parse: (value: string) => ({ value }) },
  ProgressLocation: { Notification: 15 },
  window: {
    showInformationMessage: vi.fn(),
    withProgress: vi.fn(),
  },
}));

import {
  parseRequirements,
  buildManualCommand,
  checkDependencies,
  probePythonEnvironment,
  ensureRuntimeDependencies,
} from '../../src/services/dependencyManager';
import { execFile, spawn } from 'child_process';
import { EventEmitter } from 'events';
import * as path from 'path';

const mockExecFile = execFile as unknown as ReturnType<typeof vi.fn>;
const mockSpawn = spawn as unknown as ReturnType<typeof vi.fn>;

describe('parseRequirements', () => {
  it('parses simple requirements and strips versions', () => {
    const reqs = parseRequirements('fastapi==0.104.1\nuvicorn[standard]==0.24.0\n');
    expect(reqs.map((r) => r.package)).toEqual(['fastapi', 'uvicorn']);
  });

  it('ignores comments and blank lines', () => {
    const reqs = parseRequirements('# comment\n\npydantic>=2.5.0\n');
    expect(reqs.map((r) => r.package)).toEqual(['pydantic']);
  });

  it('maps pip names to import specs', () => {
    const reqs = parseRequirements('python-dotenv==1.0.0\ngoogle-generativeai==0.7.2\npydantic-settings==2.1.0\n');
    const specs = Object.fromEntries(reqs.map((r) => [r.package, r.importSpec]));
    expect(specs['python-dotenv']).toBe('dotenv');
    expect(specs['google-generativeai']).toBe('google.generativeai');
    expect(specs['pydantic-settings']).toBe('pydantic_settings');
  });

  it('strips environment markers', () => {
    const reqs = parseRequirements('tomli>=1.2.3; python_version < "3.11"\n');
    expect(reqs.map((r) => r.package)).toEqual(['tomli']);
  });
});

describe('buildManualCommand', () => {
  it('quotes the requirements path and python executable', () => {
    const cmd = buildManualCommand('C:\\Python310\\python.exe', 'C:\\proj\\backend\\requirements.txt');
    expect(cmd).toBe('"C:\\Python310\\python.exe" -m pip install -r "C:\\proj\\backend\\requirements.txt"');
  });

  it('does not quote a bare python command', () => {
    const cmd = buildManualCommand('python', 'C:\\proj\\requirements.txt');
    expect(cmd).toBe('python -m pip install -r "C:\\proj\\requirements.txt"');
  });
});

describe('checkDependencies', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns available=false when requirements.txt is missing', async () => {
    mockExistsSync.mockReturnValue(false);
    const result = await checkDependencies({
      pythonExe: 'python',
      backendMainPath: 'C:\\backend\\main.py',
    });
    expect(result.available).toBe(false);
    expect(result.message).toContain('Could not find requirements file');
    expect(mockReadFileSync).not.toHaveBeenCalled();
  });

  it('returns available=true when all imports succeed', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue('fastapi==0.104.1\nuvicorn==0.24.0\n');
    mockExecFile.mockImplementation((_py: string, _args: string[], _opts: unknown, cb: (err: null) => void) => {
      cb(null);
    });
    const result = await checkDependencies({
      pythonExe: 'python',
      backendMainPath: path.join('C:', 'backend', 'main.py'),
    });
    expect(result.available).toBe(true);
    expect(result.message).toContain('available');
  });

  it('reports missing packages when imports fail', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue('fastapi==0.104.1\nsqlalchemy==2.0.23\n');
    // First call (combined) fails, then per-package checks.
    mockExecFile.mockImplementation(
      (_py: string, args: string[], _opts: unknown, cb: (err: Error | null) => void) => {
        const snippet = args[1] as string;
        // Simulate sqlalchemy missing: fail any snippet that imports it
        // (both the combined check and the per-package check).
        const err = snippet.includes('sqlalchemy') ? new Error('not found') : null;
        cb(err);
      }
    );
    const result = await checkDependencies({
      pythonExe: 'python',
      backendMainPath: path.join('C:', 'backend', 'main.py'),
    });
    expect(result.available).toBe(false);
    expect(result.missing).toContain('sqlalchemy');
  });
});

describe('checkDependencies with the bundled requirements checker', () => {
  const backendMainPath = path.join('C:', 'backend', 'main.py');

  beforeEach(() => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue('fastapi==0.104.1\ntomli>=1.2.3; python_version < "3.11"\n');
  });

  const reply = (report: Record<string, unknown>) =>
    mockExecFile.mockImplementation(
      (_py: string, _args: string[], _opts: unknown, cb: (err: null, stdout: string) => void) => {
        cb(null, JSON.stringify(report));
      }
    );

  it('runs check_requirements.py against the bundled requirements file', async () => {
    reply({ python: '3.12.1', ok: true, checked: 1, missing: [], outdated: [], skipped: ['tomli'] });
    const result = await checkDependencies({ pythonExe: 'python3', backendMainPath });
    expect(result.available).toBe(true);
    const args = mockExecFile.mock.calls[0][1] as string[];
    expect(args[0]).toContain('check_requirements.py');
    expect(args[1]).toContain('requirements.txt');
    // A marker-excluded package is not demanded, so no install loop.
    expect(result.missing).toEqual([]);
  });

  it('reports missing and too-old packages with their versions', async () => {
    reply({
      python: '3.13.1',
      ok: false,
      checked: 3,
      missing: ['fastapi'],
      outdated: [{ name: 'sqlalchemy', installed: '2.0.23', required: '>=2.0.30' }],
      skipped: [],
    });
    const result = await checkDependencies({ pythonExe: 'python3', backendMainPath });
    expect(result.available).toBe(false);
    expect(result.missing).toEqual(['fastapi', 'sqlalchemy (installed 2.0.23, needs >=2.0.30)']);
    expect(result.message).toContain('outdated');
  });

  it('falls back to the import check when the checker cannot run', async () => {
    mockExecFile.mockImplementation(
      (_py: string, args: string[], _opts: unknown, cb: (err: Error | null, stdout?: string) => void) => {
        cb(String(args[0]).includes('check_requirements') ? new Error('boom') : null);
      }
    );
    const result = await checkDependencies({ pythonExe: 'python3', backendMainPath });
    expect(result.available).toBe(true);
    expect(mockExecFile.mock.calls.length).toBeGreaterThan(1);
  });
});

describe('probePythonEnvironment', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('reports no-interpreter when Python cannot be started', async () => {
    mockExecFile.mockImplementation((_py: string, _args: string[], _opts: unknown, cb: (err: Error | null) => void) => {
      cb(new Error('ENOENT'));
    });
    const probe = await probePythonEnvironment({
      pythonExe: 'missing-python',
      backendMainPath: path.join('C:', 'backend', 'main.py'),
    });
    expect(probe.status).toBe('no-interpreter');
  });

  it('rejects interpreters older than 3.10', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue('fastapi==0.104.1\n');
    mockExecFile.mockImplementation((_py: string, args: string[], _opts: unknown, cb: (e: Error | null, out?: string) => void) => {
      const snippet = String(args[args.length - 1]);
      if (snippet.includes('sys.version_info')) {
        cb(null, '3.9.13');
        return;
      }
      cb(null);
    });
    const probe = await probePythonEnvironment({
      pythonExe: 'python',
      backendMainPath: path.join('C:', 'backend', 'main.py'),
    });
    expect(probe.status).toBe('unsupported-version');
  });

  it('reports ready without prompting when everything imports', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue('fastapi==0.104.1\n');
    mockExecFile.mockImplementation((_py: string, args: string[], _opts: unknown, cb: (e: Error | null, out?: string) => void) => {
      const snippet = String(args[args.length - 1]);
      cb(null, snippet.includes('sys.version_info') ? '3.12.1' : '');
    });
    const probe = await probePythonEnvironment({
      pythonExe: 'python',
      backendMainPath: path.join('C:', 'backend', 'main.py'),
    });
    expect(probe.status).toBe('ready');
  });
});

describe('ensureRuntimeDependencies', () => {
  const channel = () => ({ appendLine: vi.fn(), show: vi.fn() }) as unknown as import('vscode').OutputChannel;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns true immediately when the environment is ready (no prompt, no install)', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue('fastapi==0.104.1\n');
    mockExecFile.mockImplementation((_py: string, args: string[], _opts: unknown, cb: (e: Error | null, out?: string) => void) => {
      const snippet = String(args[args.length - 1]);
      cb(null, snippet.includes('sys.version_info') ? '3.12.0' : '');
    });
    const showPrompt = vi.fn();
    const ok = await ensureRuntimeDependencies({
      pythonExe: 'python',
      backendMainPath: path.join('C:', 'backend', 'main.py'),
      outputChannel: channel(),
      callbacks: { showPrompt, withProgress: async (_t, fn) => fn() },
    });
    expect(ok).toBe(true);
    expect(showPrompt).not.toHaveBeenCalled();
  });

  it('returns false without installing when the user cancels', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue('fastapi==0.104.1\nsqlalchemy==2.0.23\n');
    mockExecFile.mockImplementation((_py: string, args: string[], _opts: unknown, cb: (e: Error | null, out?: string) => void) => {
      const snippet = String(args[args.length - 1]);
      if (snippet.includes('sys.version_info')) {
        cb(null, '3.12.0');
        return;
      }
      cb(snippet.includes('sqlalchemy') ? new Error('missing') : null);
    });
    const showPrompt = vi.fn().mockResolvedValue('Cancel');
    const ok = await ensureRuntimeDependencies({
      pythonExe: 'python',
      backendMainPath: path.join('C:', 'backend', 'main.py'),
      outputChannel: channel(),
      callbacks: { showPrompt, withProgress: async (_t, fn) => fn() },
    });
    expect(ok).toBe(false);
    expect(showPrompt).toHaveBeenCalledOnce();
  });

  describe('automatic install (the default)', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    // pip fake: finishes with `code`, optionally printing `stderr` first.
    const fakePip = (code: number, stderr = '') => {
      mockSpawn.mockImplementation(() => {
        const child = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter };
        child.stdout = new EventEmitter();
        child.stderr = new EventEmitter();
        setImmediate(() => {
          if (stderr) child.stderr.emit('data', Buffer.from(stderr));
          child.emit('exit', code);
        });
        return child;
      });
    };
    // Python answers "sqlalchemy missing" until pip has run, then "all present".
    const pythonWithPackages = (installed: () => boolean) => {
      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue('fastapi==0.104.1\nsqlalchemy==2.0.23\n');
      mockExecFile.mockImplementation((_py: string, args: string[], _opts: unknown, cb: (e: Error | null, out?: string) => void) => {
        const snippet = String(args[args.length - 1]);
        if (snippet.includes('sys.version_info')) {
          cb(null, '3.12.0');
          return;
        }
        cb(!installed() && snippet.includes('sqlalchemy') ? new Error('missing') : null);
      });
    };

    it('installs backend/requirements.txt without asking, then re-checks', async () => {
      let installed = false;
      pythonWithPackages(() => installed);
      mockSpawn.mockImplementation(() => {
        installed = true;
        const child = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter };
        child.stdout = new EventEmitter();
        child.stderr = new EventEmitter();
        setImmediate(() => child.emit('exit', 0));
        return child;
      });
      const showPrompt = vi.fn();
      const ok = await ensureRuntimeDependencies({
        pythonExe: 'python',
        backendMainPath: path.join('C:', 'backend', 'main.py'),
        outputChannel: channel(),
        callbacks: { showPrompt, withProgress: async (_t, fn) => fn() },
        autoInstall: true,
      });
      expect(ok).toBe(true);
      expect(showPrompt).not.toHaveBeenCalled();
      const args = mockSpawn.mock.calls[0][1] as string[];
      expect(args.slice(0, 4)).toEqual(['-m', 'pip', 'install', '-r']);
      // Only the bundled runtime list is ever installed, never the developer one.
      expect(args[4]).toBe(path.join('C:', 'backend', 'requirements.txt'));
    });

    it('still asks first when automatic install is switched off', async () => {
      pythonWithPackages(() => false);
      const showPrompt = vi.fn().mockResolvedValue('Cancel');
      const ok = await ensureRuntimeDependencies({
        pythonExe: 'python',
        backendMainPath: path.join('C:', 'backend', 'main.py'),
        outputChannel: channel(),
        callbacks: { showPrompt, withProgress: async (_t, fn) => fn() },
        autoInstall: false,
      });
      expect(ok).toBe(false);
      expect(showPrompt).toHaveBeenCalledOnce();
      expect(mockSpawn).not.toHaveBeenCalled();
    });

    it('explains the virtual-environment fix when pip refuses a system Python', async () => {
      pythonWithPackages(() => false);
      fakePip(1, 'error: externally-managed-environment\n× This environment is externally managed');
      const lines: string[] = [];
      const out = { appendLine: (l: string) => lines.push(l), show: vi.fn() } as unknown as import('vscode').OutputChannel;
      const ok = await ensureRuntimeDependencies({
        pythonExe: 'python3',
        backendMainPath: path.join('C:', 'backend', 'main.py'),
        outputChannel: out,
        callbacks: { showPrompt: vi.fn(), withProgress: async (_t, fn) => fn() },
        autoInstall: true,
      });
      expect(ok).toBe(false);
      expect(lines.join('\n')).toContain('python -m venv');
      expect(lines.join('\n')).toContain('llmTrainingAgent.pythonPath');
    });
  });
});
