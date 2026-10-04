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

import {
  parseRequirements,
  buildManualCommand,
  checkDependencies,
} from '../../src/services/dependencyManager';
import { execFile } from 'child_process';
import * as path from 'path';

const mockExecFile = execFile as unknown as ReturnType<typeof vi.fn>;

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