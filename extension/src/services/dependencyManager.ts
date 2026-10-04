/**
 * DependencyManager — checks whether the Python backend dependencies listed in
 * `backend/requirements.txt` are available in the resolved Python environment.
 *
 * This service ONLY *checks* the environment. It never installs, downloads, or
 * modifies anything. Any installation is performed by the caller only after the
 * user has explicitly granted permission.
 */
import * as path from 'path';
import * as fs from 'fs';
import { execFile, spawn, ChildProcess } from 'child_process';
import * as vscode from 'vscode';

export interface DependencyCheckResult {
  /** True when every requirement from requirements.txt imports successfully. */
  available: boolean;
  /** Full list of requirements that could not be imported. */
  missing: string[];
  /** The path to requirements.txt that was checked. */
  requirementsTxtPath: string | null;
  /** Human-readable summary used in messages to the user. */
  message: string;
  /** Manual installation command the user can run (e.g. `pip install -r ...`). */
  manualCommand: string;
}

/**
 * Mapping from pip requirement names to the Python import spec used to verify
 * availability. Most package names map 1:1 after replacing `-` with `_`, but a
 * few packages (extras, dotted modules, or renamed modules) need explicit
 * handling to stay deterministic.
 */
const IMPORT_OVERRIDES: Record<string, string> = {
  'uvicorn[standard]': 'uvicorn',
  'uvicorn': 'uvicorn',
  'python-dotenv': 'dotenv',
  'google-generativeai': 'google.generativeai',
  'pydantic-settings': 'pydantic_settings',
  'pydantic': 'pydantic',
  'pytest-asyncio': 'pytest_asyncio',
  'tomli': 'tomli',
};

/**
 * Strip extras (`[standard]`) and environment markers (`; python_version < ...`)
 * from a requirement line and return the base package name.
 */
function basePackageName(requirement: string): string {
  let name = requirement.trim();
  // Strip extras, e.g. uvicorn[standard]
  const bracket = name.indexOf('[');
  if (bracket !== -1) {
    name = name.slice(0, bracket);
  }
  // Strip environment markers, e.g. ; python_version < "3.11"
  const marker = name.indexOf(';');
  if (marker !== -1) {
    name = name.slice(0, marker);
  }
  // Strip version specifiers, e.g. ==0.104.1 or >=2.5.0
  const versionMatch = name.match(/^([A-Za-z0-9_.-]+)/);
  return versionMatch ? versionMatch[1].trim() : name.trim();
}

/**
 * Convert a base pip package name into the import spec used by the availability
 * check. Uses the override table first, then falls back to replacing `-` with `_`.
 */
function toImportSpec(baseName: string): string {
  if (IMPORT_OVERRIDES[baseName]) {
    return IMPORT_OVERRIDES[baseName];
  }
  return baseName.replace(/-/g, '_');
}

/**
 * Parse a requirements.txt file into a list of { package, importSpec } entries,
 * ignoring blank lines and comments.
 */
export function parseRequirements(content: string): Array<{ package: string; importSpec: string }> {
  const requirements: Array<{ package: string; importSpec: string }> = [];
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) {
      continue;
    }
    const base = basePackageName(line);
    const importSpec = toImportSpec(base);
    requirements.push({
      package: base,
      importSpec,
    });
  }
  return requirements;
}

/**
 * Build a manual installation command string from the Python executable and the
 * path to requirements.txt (quoted safely for the user to copy-paste).
 */
export function buildManualCommand(pythonExe: string, requirementsTxtPath: string): string {
  const pypath = `"${requirementsTxtPath.replace(/"/g, '\\"')}"`;
  const pyexe = pythonExe.includes('\\') || pythonExe.includes('/') ? `"${pythonExe}"` : pythonExe;
  return `${pyexe} -m pip install -r ${pypath}`;
}

/**
 * Execute a short Python snippet with the resolved interpreter and return true
 * if it exits successfully (exit code 0).
 */
function runPythonImportCheck(pythonExe: string, cwd: string, snippet: string): Promise<boolean> {
  return new Promise((resolve) => {
    execFile(
      pythonExe,
      ['-c', snippet],
      { cwd, timeout: 60000 },
      (error) => {
        resolve(!error);
      }
    );
  });
}

/**
 * Install the dependencies listed in `requirementsTxtPath` using the resolved
 * Python interpreter's pip. This MODIFIES the environment and MUST ONLY be
 * called after the user has explicitly granted permission.
 *
 * Streams output to an output channel so the user can see progress.
 */
export function installDependencies(options: {
  pythonExe: string;
  requirementsTxtPath: string;
  backendDir: string;
  outputChannel: vscode.OutputChannel;
  onDone?: (success: boolean, error?: string) => void;
}): ChildProcess {
  const { pythonExe, requirementsTxtPath, backendDir, outputChannel, onDone } = options;

  outputChannel.appendLine(`Installing dependencies from ${requirementsTxtPath} ...`);
  outputChannel.show(true);

  const args = ['-m', 'pip', 'install', '-r', requirementsTxtPath];
  const child = spawn(pythonExe, args, {
    cwd: backendDir,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  const stripAnsiLocal = (text: string): string =>
    // eslint-disable-next-line no-control-regex
    text.replace(/\x1b\[[0-9;]*m/g, '');

  child.stdout?.on('data', (data: Buffer) => {
    const out = stripAnsiLocal(data.toString()).trimEnd();
    if (out) {
      outputChannel.appendLine(`[pip] ${out}`);
    }
  });

  child.stderr?.on('data', (data: Buffer) => {
    const out = stripAnsiLocal(data.toString()).trimEnd();
    if (out) {
      outputChannel.appendLine(`[pip:stderr] ${out}`);
    }
  });

  child.on('error', (err) => {
    const msg = err.message || String(err);
    outputChannel.appendLine(`Failed to run pip: ${msg}`);
    onDone?.(false, msg);
  });

  child.on('exit', (code) => {
    if (code === 0) {
      outputChannel.appendLine('Dependency installation completed successfully.');
      onDone?.(true);
    } else {
      const msg = `pip exited with code ${code}`;
      outputChannel.appendLine(msg);
      onDone?.(false, msg);
    }
  });

  return child;
}

/**
 * Check that every dependency in `requirementsTxtPath` can be imported by the
 * resolved Python interpreter. This is a read-only check — it never installs or
 * modifies the environment.
 */
export async function checkDependencies(
  options: {
    pythonExe: string;
    backendMainPath: string;
  }
): Promise<DependencyCheckResult> {
  const { pythonExe, backendMainPath } = options;
  const backendDir = path.dirname(backendMainPath);
  const requirementsTxtPath = path.join(backendDir, 'requirements.txt');

  const missing: string[] = [];
  let requirements: Array<{ package: string; importSpec: string }> = [];

  if (!fs.existsSync(requirementsTxtPath)) {
    return {
      available: false,
      missing: [],
      requirementsTxtPath: null,
      message: `Could not find requirements file at ${requirementsTxtPath}.`,
      manualCommand: '',
    };
  }

  try {
    const content = fs.readFileSync(requirementsTxtPath, 'utf8');
    requirements = parseRequirements(content);
  } catch (err) {
    return {
      available: false,
      missing: [],
      requirementsTxtPath,
      message: `Could not read requirements file: ${err instanceof Error ? err.message : String(err)}`,
      manualCommand: buildManualCommand(pythonExe, requirementsTxtPath),
    };
  }

  if (requirements.length === 0) {
    return {
      available: true,
      missing: [],
      requirementsTxtPath,
      message: 'requirements.txt is empty; nothing to check.',
      manualCommand: buildManualCommand(pythonExe, requirementsTxtPath),
    };
  }

  // Build a single import snippet so we only spawn one subprocess.
  const imports = requirements.map((r) => r.importSpec).join(', ');
  const snippet = `import ${imports}`;

  const ok = await runPythonImportCheck(pythonExe, backendDir, snippet);

  if (ok) {
    return {
      available: true,
      missing: [],
      requirementsTxtPath,
      message: `All ${requirements.length} backend dependencies are available.`,
      manualCommand: buildManualCommand(pythonExe, requirementsTxtPath),
    };
  }

  // Fall back to checking each requirement individually to report exactly which
  // ones are missing.
  for (const req of requirements) {
    const importOk = await runPythonImportCheck(pythonExe, backendDir, `import ${req.importSpec}`);
    if (!importOk) {
      missing.push(req.importSpec);
    }
  }

  if (missing.length === 0) {
    // Importing them together failed for some other reason; report the whole set.
    missing.push(...requirements.map((r) => r.importSpec));
  }

  const manualCommand = buildManualCommand(pythonExe, requirementsTxtPath);
  const message = missing.length > 0
    ? `Missing backend dependencies: ${missing.join(', ')}`
    : `Backend dependencies are not all importable (requirements: ${requirements.length}).`;

  return {
    available: missing.length === 0,
    missing,
    requirementsTxtPath,
    message,
    manualCommand,
  };
}