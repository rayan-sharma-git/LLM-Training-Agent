/**
 * DependencyManager — checks whether the Python backend dependencies listed in
 * `backend/requirements.txt` are available in the resolved Python environment,
 * and installs them (only after explicit user consent) when they are missing.
 *
 * The install step streams pip output to the backend channel, reports
 * interrupted installs / missing interpreters / network failures distinctly,
 * and re-checks imports afterwards instead of assuming success.
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
    const prefix =
      path.basename(pythonExe).toLowerCase() === 'py' ||
      path.basename(pythonExe).toLowerCase() === 'py.exe'
        ? ['-3']
        : [];
    execFile(
      pythonExe,
      [...prefix, '-c', snippet],
      { cwd, timeout: 60000 },
      (error) => {
        resolve(!error);
      }
    );
  });
}

export interface DependencyFlowCallbacks {
  showPrompt: (
    message: string,
    options: { modal: boolean },
    ...items: string[]
  ) => Promise<string | undefined> | Thenable<string | undefined>;
  withProgress: (title: string, fn: () => Promise<boolean>) => Promise<boolean> | Thenable<boolean>;
}

/** Minimum Python version the bundled backend supports. */
export const MIN_PYTHON_VERSION: readonly [number, number] = [3, 10];

/**
 * Install the dependencies listed in `requirementsTxtPath` using the resolved
 * Python interpreter's pip. This MODIFIES the environment and MUST ONLY be
 * called after the user has explicitly granted permission (see
 * `ensureRuntimeDependencies` below). Only ever called with the bundled
 * `backend/requirements.txt` — never the root developer requirements.
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

  const pipPrefix =
    path.basename(pythonExe).toLowerCase() === 'py' ||
    path.basename(pythonExe).toLowerCase() === 'py.exe'
      ? ['-3']
      : [];
  const args = [...pipPrefix, '-m', 'pip', 'install', '-r', requirementsTxtPath];
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

  const manualCommand = buildManualCommand(pythonExe, requirementsTxtPath);

  // Preferred: the bundled stdlib-only checker. Unlike "import every package"
  // it honours environment markers (a `tomli; python_version < "3.11"` line
  // must not be demanded on Python 3.12) and it notices packages that are
  // installed but older than required, which still import fine and then fail
  // at runtime.
  const checker = path.join(backendDir, 'check_requirements.py');
  if (fs.existsSync(checker)) {
    const report = await runRequirementsChecker(pythonExe, backendDir, checker, requirementsTxtPath);
    if (report) {
      const problems = [
        ...report.missing,
        ...report.outdated.map((o) => `${o.name} (installed ${o.installed}, needs ${o.required})`),
      ];
      return {
        available: report.ok,
        missing: problems,
        requirementsTxtPath,
        message: report.ok
          ? `All ${report.checked} backend dependencies are available (Python ${report.python}).`
          : `Missing or outdated backend dependencies: ${problems.join(', ')}`,
        manualCommand,
      };
    }
    // The checker itself could not run; fall back to the import check below.
  }

  // Fallback (older bundles without the checker): a single import snippet.
  const imports = requirements.map((r) => r.importSpec).join(', ');
  const snippet = `import ${imports}`;

  const ok = await runPythonImportCheck(pythonExe, backendDir, snippet);

  if (ok) {
    return {
      available: true,
      missing: [],
      requirementsTxtPath,
      message: `All ${requirements.length} backend dependencies are available.`,
      manualCommand,
    };
  }

  // Check each requirement individually to report exactly which are missing.
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

  const message = `Missing backend dependencies: ${missing.join(', ')}`;

  return {
    available: false,
    missing,
    requirementsTxtPath,
    message,
    manualCommand,
  };
}

interface RequirementsReport {
  python: string;
  ok: boolean;
  checked: number;
  missing: string[];
  outdated: Array<{ name: string; installed: string; required: string }>;
}

/** Run backend/check_requirements.py; null when it cannot run or prints junk. */
function runRequirementsChecker(
  pythonExe: string,
  cwd: string,
  checker: string,
  requirementsTxtPath: string
): Promise<RequirementsReport | null> {
  const base = path.basename(pythonExe).toLowerCase();
  const prefix = base === 'py' || base === 'py.exe' ? ['-3'] : [];
  return new Promise((resolve) => {
    execFile(
      pythonExe,
      [...prefix, checker, requirementsTxtPath],
      { cwd, timeout: 60000 },
      (error, stdout) => {
        if (error) {
          resolve(null);
          return;
        }
        try {
          const parsed = JSON.parse(String(stdout)) as RequirementsReport;
          if (typeof parsed.ok !== 'boolean' || !Array.isArray(parsed.missing) || !Array.isArray(parsed.outdated)) {
            resolve(null);
            return;
          }
          resolve(parsed);
        } catch {
          resolve(null);
        }
      }
    );
  });
}

/** Outcome of probing one Python interpreter for backend readiness. */
export type PythonProbeResult =
  | { status: 'ready'; version: string }
  | { status: 'no-interpreter'; detail: string }
  | { status: 'unsupported-version'; version: string }
  | { status: 'missing-deps'; check: DependencyCheckResult };

/**
 * Probe one interpreter: spawnable, version >= 3.10, and able to import
 * every package from the bundled backend/requirements.txt.
 */
export function probePythonEnvironment(options: {
  pythonExe: string;
  backendMainPath: string;
}): Promise<PythonProbeResult> {
  const { pythonExe, backendMainPath } = options;
  const backendDir = path.dirname(backendMainPath);
  const base = path.basename(pythonExe).toLowerCase();
  const prefix = base === 'py' || base === 'py.exe' ? ['-3'] : [];
  return new Promise((resolve) => {
    execFile(
      pythonExe,
      [...prefix, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'],
      { cwd: backendDir, timeout: 20000 },
      (error, stdout) => {
        if (error) {
          resolve({ status: 'no-interpreter', detail: `Could not run "${pythonExe}".` });
          return;
        }
        const version = String(stdout).trim();
        const nums = version.split('.').map((n) => parseInt(n, 10));
        const major = nums[0] ?? 0;
        const minor = nums[1] ?? 0;
        if (major < MIN_PYTHON_VERSION[0] || (major === MIN_PYTHON_VERSION[0] && minor < MIN_PYTHON_VERSION[1])) {
          resolve({ status: 'unsupported-version', version });
          return;
        }
        void checkDependencies({ pythonExe, backendMainPath }).then((check) => {
          if (check.available) {
            resolve({ status: 'ready', version });
          } else {
            resolve({ status: 'missing-deps', check });
          }
        });
      }
    );
  });
}

/** Run pip install for the bundled runtime requirements (consent required). */
function runPipInstall(options: {
  pythonExe: string;
  requirementsTxtPath: string;
  backendDir: string;
  outputChannel: vscode.OutputChannel;
}): Promise<boolean> {
  const { pythonExe, requirementsTxtPath, backendDir, outputChannel } = options;
  return new Promise<boolean>((resolve) => {
    let settled = false;
    const done = (ok: boolean) => {
      if (!settled) {
        settled = true;
        resolve(ok);
      }
    };
    try {
      const child = installDependencies({
        pythonExe,
        requirementsTxtPath,
        backendDir,
        outputChannel,
        onDone: (success, error) => {
          if (!success) {
            outputChannel.appendLine(`Dependency installation failed: ${error ?? 'unknown error'}.`);
          }
          done(success);
        },
      });
      child.on('error', (err) => {
        const msg = err instanceof Error ? err.message : String(err);
        outputChannel.appendLine(`Dependency installation could not start (${msg}).`);
        done(false);
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      outputChannel.appendLine(`Dependency installation could not start (${msg}).`);
      done(false);
    }
  });
}

/**
 * Ensure runtime deps are present, installing them after explicit consent.
 * Only the bundled backend/requirements.txt is ever installed — never the
 * root developer requirements. Re-checks imports after install; handles
 * missing interpreters, old Pythons, pip/network failures, and interrupted
 * installs with distinct actionable messages plus the manual fallback.
 */
export async function ensureRuntimeDependencies(options: {
  pythonExe: string;
  backendMainPath: string;
  outputChannel: vscode.OutputChannel;
  callbacks: DependencyFlowCallbacks;
}): Promise<boolean> {
  const { pythonExe, backendMainPath, outputChannel, callbacks } = options;
  const probe = await probePythonEnvironment({ pythonExe, backendMainPath });
  if (probe.status === 'ready') {
    return true;
  }
  if (probe.status === 'no-interpreter') {
    outputChannel.appendLine(`Backend setup failed: ${probe.detail}`);
    outputChannel.show(true);
    const pick = await callbacks.showPrompt(
      'LLM Training Agent: no compatible Python interpreter was found. Install Python 3.10 or newer, then retry.',
      { modal: false },
      'Open Python Download Page',
      'Cancel'
    );
    if (pick === 'Open Python Download Page') {
      void vscode.env.openExternal(vscode.Uri.parse('https://www.python.org/downloads/'));
    }
    return false;
  }
  if (probe.status === 'unsupported-version') {
    outputChannel.appendLine(`Backend setup failed: Python ${probe.version} is too old (need 3.10+).`);
    outputChannel.show(true);
    await callbacks.showPrompt(
      `LLM Training Agent: Python ${probe.version} is too old — 3.10+ is required.`,
      { modal: false },
      'OK'
    );
    return false;
  }
  const check = probe.check;
  if (!check.requirementsTxtPath) {
    outputChannel.appendLine(`Backend setup failed: ${check.message}`);
    outputChannel.show(true);
    await callbacks.showPrompt(
      `LLM Training Agent: ${check.message} Reinstall the extension.`,
      { modal: false },
      'OK'
    );
    return false;
  }
  const backendDir = path.dirname(backendMainPath);
  const manualCommand = check.manualCommand;
  const missingList = check.missing.length > 0 ? check.missing.join(', ') : 'required packages';
  outputChannel.appendLine(`Backend setup: ${check.message}`);
  const pick = await callbacks.showPrompt(
    `LLM Training Agent: backend needs packages (${missingList}). Install now?`,
    { modal: false },
    'Install Dependencies',
    'Show Manual Command',
    'Cancel'
  );
  if (pick === 'Show Manual Command') {
    outputChannel.appendLine(`Manual install command: ${manualCommand}`);
    outputChannel.show(true);
    return false;
  }
  if (pick !== 'Install Dependencies') {
    return false;
  }
  const installed = await callbacks.withProgress(
    'Installing backend dependencies...',
    () => runPipInstall({ pythonExe, requirementsTxtPath: check.requirementsTxtPath as string, backendDir, outputChannel })
  );
  if (!installed) {
    outputChannel.appendLine(`Install failed. Retry, or run manually: ${manualCommand}`);
    outputChannel.show(true);
    return false;
  }
  const recheck = await checkDependencies({ pythonExe, backendMainPath });
  if (recheck.available) {
    outputChannel.appendLine('Backend setup: all runtime dependencies verified after install.');
    return true;
  }
  outputChannel.appendLine(`Still missing after install. Run manually: ${recheck.manualCommand || manualCommand}`);
  outputChannel.show(true);
  return false;
}
