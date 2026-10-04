/**
 * BackendManager — manages the Python backend subprocess lifecycle.
 *
 * Responsibilities:
 *   - Start the bundled FastAPI backend on demand (lazily, not at activation).
 *   - Poll the /health endpoint until the backend is ready.
 *   - Provide the base URL for the ApiClient.
 *   - Shut the process down on extension deactivation.
 *
 * Path policy (see `extensionPaths.ts`):
 *   The backend entry point and the Python interpreter are resolved from the
 *   VS Code *extension installation directory* (`context.extensionUri`) only.
 *   The open workspace is never used to locate extension resources, so the same
 *   installed extension behaves identically for Project A and Project B.
 *
 * Storage policy:
 *   The backend database lives in the extension's global storage
 *   (`context.globalStorageUri`) — never inside the extension install folder
 *   (which VS Code may replace on update) and never inside a user project.
 */
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { spawn, ChildProcess, execFileSync } from 'child_process';
import {
  resolveBackendEntry,
  resolvePythonExecutable,
  listWorkspaceRoots,
  publishAllowedRoots,
  ALLOWED_ROOTS_FILE,
} from './extensionPaths';

export interface BackendStatus {
  status: 'starting' | 'running' | 'stopped' | 'failed';
  url?: string;
  error?: string;
}

/** Default backend origin; overridable via the user-level `backendUrl` setting. */
const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8000';

/** Health-check endpoint exposed by the FastAPI app. */
const HEALTH_PATH = '/api/v1/health';

/** How long to wait for the backend to answer /health before giving up. */
const STARTUP_TIMEOUT_MS = 30_000;

/** Interval between health probes while the backend starts. */
const HEALTH_POLL_INTERVAL_MS = 1_000;

export class BackendManager {
  private readonly context: vscode.ExtensionContext;
  private backendProcess: ChildProcess | null = null;
  private backendUrl: string = DEFAULT_BACKEND_URL;
  private healthCheckTimer: NodeJS.Timeout | null = null;
  private resolveReady: (() => void) | null = null;
  private readyPromise: Promise<void> | null = null;
  private startingPromise: Promise<boolean> | null = null;
  private shutdownRequested = false;
  private readonly outputChannel: vscode.OutputChannel;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.outputChannel = vscode.window.createOutputChannel('LLM Training Agent: Backend');
  }

  /**
   * Log a message to the backend output channel.
   */
  private log(message: string): void {
    this.outputChannel.appendLine(`[${new Date().toISOString()}] ${message}`);
  }

  // ------------------------------------------------------------------
  // Extension-owned resource resolution
  // ------------------------------------------------------------------

  /**
   * Absolute path of the bundled `backend/main.py`, or null when the
   * installation is incomplete.
   */
  getBackendPath(): string | null {
    return resolveBackendEntry(this.context.extensionUri.fsPath, fs.existsSync);
  }

  /** Python interpreter used to run the backend. */
  getPythonExecutable(): string {
    const configured = vscode.workspace
      .getConfiguration('llmTrainingAgent')
      .get<string>('pythonPath', '');
    return resolvePythonExecutable({
      extensionPath: this.context.extensionUri.fsPath,
      configuredPath: configured,
      exists: fs.existsSync,
      platform: process.platform,
    });
  }

  /** Origin of the backend, honouring the user-level `backendUrl` setting. */
  getBaseUrl(): string {
    const configured = vscode.workspace
      .getConfiguration('llmTrainingAgent')
      .get<string>('backendUrl', DEFAULT_BACKEND_URL)
      .trim();
    return configured || DEFAULT_BACKEND_URL;
  }

  /** Path of the file through which the backend learns the open project roots. */
  getAllowedRootsFile(): string {
    return path.join(this.context.globalStorageUri.fsPath, ALLOWED_ROOTS_FILE);
  }

  /**
   * Re-publish the set of authorized project roots.
   *
   * Called whenever the open folders change so a running backend immediately
   * follows a Project A -> Project B switch. This is what keeps the two
   * projects' data isolated from each other.
   */
  refreshAllowedRoots(): void {
    try {
      fs.mkdirSync(this.context.globalStorageUri.fsPath, { recursive: true });
      publishAllowedRoots({
        globalStoragePath: this.context.globalStorageUri.fsPath,
        workspaceFolders: vscode.workspace.workspaceFolders,
        writeFile: (filePath, contents) => fs.writeFileSync(filePath, contents, 'utf8'),
        delimiter: path.delimiter,
      });
      this.log(
        `Authorized project roots refreshed: ${
          listWorkspaceRoots(vscode.workspace.workspaceFolders).join(', ') || '(none)'
        }`
      );
    } catch (error) {
      this.log(
        `Could not publish authorized project roots: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  /**
   * Ensure the backend is running. If it is not, start it.
   * Returns a promise that resolves once the backend health check passes.
   *
   * Safe to call concurrently: a single startup attempt is shared.
   */
  async ensureRunning(): Promise<boolean> {
    this.backendUrl = this.getBaseUrl();

    // Already healthy — either our own process or a manually started backend.
    if (await this.checkHealth()) {
      return true;
    }

    if (this.startingPromise) {
      return this.startingPromise;
    }

    this.startingPromise = this.start();
    try {
      return await this.startingPromise;
    } finally {
      this.startingPromise = null;
    }
  }

  /**
   * Start the backend subprocess and wait for it to become healthy.
   */
  async start(): Promise<boolean> {
    this.shutdownRequested = false;

    // Extension-owned resources come from the extension install directory.
    const backendEntry = this.getBackendPath();
    if (!backendEntry) {
      this.log(
        `Bundled backend not found under ${this.context.extensionUri.fsPath}.`
      );
      vscode.window.showWarningMessage(
        'LLM Training Agent: the bundled Python backend is missing from the installed ' +
          'extension. Please reinstall it via Extensions → ⋯ → "Install from VSIX…".'
      );
      return false;
    }

    const backendDir = path.dirname(backendEntry);
    const pythonExe = this.getPythonExecutable();

    this.log(`Extension install dir : ${this.context.extensionUri.fsPath}`);
    this.log(`Backend entry point   : ${backendEntry}`);
    this.log(`Python interpreter    : ${pythonExe}`);

    if (!this.backendDependenciesAvailable(pythonExe, backendDir)) {
      this.cleanupProcess();
      return false;
    }

    // The workspace is read HERE, at start time — it is the current project
    // context, never a source of extension resources.
    const workspaceRoots = listWorkspaceRoots(vscode.workspace.workspaceFolders);
    this.log(`Authorized project roots: ${workspaceRoots.join(', ') || '(none)'}`);

    // Publish the roots to a file so a later Project A -> Project B switch is
    // honoured by this same process, without a reload or reinstall.
    this.refreshAllowedRoots();

    this.readyPromise = new Promise<void>((resolve) => {
      this.resolveReady = resolve;
    });

    try {
      this.log(`Starting backend: ${pythonExe} -m uvicorn main:app`);
      this.backendProcess = spawn(
        pythonExe,
        [
          '-m',
          'uvicorn',
          'main:app',
          '--host',
          '127.0.0.1',
          '--port',
          new URL(this.backendUrl).port || '8000',
          '--no-server-header',
        ],
        {
          // The backend runs *from the extension*, never from the user's project.
          cwd: backendDir,
          stdio: ['ignore', 'pipe', 'pipe'],
          env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            PYTHONPATH: backendDir,
            // Keep the database in VS Code's global storage for this extension,
            // never in the install folder (replaced on update) or a project.
            DATABASE_URL: this.databaseUrl(),
            LLM_TRAINING_AGENT_ALLOWED_ROOTS: workspaceRoots.join(path.delimiter),
            // Re-read on change so switching projects takes effect immediately.
            LLM_TRAINING_AGENT_ALLOWED_ROOTS_FILE: this.getAllowedRootsFile(),
          },
        }
      );

      this.backendProcess.stdout?.on('data', (data: Buffer) => {
        this.log(`[stdout] ${stripAnsi(data.toString()).trimEnd()}`);
      });

      this.backendProcess.stderr?.on('data', (data: Buffer) => {
        this.log(`[stderr] ${stripAnsi(data.toString()).trimEnd()}`);
      });

      this.backendProcess.on('exit', (code) => {
        this.log(`Backend process exited with code ${code}`);
        this.backendProcess = null;
      });

      this.startHealthPolling();

      // Wait for the health check to pass, bounded by STARTUP_TIMEOUT_MS so a
      // hung backend can never block a command forever.
      const timedOut = await Promise.race([
        this.readyPromise.then(() => false),
        delay(STARTUP_TIMEOUT_MS).then(() => true),
      ]);

      this.stopHealthPolling();

      if (timedOut) {
        this.log(`Backend did not become healthy within ${STARTUP_TIMEOUT_MS}ms.`);
      }

      if (await this.checkHealth()) {
        this.log('Backend is ready.');
        return true;
      }

      this.log('Backend started but health check failed.');
      vscode.window.showWarningMessage(
        'LLM Training Agent: the backend started but its health check failed. ' +
          'See the "LLM Training Agent: Backend" output channel for details.'
      );
      return false;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.log(`Failed to start backend: ${message}`);
      vscode.window.showWarningMessage(
        `LLM Training Agent: failed to start the backend: ${message}`
      );
      this.cleanupProcess();
      return false;
    }
  }

  /**
   * SQLite URL pointing at this extension's VS Code global storage.
   *
   * Global storage is the VS Code supported home for persistent,
   * non-project-specific data: it survives extension updates and is never
   * written into a user project.
   */
  private databaseUrl(): string {
    const dbPath = path.join(
      this.context.globalStorageUri.fsPath,
      'llm_training_agent.db'
    );
    return `sqlite+aiosqlite:///${dbPath.replace(/\\/g, '/')}`;
  }

  /**
   * Read-only dependency probe so a backend with missing dependencies fails
   * fast with an actionable message instead of a silent crash loop.
   * This never installs or modifies anything.
   */
  private backendDependenciesAvailable(pythonExe: string, backendDir: string): boolean {
    try {
      execFileSync(
        pythonExe,
        ['-c', 'import fastapi, uvicorn, pydantic, pydantic_settings'],
        { cwd: backendDir, stdio: 'ignore', timeout: 20_000 }
      );
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.log(`Backend dependency check failed: ${message}`);
      vscode.window.showWarningMessage(
        'LLM Training Agent: the Python backend dependencies are not available to ' +
          `"${pythonExe}". Install them with: "${pythonExe}" -m pip install -r ` +
          `"${path.join(backendDir, 'requirements.txt')}", or point ` +
          '"llmTrainingAgent.pythonPath" at a suitable interpreter.'
      );
      return false;
    }
  }

  /**
   * Check the backend health endpoint.
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.backendUrl}${HEALTH_PATH}`, {
        signal: AbortSignal.timeout(5_000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Poll the health endpoint until the backend answers, then stop.
   */
  private startHealthPolling(): void {
    this.stopHealthPolling();
    this.healthCheckTimer = setInterval(async () => {
      if (!this.backendProcess || this.backendProcess.exitCode !== null) {
        return;
      }
      if (await this.checkHealth()) {
        this.log('Health check passed.');
        this.resolveReady?.();
        this.resolveReady = null;
        this.stopHealthPolling();
      }
    }, HEALTH_POLL_INTERVAL_MS);
  }

  private stopHealthPolling(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = null;
    }
  }

  /**
   * Clean up the backend process.
   */
  private cleanupProcess(): void {
    this.stopHealthPolling();
    if (this.backendProcess) {
      this.backendProcess.kill();
      this.backendProcess = null;
    }
  }

  /**
   * Shut down the backend gracefully.
   */
  async shutdown(): Promise<void> {
    this.shutdownRequested = true;
    this.cleanupProcess();
    this.outputChannel.dispose();
  }

  /**
   * Get current backend status.
   */
  getStatus(): BackendStatus {
    if (!this.backendProcess) {
      return { status: 'stopped', url: this.backendUrl };
    }
    if (this.backendProcess.exitCode === null) {
      return { status: 'running', url: this.backendUrl };
    }
    return { status: 'failed', url: this.backendUrl, error: 'Process exited' };
  }

  /**
   * Get the backend URL, starting the backend if needed.
   */
  async getBackendUrl(): Promise<string> {
    if (!(await this.ensureRunning())) {
      throw new Error('Backend failed to start');
    }
    return this.backendUrl;
  }
}

/**
 * Resolve after `ms` milliseconds.
 */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Utility to strip ANSI escape sequences from terminal output.
 */
function stripAnsi(text: string): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/\x1b\[[0-9;]*m/g, '');
}