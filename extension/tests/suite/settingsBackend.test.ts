/**
 * Regression guards for the settings backend-connection failure.
 *
 * Root cause: the Settings webview called the backend API without starting
 * the bundled backend first (every other caller awaits `ensureBackend`),
 * and the dev launch config never synced the bundled backend, so in a fresh
 * window saving settings failed with ECONNREFUSED ("Settings update failed:
 * Could not connect to the backend...").
 *
 * These are source-level guards (same style as settings.test.ts) because the
 * providers import `vscode` and cannot run under plain vitest/Node.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

const EXTENSION_DIR = path.resolve(__dirname, '..', '..');
const read = (relative: string) => fs.readFileSync(path.join(EXTENSION_DIR, relative), 'utf8');

describe('settings backend startup wiring', () => {
  const providerSrc = read(path.join('src', 'views', 'settingsWebviewProvider.ts'));
  const extensionSrc = read(path.join('src', 'extension.ts'));

  it('gives the settings view access to the backend starter', () => {
    expect(providerSrc).toContain('_ensureBackend');
    expect(providerSrc).toContain('_ensureBackendReady');
  });

  it('stores the key and VS Code settings before touching the backend', () => {
    // A slow or broken backend must never lose what the user just entered; the
    // backend gets everything again from SecretStorage whenever it starts.
    const handlerIndex = providerSrc.indexOf('private async _save(');
    expect(handlerIndex).toBeGreaterThan(-1);
    const handler = providerSrc.slice(handlerIndex, handlerIndex + 3500);
    const settingsIndex = handler.indexOf('updateSettings');
    const keyIndex = handler.indexOf('setApiKey');
    const backendIndex = handler.indexOf('_ensureBackendReady');
    const selectIndex = handler.indexOf('selectProvider');
    expect(settingsIndex).toBeGreaterThan(-1);
    expect(keyIndex).toBeGreaterThan(settingsIndex);
    expect(backendIndex).toBeGreaterThan(keyIndex);
    expect(selectIndex).toBeGreaterThan(backendIndex);
    expect(handler).toContain('applied automatically the next time the backend starts');
  });

  it('starts the backend before every other settings API call', () => {
    for (const api of ['testConnection(', 'removeApiKeyBackend(', 'getProviderModels(', 'updateConfig(']) {
      const apiIndex = providerSrc.indexOf(`this._apiClient.${api}`);
      expect(apiIndex).toBeGreaterThan(-1);
      const guardIndex = providerSrc.lastIndexOf('_ensureBackendReady', apiIndex);
      expect(guardIndex).toBeGreaterThan(-1);
    }
  });

  it('passes ensureBackend into the settings provider at activation', () => {
    expect(extensionSrc).toContain(
      'new SettingsWebviewProvider(context.extensionUri, apiClient, settings, ensureBackend)'
    );
  });

  it('re-checks backend health on every call and re-sends saved settings to new processes', () => {
    // Caching "started OK" forever reported a crashed backend as running.
    expect(extensionSrc).not.toContain('startupPromise');
    expect(extensionSrc).toContain('backendManager.ensureRunning()');
    expect(extensionSrc).toContain('syncBackendConfig');
    expect(extensionSrc).toContain('backendManager.generation');
  });

  it('probes multiple interpreters instead of trusting bare `python`', () => {
    const managerSrc = read(path.join('src', 'services', 'backendManager.ts'));
    expect(managerSrc).toContain('getPythonCandidates');
  });

  it('handles spawn failures instead of timing out silently', () => {
    const managerSrc = read(path.join('src', 'services', 'backendManager.ts'));
    expect(managerSrc).toContain('failed to spawn');
  });

  it('no longer tells users to start the backend with a stale manual command', () => {
    const managerSrc = read(path.join('src', 'services', 'backendManager.ts'));
    expect(managerSrc).not.toContain('python -m uvicorn backend.main');
  });
});

describe('dev launch backend sync', () => {
  it('syncs the bundled backend before a dev-host launch', () => {
    const manifest = JSON.parse(read('package.json'));
    expect(manifest.scripts.prelaunch).toContain('package:backend');
    const launch = JSON.parse(read(path.join('.vscode', 'launch.json')));
    const config = launch.configurations.find((c: { name: string }) => c.name === 'Run Extension');
    expect(config.preLaunchTask).toBe('npm: prelaunch');
  });
});

describe('webview wiring', () => {
  // The chat script used to live in a TypeScript template literal where
  // `split('\n')` turned into a real newline inside a JS string. The page then
  // failed to parse and every button was dead with an empty console. Scripts
  // are real files now, and the providers must load them (not inline code).
  const settingsSrc = read(path.join('src', 'views', 'settingsWebviewProvider.ts'));
  const chatSrc = read(path.join('src', 'views', 'chatWebviewProvider.ts'));

  it('loads the views from media/ scripts and no longer inlines any script', () => {
    expect(chatSrc).toContain("'media', 'chat.js'");
    expect(settingsSrc).toContain("'media', 'settings.js'");
    expect(chatSrc).not.toContain('<script>');
    expect(settingsSrc).not.toContain('<script>');
  });

  it('logs webview message-handler failures instead of failing silently', () => {
    expect(chatSrc).toContain('Chat webview handler failed');
    expect(settingsSrc).toContain('Settings webview handler failed');
  });

  it('reports settings results without re-rendering (and wiping) the form', () => {
    expect(settingsSrc).toContain("'showStatus'");
    expect(settingsSrc).toContain('_postStatus');
    // `webview.html =` may only appear in the initial render.
    expect(settingsSrc.split('webview.html =').length - 1).toBe(1);
  });

  it('never sends key material back to the page', () => {
    // The page only learns whether a key is saved.
    expect(settingsSrc).toContain('hasApiKey');
    expect(settingsSrc).not.toMatch(/postMessage\([^)]*apiKey/);
  });

  it('bundles the media scripts and the requirements checker into the VSIX', () => {
    const ignore = read('.vscodeignore');
    expect(ignore).not.toMatch(/^media\/?\*?\*?$/m);
    expect(ignore).not.toMatch(/^\*\*\/\*\.js$/m);
  });
});

describe('backend startup dependency flow', () => {
  const managerSrc = read(path.join('src', 'services', 'backendManager.ts'));
  const dependencySrc = read(path.join('src', 'services', 'dependencyManager.ts'));

  it('verifies version + full runtime requirements before starting Python', () => {
    expect(managerSrc).toContain('probePythonEnvironment');
    expect(dependencySrc).toContain('MIN_PYTHON_VERSION');
  });

  it('offers a consented install exactly once per session, then re-verifies', () => {
    expect(managerSrc).toContain('ensureRuntimeDependencies');
    expect(managerSrc).toContain('dependencyPromptDismissed');
    expect(dependencySrc).toContain('recheck.available');
  });

  it('never installs the root developer requirements for normal users', () => {
    expect(dependencySrc).toContain('backend/requirements.txt');
    // checkDependencies resolves requirements.txt next to main.py (the
    // bundled backend dir), never the repository root.
    expect(dependencySrc).toContain("path.join(backendDir, 'requirements.txt')");
  });

  it('logs spawn/exit failures to the backend output channel', () => {
    expect(managerSrc).toContain('failed to spawn');
    expect(managerSrc).toContain('Backend process exited with code');
  });
});
