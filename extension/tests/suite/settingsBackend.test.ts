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

  it('saves VS Code settings before syncing with the backend', () => {
    // updateProvider must persist the setting first so a slow/failed backend
    // start can never surface as "Settings update failed". The backend sync
    // afterwards is best-effort with its own warning message.
    const handlerIndex = providerSrc.indexOf("case 'updateProvider'");
    expect(handlerIndex).toBeGreaterThan(-1);
    const handler = providerSrc.slice(handlerIndex, handlerIndex + 2000);
    const settingsIndex = handler.indexOf('updateSettings');
    const backendIndex = handler.indexOf('_ensureBackendReady');
    const providerIndex = handler.indexOf('selectProvider');
    expect(settingsIndex).toBeGreaterThan(-1);
    expect(backendIndex).toBeGreaterThan(settingsIndex);
    expect(providerIndex).toBeGreaterThan(backendIndex);
    expect(handler).toContain('saved in settings, but the backend sync failed');
  });

  it('starts the backend before every other settings API call', () => {
    for (const api of [
      'setApiKey',
      'testConnection',
      'removeApiKey',
      'getProviderModels',
    ]) {
      const apiIndex = providerSrc.indexOf(api);
      expect(apiIndex).toBeGreaterThan(-1);
      // The startup guard must appear before the API call in each handler.
      const guardIndex = providerSrc.lastIndexOf('_ensureBackendReady', apiIndex);
      expect(guardIndex).toBeGreaterThan(-1);
    }
  });

  it('passes ensureBackend into the settings provider at activation', () => {
    expect(extensionSrc).toContain(
      'new SettingsWebviewProvider(context.extensionUri, apiClient, settings, ensureBackend)'
    );
  });

  it('probes multiple interpreters instead of trusting bare `python`', () => {
    const managerSrc = read(path.join('src', 'services', 'backendManager.ts'));
    expect(managerSrc).toContain('getPythonCandidates');
    expect(managerSrc).toContain('missingBackendDependencies');
    expect(managerSrc).toContain('resolvePythonCandidates');
  });

  it('handles spawn failures instead of timing out silently', () => {
    const managerSrc = read(path.join('src', 'services', 'backendManager.ts'));
    expect(managerSrc).toContain(".on('error'");
    expect(managerSrc).toContain('failed to spawn');
  });

  it('no longer tells users to start the backend with a stale manual command', () => {
    const apiClientSrc = read(path.join('src', 'services', 'apiClient.ts'));
    expect(apiClientSrc).not.toContain('cd backend && python -m uvicorn');
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

describe('webview button wiring', () => {
  // The Settings webview once wrapped all of its button wiring in a
  // DOMContentLoaded listener. Because the script runs at the end of <body>,
  // the webview document may already be past that point, so the listener
  // never fired and every Settings button silently did nothing. The Chat
  // webview already binds immediately, and the Settings webview must too.
  const settingsSrc = read(path.join('src', 'views', 'settingsWebviewProvider.ts'));
  const chatSrc = read(path.join('src', 'views', 'chatWebviewProvider.ts'));

  it('binds settings buttons immediately instead of waiting for DOMContentLoaded', () => {
    expect(settingsSrc).not.toContain("addEventListener('DOMContentLoaded'");
    expect(settingsSrc).not.toContain('addEventListener("DOMContentLoaded"');
  });

  it('keeps the chat send button wired through direct element binding', () => {
    expect(chatSrc).toContain("getElementById('send')");
    expect(chatSrc).toContain("getElementById('input')");
  });
});
