import * as vscode from 'vscode';
import { ApiClient } from '../services/apiClient';
import { SettingsManager } from '../services/settings';
import { formatError } from '../utils';
import { BASE_URL_PROVIDERS, PROVIDER_IDS, isKnownProvider } from '../services/providers';
import { baseUrlFor } from '../services/backendSync';
import { buildSettingsHtml, createNonce } from './webviewHtml';
import { SETTINGS_VIEW_ID } from './viewIds';

const MAX_MODEL_LENGTH = 255;
const MAX_KEY_LENGTH = 4096;
const MAX_URL_LENGTH = 2048;

/** A Save/Test message from `media/settings.js`, validated before use. */
interface ConfigurationMessage {
  provider: string;
  model: string;
  apiKey?: string;
  baseUrl?: string;
}

function parseConfiguration(message: Record<string, unknown>): ConfigurationMessage | undefined {
  const { provider, model, apiKey, baseUrl } = message;
  if (!isKnownProvider(provider)) {
    return undefined;
  }
  if (typeof model !== 'string' || model.length > MAX_MODEL_LENGTH) {
    return undefined;
  }
  if (apiKey !== undefined && (typeof apiKey !== 'string' || apiKey.length > MAX_KEY_LENGTH)) {
    return undefined;
  }
  if (baseUrl !== undefined && (typeof baseUrl !== 'string' || baseUrl.length > MAX_URL_LENGTH)) {
    return undefined;
  }
  return {
    provider,
    model: model.trim(),
    apiKey: apiKey?.trim() || undefined,
    baseUrl: baseUrl?.trim() || undefined,
  };
}

export class SettingsWebviewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'llmTrainingAgent.settings';

  private _view?: vscode.WebviewView;
  private _disposables: vscode.Disposable[] = [];

  constructor(
    private readonly _extensionUri: vscode.Uri,
    private readonly _apiClient: ApiClient,
    private readonly _settings: SettingsManager,
    private readonly _ensureBackend?: () => Promise<boolean>
  ) {}

  /**
   * Start the bundled backend on first use (mirrors the chat view). Throws with
   * an actionable message when it cannot be started.
   */
  private async _ensureBackendReady(): Promise<void> {
    if (!this._ensureBackend) {
      return;
    }
    if (!(await this._ensureBackend())) {
      throw new Error(
        'The Python backend could not be started. ' +
          'Check the "LLM Training Agent: Backend" output channel for details.'
      );
    }
  }

  public resolveWebviewView(
    view: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): Thenable<void> {
    this._view = view;
    view.onDidDispose(() => {
      if (this._view === view) {
        this._view = undefined;
      }
    });

    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    // Messages are handled before the first render so none can be missed.
    // Failures are logged by command name only: message bodies can hold keys.
    view.webview.onDidReceiveMessage(
      async (message) => {
        if (!message || typeof message !== 'object' || typeof message.command !== 'string') {
          return;
        }
        try {
          await this._handleMessage(message as Record<string, unknown> & { command: string });
        } catch (error) {
          console.error(`Settings webview handler failed for "${message.command}":`, formatError(error));
          this._postStatus(`Settings update failed: ${formatError(error)}`, true);
        } finally {
          void view.webview.postMessage({ command: 'idle' });
        }
      },
      null,
      this._disposables
    );

    return this._render(view);
  }

  private async _handleMessage(message: Record<string, unknown> & { command: string }): Promise<void> {
    switch (message.command) {
      case 'ready':
        return;
      case 'saveConfiguration': {
        const config = parseConfiguration(message);
        if (!config || !config.model) {
          this._postStatus('Choose a provider and enter a model name.', true);
          return;
        }
        await this._save(config);
        return;
      }
      case 'testConnection': {
        const config = parseConfiguration(message);
        if (!config) {
          this._postStatus('Choose a provider and enter a model name.', true);
          return;
        }
        await this._test(config);
        return;
      }
      case 'removeApiKey': {
        if (!isKnownProvider(message.provider)) {
          return;
        }
        await this._removeKey(message.provider);
        return;
      }
      case 'refreshModels': {
        if (!isKnownProvider(message.provider)) {
          return;
        }
        await this._refreshModels(message.provider);
        return;
      }
    }
  }

  /**
   * Persist the configuration, then apply it to the backend.
   *
   * Order matters: the key and VS Code settings are stored first, so a backend
   * that is slow or broken can never lose what the user just entered. The
   * backend also re-reads all of it from here every time it starts.
   */
  private async _save(config: ConfigurationMessage): Promise<void> {
    const update: Parameters<SettingsManager['updateSettings']>[0] = {
      provider: config.provider,
      model: config.model,
    };
    if (config.baseUrl && config.provider === 'ollama') {
      update.ollamaBaseUrl = config.baseUrl;
    } else if (config.baseUrl && config.provider === 'openai_compatible') {
      update.openaiCompatibleBaseUrl = config.baseUrl;
    }
    await this._settings.updateSettings(update);

    if (config.apiKey) {
      await this._apiClient.setApiKey(config.provider, config.apiKey);
      this._postKeyState(config.provider, true);
    }

    const keySaved = await this._apiClient.hasApiKey(config.provider);
    const needsKey = config.provider !== 'ollama';
    const saved = `Saved ${config.provider} / ${config.model}${config.apiKey ? ' and the API key' : ''}.`;

    try {
      await this._ensureBackendReady();
      const payload: Record<string, unknown> = { provider: config.provider, model: config.model };
      if (config.apiKey) {
        payload.apiKey = config.apiKey;
      }
      if (config.baseUrl && BASE_URL_PROVIDERS.includes(config.provider)) {
        payload.baseUrl = config.baseUrl;
      }
      await this._apiClient.updateConfig(payload);
      await this._apiClient.selectProvider(config.provider, config.model);
    } catch (error) {
      this._postStatus(
        `${saved} The backend could not apply it yet (${formatError(error)}). ` +
          'It will be applied automatically the next time the backend starts.',
        true
      );
      return;
    }

    if (needsKey && !keySaved) {
      this._postStatus(`${saved} No API key is saved for ${config.provider} yet, so chat will not work until you add one.`, true);
    } else {
      this._postStatus(`${saved} Ready to use.`, false);
    }
  }

  private async _test(config: ConfigurationMessage): Promise<void> {
    await this._ensureBackendReady();
    // A key typed in the form wins; otherwise test with the saved one, so the
    // button works after the settings are reopened and the field is empty.
    const apiKey = config.apiKey ?? (await this._apiClient.getApiKey(config.provider));
    const baseUrl =
      config.baseUrl ??
      baseUrlFor(config.provider, this._settings.getSettings()) ??
      undefined;
    const result = await this._apiClient.testConnection({
      provider: config.provider,
      model: config.model || undefined,
      apiKey,
      baseUrl: BASE_URL_PROVIDERS.includes(config.provider) ? baseUrl || undefined : undefined,
    });
    this._postStatus(result.message, !result.success);
  }

  private async _removeKey(provider: string): Promise<void> {
    await this._apiClient.removeApiKey(provider);
    this._postKeyState(provider, false);
    try {
      await this._ensureBackendReady();
      await this._apiClient.removeApiKeyBackend(provider);
      this._postStatus(`API key removed for ${provider}.`, false);
    } catch (error) {
      this._postStatus(
        `API key removed from VS Code for ${provider}. The running backend may still hold it until it restarts (${formatError(error)}).`,
        true
      );
    }
  }

  private async _refreshModels(provider: string): Promise<void> {
    await this._ensureBackendReady();
    const result = await this._apiClient.getProviderModels(provider);
    if (result.models.length > 0) {
      void this._view?.webview.postMessage({ command: 'models', provider, models: result.models });
      this._postStatus(`Loaded ${result.models.length} models for ${provider}.`, false);
    } else {
      this._postStatus(
        result.error
          ? `No models found for ${provider}: ${result.error}`
          : `No models found for ${provider}. Check that the service is running and configured.`,
        true
      );
    }
  }

  /** Reveal the settings view for the "Configure AI Provider" command. */
  public async reveal(): Promise<void> {
    if (this._view) {
      this._view.show?.(true);
      return;
    }
    await vscode.commands.executeCommand('workbench.view.extension.llm-training-agent');
    await vscode.commands.executeCommand(`${SETTINGS_VIEW_ID}.focus`);
  }

  /**
   * Status goes to the live page via postMessage. Re-rendering `webview.html`
   * would destroy the form (including a key the user just typed).
   */
  private _postStatus(message: string, isError: boolean): void {
    void this._view?.webview.postMessage({ command: 'showStatus', text: message, isError });
  }

  private _postKeyState(provider: string, saved: boolean): void {
    void this._view?.webview.postMessage({ command: 'keyState', provider, saved });
  }

  private async _render(view: vscode.WebviewView): Promise<void> {
    const config = this._settings.getSettings();
    const keyStates: Record<string, boolean> = {};
    for (const provider of PROVIDER_IDS) {
      keyStates[provider] = await this._apiClient.hasApiKey(provider);
    }
    const baseUrls: Record<string, string> = {};
    for (const provider of BASE_URL_PROVIDERS) {
      baseUrls[provider] = baseUrlFor(provider, config);
    }
    const scriptUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'settings.js'));
    view.webview.html = buildSettingsHtml({
      cspSource: view.webview.cspSource,
      nonce: createNonce(),
      scriptUri: scriptUri.toString(),
      provider: isKnownProvider(config.provider) ? config.provider : 'ollama',
      model: config.model || '',
      keyStates,
      baseUrls,
    });
  }
}
