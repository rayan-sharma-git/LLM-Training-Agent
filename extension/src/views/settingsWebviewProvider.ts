import * as vscode from 'vscode';
import { ApiClient } from '../services/apiClient';
import { SettingsManager } from '../services/settings';
import { escapeJsString, formatError } from '../utils';
import { SETTINGS_VIEW_ID } from './viewIds';

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
   * Start the bundled backend on first use (mirrors the chat view).
   *
   * The extension activates lazily and never spawns Python until it is
   * needed. Every other caller awaits `ensureBackend` before touching the
   * API; the settings view must do the same, otherwise saving settings in a
   * fresh window fails with ECONNREFUSED even though nothing is broken.
   * Throws when the backend could not be started so the caller surfaces the
   * standard "Settings update failed" message with actionable details.
   */
  private async _ensureBackendReady(): Promise<void> {
    if (!this._ensureBackend) {
      return;
    }
    const ok = await this._ensureBackend();
    if (!ok) {
      throw new Error(
        'The Python backend could not be started. ' +
          'Check the "LLM Training Agent: Backend" output channel for details.'
      );
    }
  }

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      // Allow scripts so the settings UI can interact.
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    // Initial render with current settings.
    this._updateUI();

    // Handle messages from the webview. Command names (never key material)
    // are logged on failure so "clicking Save does nothing" is diagnosable.
    webviewView.webview.onDidReceiveMessage(
      async (message) => {
        if (!message || typeof message !== 'object' || typeof message.command !== 'string') {
          return;
        }
        try {
          switch (message.command) {
            case 'updateProvider': {
              const { provider, model } = message;
              if (typeof provider !== 'string' || typeof model !== 'string') return;
              if (provider.length > 64 || model.length > 255) return;
              // Save the VS Code setting FIRST so it can never be blocked by a
              // backend that is still starting (or that fails to start). The
              // backend sync below is then best-effort with its own message.
              // Never re-render the whole view here: replacing `webview.html`
              // destroys the DOM (including a just-typed API key) which is why
              // "Save clears the key field". Status goes back via postMessage.
              await this._settings.updateSettings({ provider, model });
              this._postStatus(`Provider "${provider}" saved. Syncing with backend…`, false);
              try {
                await this._ensureBackendReady();
                await this._apiClient.selectProvider(provider, model);
                this._postStatus(`Provider "${provider}" saved and synced.`, false);
              } catch (error) {
                // The VS Code setting is already saved; still surface the
                // backend failure so the mismatch is visible.
                this._postStatus(
                  `Provider "${provider}" saved in settings, but the backend sync failed: ${formatError(error)}`,
                  true
                );
                vscode.window.showWarningMessage(
                  `Provider "${provider}" saved in settings, but the backend sync failed: ${formatError(error)}`
                );
                return;
              }
              break;
            }
            case 'saveApiKey': {
              const { provider, apiKey } = message;
              if (typeof provider !== 'string' || provider.length > 64 || typeof apiKey !== 'string' || apiKey.length > 4096) return;
              await this._ensureBackendReady();
              if (apiKey && apiKey.trim()) {
                await this._apiClient.setApiKey(provider, apiKey.trim());
                await this._apiClient.updateConfig({ provider, apiKey: apiKey.trim() });
                this._postStatus(`API key saved securely for ${provider}.`, false);
                vscode.window.showInformationMessage(
                  `API key saved securely for ${provider}.`
                );
              } else {
                await this._apiClient.removeApiKey(provider);
                await this._apiClient.removeApiKeyBackend(provider);
                this._postStatus(`API key removed for ${provider}.`, false);
              }
              break;
            }
            case 'testConnection': {
              const { provider, model, apiKey, baseUrl } = message;
              await this._ensureBackendReady();
              const result = await this._apiClient.testConnection({
                provider,
                model,
                apiKey,
                baseUrl,
              });
              if (result.success) {
                this._postStatus(result.message, false);
                vscode.window.showInformationMessage(result.message);
              } else {
                this._postStatus(result.message, true);
                this._handleConnectionError(result);
              }
              break;
            }
            case 'removeApiKey': {
              const { provider } = message;
              await this._ensureBackendReady();
              await this._apiClient.removeApiKey(provider);
              await this._apiClient.removeApiKeyBackend(provider);
              this._postStatus(`API key removed for ${provider}.`, false);
              break;
            }
            case 'refreshModels': {
              const { provider } = message;
              await this._refreshModels(provider);
              break;
            }
          }
        } catch (error) {
          console.error(
            `Settings webview handler failed for "${message?.command ?? 'unknown'}":`,
            error instanceof Error ? error.message : String(error)
          );
          vscode.window.showErrorMessage(`Settings update failed: ${formatError(error)}`);
        }
      },
      null,
      this._disposables
    );
  }

  /**
   * Reveals the settings webview in the sidebar when the user runs the
   * "Configure AI Provider" command.
   */
  public async reveal(): Promise<void> {
    if (this._view) {
      this._view.show?.(true);
      return;
    }
    // If the view hasn't been created yet, focus the container.
    await vscode.commands.executeCommand(`${SETTINGS_VIEW_ID}.focus`);
    await vscode.commands.executeCommand('workbench.view.extension.llm-training-agent');
  }

  private _handleConnectionError(result: { success: boolean; message: string; error?: string }): void {
    switch (result.error) {
      case 'VALIDATION_FAILED':
        vscode.window.showErrorMessage(
          `Connection failed: ${result.message}. ` +
          'Please check your API key and model.'
        );
        break;
      case 'CONNECTION_FAILED':
        vscode.window.showErrorMessage(
          `Connection error: ${result.message}. ` +
          'Please check your network and try again.'
        );
        break;
      default:
        vscode.window.showErrorMessage(result.message);
    }
  }

  private async _refreshModels(provider: string): Promise<void> {
    try {
      await this._ensureBackendReady();
      const result = await this._apiClient.getProviderModels(provider);
      if (result.models.length > 0) {
        this._postStatus(`Loaded ${result.models.length} models for ${provider}.`, false);
        vscode.window.showInformationMessage(
          `Loaded ${result.models.length} models for ${provider}.`
        );
      } else {
        this._postStatus(`No models found for ${provider}. Check if the service is running and configured.`, true);
        vscode.window.showWarningMessage(
          `No models found for ${provider}. Check if the service is running and configured.`
        );
      }
    } catch (error) {
      const message = `Failed to refresh models: ${formatError(error)}`;
      this._postStatus(message, true);
      vscode.window.showErrorMessage(message);
    }
  }

  /**
   * Send a status line into the live settings DOM without replacing
   * `webview.html`. Re-rendering the whole view destroys form state (the
   * "Save clears the API-key field" symptom); postMessage keeps the user's
   * input intact while still confirming persistence.
   */
  private _postStatus(message: string, isError: boolean): void {
    try {
      void this._view?.webview.postMessage({ command: 'showStatus', text: message, isError });
    } catch (error) {
      console.error('Failed to post settings status:', error instanceof Error ? error.message : String(error));
    }
  }

  private _getModelForProvider(provider: string): string | undefined {
    const settings = this._settings.getSettings();
    return settings.model || undefined;
  }

  private async _updateUI(): Promise<void> {
    try {
      const config = this._settings.getSettings();
      let providersResult: { providers: Record<string, any> } | null = null;
      try {
        providersResult = await this._apiClient.listProviders();
      } catch (error) {
        // Backend not available; still render the UI with defaults.
        console.log('listProviders failed (backend may be starting):', formatError(error));
      }

      const providerStates: Record<string, any> = {};
      if (providersResult) {
        for (const [name, info] of Object.entries(providersResult.providers)) {
          providerStates[name] = {
            configured: info.configured,
            requiresKey: info.requiresKey,
          };
        }
      }

      const activeProvider = config.provider || 'ollama';
      const activeModel = config.model || 'llama3.2';

      if (this._view) {
        this._view.webview.html = this._getHtml(
          this._view.webview,
          activeProvider,
          activeModel,
          providerStates
        );
      }
    } catch (error) {
      console.error('Failed to update settings UI:', error);
    }
  }

  private _getHtml(
    webview: vscode.Webview,
    activeProvider: string,
    activeModel: string,
    providerStates: Record<string, any>
  ): string {
    // Same service-worker CSP contract as the chat view: VS Code bootstraps
    // every webview through an internal worker, and `default-src 'none'`
    // without an explicit `worker-src` fails that registration on some
    // platforms ("The document is in an invalid state"), leaving a blank
    // view whose buttons silently do nothing.
    const csp =
      `default-src 'none'; ` +
      `script-src ${webview.cspSource} 'unsafe-inline'; ` +
      `style-src ${webview.cspSource} 'unsafe-inline'; ` +
      `img-src ${webview.cspSource} data: https:; ` +
      `font-src ${webview.cspSource} data:; ` +
      `connect-src 'none'; ` +
      `worker-src ${webview.cspSource} blob:;`;
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <title>AI Provider Settings</title>
  <style>
    :root {
      --vscode-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      --vscode-font-size: 14px;
      --vscode-foreground: #1e1e1e;
      --vscode-button-background: #007acc;
      --vscode-button-foreground: #ffffff;
      --vscode-input-background: #ffffff;
      --vscode-input-foreground: #1e1e1e;
      --vscode-input-border: #d9d9d9;
      --vscode-button-hoverBackground: #0062a3;
      --vscode-descriptionForeground: #6a6a6a;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      color: var(--vscode-foreground);
      padding: 16px;
      background: var(--vscode-input-background);
    }

    h1 {
      color: #1a1a1a;
      border-bottom: 2px solid #007acc;
      padding-bottom: 8px;
      margin-bottom: 16px;
      font-size: 18px;
    }

    .section { margin-bottom: 20px; }

    .section-title {
      font-weight: bold;
      font-size: 14px;
      color: #333;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
    }

    .section-title::before {
      content: '';
      width: 4px;
      height: 14px;
      background: #007acc;
      border-radius: 2px;
      margin-right: 8px;
    }

    .form-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 10px;
    }

    .form-row.label-first {
      flex-direction: row;
      align-items: center;
      gap: 8px;
    }

    .form-row label {
      font-size: 12px;
      color: #555;
      min-width: 75px;
    }

    .form-row input,
    .form-row select {
      flex: 1;
      padding: 6px 8px;
      border: 1px solid var(--vscode-input-border);
      border-radius: 4px;
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      font-family: inherit;
      font-size: 13px;
      width: 100%;
    }

    .form-row input[type="password"] {
      -webkit-text-security: disc;
    }

    .note { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 4px; }

    .btn {
      padding: 8px 16px;
      border: none;
      border-radius: 4px;
      font-size: 13px;
      font-family: inherit;
      cursor: pointer;
      margin-right: 8px;
      margin-bottom: 8px;
    }

    .btn-primary { background: #007acc; color: #fff; }
    .btn-primary:hover { background: #0062a3; }
    .btn-secondary { background: #6a6a6a; color: #fff; }
    .btn-secondary:hover { opacity: 0.9; }
    .btn-danger { background: #e57373; color: #fff; }
    .btn-danger:hover { opacity: 0.9; }

    .status {
      padding: 8px;
      border-radius: 4px;
      margin-top: 8px;
      font-size: 12px;
    }
    .status.ok { background: #e6f4ea; border: 1px solid #a3cf82; color: #31a354; }
    .status.error { background: #fce8e6; border: 1px solid #e5a9a9; color: #e57373; }

    .provider-description { font-size: 12px; color: var(--vscode-descriptionForeground); margin-bottom: 8px; }
  </style>
</head>
<body>
  <h1>AI Provider Settings</h1>

  <div class="section">
    <div class="section-title">AI Provider</div>
    <div class="form-row label-first">
      <label>Provider:</label>
      <select id="providerSelect">
        <option value="ollama" ${activeProvider === 'ollama' ? 'selected' : ''}>Ollama — Local (no API key)</option>
        <option value="openai" ${activeProvider === 'openai' ? 'selected' : ''}>OpenAI</option>
        <option value="anthropic" ${activeProvider === 'anthropic' ? 'selected' : ''}>Anthropic</option>
        <option value="gemini" ${activeProvider === 'gemini' ? 'selected' : ''}>Google Gemini</option>
        <option value="deepseek" ${activeProvider === 'deepseek' ? 'selected' : ''}>DeepSeek</option>
        <option value="cohere" ${activeProvider === 'cohere' ? 'selected' : ''}>Cohere</option>
        <option value="openrouter" ${activeProvider === 'openrouter' ? 'selected' : ''}>OpenRouter</option>
        <option value="openai_compatible" ${activeProvider === 'openai_compatible' ? 'selected' : ''}>OpenAI-compatible (custom endpoint)</option>
      </select>
    </div>

    <div id="providerDescription" class="provider-description"></div>

    <div class="form-row label-first">
      <label>Model:</label>
      <div id="modelContainer" style="flex: 1;"></div>
    </div>

    <div class="form-row" id="apiKeySection" style="display: none;">
      <label>API Key:</label>
      <input type="password" id="apiKeyInput" placeholder="Enter API key" />
      <div class="note">Stored securely via VS Code SecretStorage.</div>
    </div>

    <div class="form-row" id="ollamaSection" style="display: none;">
      <label>Ollama URL:</label>
      <input type="text" id="ollamaBaseUrl" placeholder="http://localhost:11434" value="http://localhost:11434" />
      <div class="note">Make sure Ollama is running locally (ollama serve).</div>
    </div>

    <div id="statusMessage" class="status" style="display: none;"></div>
  </div>

  <div class="section">
    <div class="section-title">Actions</div>
    <button class="btn btn-primary" id="saveBtn">Save Configuration</button>
    <button class="btn btn-secondary" id="testBtn" style="display: none;">Test Connection</button>
    <button class="btn btn-danger" id="removeKeyBtn" style="display: none;">Remove API Key</button>
    <button class="btn btn-secondary" id="refreshModelsBtn">Refresh Models</button>
  </div>

  <script>
    const vscode = acquireVsCodeApi();

    let currentProvider = '${escapeJsString(activeProvider)}';
    let currentModel = '${escapeJsString(activeModel)}';
    const providerStates = ${JSON.stringify(providerStates)};

    const providerDescriptions = {
      ollama: 'Fully local. No API key required. Models are downloaded and run on your machine.',
      openai: 'Commercial API. Requires an API key from platform.openai.com.',
      anthropic: 'Commercial API. Requires an API key from console.anthropic.com.',
      gemini: 'Google AI Studio offers a free tier. Get a key at aistudio.google.com.',
      deepseek: 'Offers a free tier. Get a key at platform.deepseek.com.',
      cohere: 'Offers a free trial tier. Get a key at dashboard.cohere.com.',
      openrouter: 'Aggregates many models. Some are free. Get a key at openrouter.ai.',
      openai_compatible: 'Use a custom endpoint (e.g. LM Studio, vLLM, llama.cpp server).',
    };

    const staticModels = {
      openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1', 'gpt-4.1-mini'],
      anthropic: ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022'],
      gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
      deepseek: ['deepseek-chat', 'deepseek-coder'],
      cohere: ['command-r-plus', 'command-r'],
      openrouter: [],
      ollama: [],
      openai_compatible: [],
    };

    function setStatus(message, isError) {
      const el = document.getElementById('statusMessage');
      el.textContent = message;
      el.className = 'status ' + (isError ? 'error' : 'ok');
      el.style.display = 'block';
      setTimeout(() => { el.style.display = 'none'; }, 5000);
    }

    function updateProviderUI() {
      const desc = document.getElementById('providerDescription');
      desc.textContent = providerDescriptions[currentProvider] || '';

      const apiKeySection = document.getElementById('apiKeySection');
      const ollamaSection = document.getElementById('ollamaSection');
      const testBtn = document.getElementById('testBtn');
      const removeKeyBtn = document.getElementById('removeKeyBtn');

      const isOllama = currentProvider === 'ollama';
      const isOpenAICompat = currentProvider === 'openai_compatible';

      apiKeySection.style.display = isOllama ? 'none' : 'block';
      ollamaSection.style.display = (isOllama || isOpenAICompat) ? 'block' : 'none';
      testBtn.style.display = isOllama ? 'inline-block' : 'inline-block';
      removeKeyBtn.style.display = isOllama ? 'none' : 'inline-block';

      // Build model select — always replace inner HTML of the container.
      const modelContainer = document.getElementById('modelContainer');
      modelContainer.innerHTML = '';
      const models = staticModels[currentProvider] || [];
      const currentModelFromSettings = '${escapeJsString(activeModel)}';

      // If no static models and not ollama/compatible, allow free-text
      if (models.length === 0 || isOllama || isOpenAICompat) {
        const input = document.createElement('input');
        input.type = 'text';
        input.id = 'modelInput';
        input.value = currentModelFromSettings || (isOllama ? 'llama3.2' : '');
        input.placeholder = 'Enter model name';
        modelContainer.appendChild(input);
      } else {
        const select = document.createElement('select');
        select.id = 'modelSelect';
        for (const m of models) {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          if (m === currentModelFromSettings) opt.selected = true;
          select.appendChild(opt);
        }
        modelContainer.appendChild(select);
      }
    }

    function getModel() {
      const existing = document.getElementById('modelInput');
      if (existing) return existing.value;
      const select = document.getElementById('modelSelect');
      // The select only exists when a static list was rendered; otherwise the
      // input branch above already returned. Fall back to the escaped default.
      if (!select) return '${escapeJsString(activeModel)}';
      return select.value || '${escapeJsString(activeModel)}';
    }

    function getApiKey() {
      const el = document.getElementById('apiKeyInput');
      return el ? el.value.trim() : '';
    }

    function getBaseUrl() {
      const el = document.getElementById('ollamaBaseUrl');
      return el ? el.value.trim() : undefined;
    }

    // Save configuration (provider + model + api key)
    function saveConfiguration() {
      currentModel = getModel();
      const apiKey = getApiKey();

      vscode.postMessage({
        command: 'updateProvider',
        provider: currentProvider,
        model: currentModel,
      });

      if (apiKey && currentProvider !== 'ollama') {
        vscode.postMessage({
          command: 'saveApiKey',
          provider: currentProvider,
          apiKey: apiKey,
        });
      }
    }

    // Test connection
    function testConnection() {
      currentModel = getModel();
      const apiKey = getApiKey();
      const baseUrl = getBaseUrl();

      vscode.postMessage({
        command: 'testConnection',
        provider: currentProvider,
        model: currentModel,
        apiKey: apiKey || undefined,
        baseUrl: baseUrl,
      });
    }

    // Remove API key
    function removeApiKey() {
      vscode.postMessage({
        command: 'removeApiKey',
        provider: currentProvider,
      });
    }

    // Refresh models (ollama / openrouter / openai_compatible)
    function refreshModels() {
      vscode.postMessage({
        command: 'refreshModels',
        provider: currentProvider,
      });
    }

    // Init
    // Called immediately: this script runs after the elements above, and a
    // DOMContentLoaded wrapper is unreliable here — the webview document may
    // already be past that point, leaving every button unwired with no error.
    // Wrapped in try/catch with a visible fallback: if any handler fails to
    // attach, the view reports it instead of leaving dead buttons (the
    // "clicking Save/Analyze does nothing" symptom). Secrets are never
    // included in the error text.
    function initSettingsUi() {
      try {
      updateProviderUI();

      document.getElementById('providerSelect').addEventListener('change', function () {
        currentProvider = this.value;
        updateProviderUI();
      });
      document.getElementById('saveBtn').addEventListener('click', saveConfiguration);
      document.getElementById('testBtn').addEventListener('click', testConnection);
      document.getElementById('removeKeyBtn').addEventListener('click', removeApiKey);
      document.getElementById('refreshModelsBtn').addEventListener('click', refreshModels);

      document.getElementById('apiKeyInput').addEventListener('input', function () {
        if (this.value.trim()) {
          setStatus('API key entered — clicking Save stores it securely.', false);
        }
      });
      } catch (err) {
        setStatus('Settings failed to initialize. Reload the window; details are in the extension log.', true);
      }
    }

    // Status lines posted from the extension (save/connection results) that
    // must not destroy form state by re-rendering the whole view.
    window.addEventListener('message', function (event) {
      var message = event.data;
      if (message && message.command === 'showStatus') {
        setStatus(message.text || '', !!message.isError);
      }
    });
    initSettingsUi();
  </script>
</body>
</html>`;
  }
}