/**
 * HTML for the two sidebar webviews.
 *
 * Pure functions with no `vscode` import, so they can be unit-tested and
 * rendered in a plain browser. The behaviour lives in `media/chat.js` and
 * `media/settings.js`, loaded as external scripts under a nonce-based CSP. No
 * script is ever embedded in a TypeScript template literal: that is how the
 * chat script ended up with a literal newline inside a string and failed to
 * parse, leaving every button dead.
 */
import * as crypto from 'crypto';
import { PROVIDERS } from '../services/providers';

export function createNonce(): string {
  return crypto.randomBytes(16).toString('hex');
}

/** Escape text for use in HTML content or a double-quoted attribute. */
export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Locked-down policy: only the one nonce-tagged script may run, nothing may be
 * fetched (`default-src 'none'`), and the page talks to the extension solely
 * through postMessage.
 */
export function buildContentSecurityPolicy(cspSource: string, nonce: string): string {
  return [
    `default-src 'none'`,
    `style-src ${cspSource} 'unsafe-inline'`,
    `script-src 'nonce-${nonce}'`,
    `img-src ${cspSource} data:`,
    `font-src ${cspSource}`,
    // VS Code bootstraps webviews through a service worker; keep workers
    // available from the webview's own origin only.
    `worker-src ${cspSource} blob:`,
  ].join('; ');
}

export interface WebviewAssets {
  cspSource: string;
  nonce: string;
  /** Webview URI of the view's script (from `webview.asWebviewUri`). */
  scriptUri: string;
}

function head(title: string, assets: WebviewAssets, css: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${buildContentSecurityPolicy(assets.cspSource, assets.nonce)}">
  <title>${escapeHtml(title)}</title>
  <style>${css}</style>
</head>`;
}

const CHAT_CSS = `
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      color: var(--vscode-foreground);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .message {
      padding: 8px 10px;
      border-radius: 6px;
      max-width: 90%;
      word-wrap: break-word;
      white-space: pre-wrap;
      line-height: 1.4;
    }
    .message.user {
      align-self: flex-end;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    .message.assistant {
      align-self: flex-start;
      background: var(--vscode-editor-inlineValues-background, var(--vscode-editor-background));
      color: var(--vscode-foreground);
      border: 1px solid var(--vscode-panel-border);
    }
    .action-card {
      align-self: flex-start;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 6px;
      padding: 10px;
      max-width: 95%;
      white-space: pre-wrap;
      line-height: 1.4;
    }
    .action-card .card-text { margin-bottom: 10px; }
    .action-card .card-buttons { display: flex; gap: 8px; flex-wrap: wrap; }
    .action-card .card-buttons button {
      padding: 6px 14px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
    }
    .action-card .card-buttons button:disabled { opacity: 0.5; cursor: default; }
    .action-card .primary-btn {
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    .action-card .secondary-btn {
      background: var(--vscode-button-secondaryBackground, transparent);
      color: var(--vscode-button-secondaryForeground, var(--vscode-foreground));
      border: 1px solid var(--vscode-panel-border);
    }
    .message.thinking { opacity: 0.7; font-style: italic; }
    .message .meta {
      font-size: 0.8em;
      opacity: 0.7;
      margin-bottom: 4px;
    }
    #input-row {
      display: flex;
      gap: 6px;
      padding: 8px;
      border-top: 1px solid var(--vscode-panel-border);
    }
    #input {
      flex: 1;
      padding: 6px 8px;
      border: 1px solid var(--vscode-input-border, transparent);
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      border-radius: 4px;
      resize: none;
      font-family: inherit;
    }
    #send {
      padding: 6px 12px;
      border: none;
      border-radius: 4px;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      cursor: pointer;
    }
    #send:hover { background: var(--vscode-button-hoverBackground); }
    #send:disabled { opacity: 0.5; cursor: default; }
    .placeholder {
      color: var(--vscode-descriptionForeground);
      text-align: center;
      margin-top: 40px;
      font-style: italic;
    }
    /* Rendered Markdown in assistant messages */
    .body h2 {
      font-size: 1.15em;
      margin: 14px 0 6px;
      padding-bottom: 4px;
      border-bottom: 1px solid var(--vscode-panel-border);
    }
    .body h3 { font-size: 1.02em; margin: 12px 0 4px; }
    .body h4 { font-size: 0.95em; margin: 10px 0 4px; opacity: 0.9; }
    .body p { margin: 4px 0; }
    .body ul { margin: 4px 0; padding-left: 20px; }
    .body li { margin: 2px 0; }
    .body code {
      background: var(--vscode-textCodeBlock-background, rgba(127,127,127,0.15));
      padding: 1px 4px;
      border-radius: 3px;
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 0.9em;
    }
    .body pre {
      margin: 6px 0;
      padding: 8px;
      overflow-x: auto;
      white-space: pre;
      background: var(--vscode-textCodeBlock-background, rgba(127,127,127,0.15));
      border-radius: 4px;
    }
    .body pre code { background: none; padding: 0; }
    /* Code blocks: a header with the language and a Copy button */
    .code-block {
      margin: 6px 0;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 4px;
      overflow: hidden;
    }
    .code-block pre { margin: 0; border-radius: 0; }
    .code-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 3px 8px;
      font-size: 0.8em;
      background: var(--vscode-editorWidget-background, rgba(127,127,127,0.15));
      border-bottom: 1px solid var(--vscode-panel-border);
    }
    .code-lang { opacity: 0.75; text-transform: lowercase; }
    .copy-btn {
      padding: 2px 8px;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 3px;
      background: transparent;
      color: var(--vscode-foreground);
      font-family: inherit;
      font-size: inherit;
      cursor: pointer;
    }
    .copy-btn:hover { background: var(--vscode-toolbar-hoverBackground, rgba(127,127,127,0.2)); }
    .message-actions { display: flex; justify-content: flex-end; margin-top: 6px; font-size: 0.8em; }
    .body blockquote {
      margin: 6px 0;
      padding: 6px 10px;
      border-left: 3px solid var(--vscode-textBlockQuote-border, var(--vscode-panel-border));
      background: var(--vscode-textBlockQuote-background, transparent);
      opacity: 0.9;
    }
    .body hr {
      border: none;
      border-top: 1px solid var(--vscode-panel-border);
      margin: 12px 0;
    }
`;

export function buildChatHtml(assets: WebviewAssets): string {
  return `${head('Chat', assets, CHAT_CSS)}
<body>
  <div id="messages">
    <div class="placeholder">Ask about your fine-tuning project…</div>
  </div>
  <div id="input-row">
    <textarea id="input" rows="2" placeholder="Type a message…"></textarea>
    <button id="send">Send</button>
  </div>
  <script nonce="${assets.nonce}" src="${escapeHtml(assets.scriptUri)}"></script>
</body>
</html>`;
}

const SETTINGS_CSS = `
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
    .form-row { display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; }
    .form-row.label-first { flex-direction: row; align-items: center; gap: 8px; }
    .form-row label { font-size: 12px; color: #555; min-width: 75px; }
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
    .note { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 4px; }
    .key-state { font-size: 12px; margin-top: 4px; }
    .key-state.saved { color: #31a354; }
    .key-state.missing { color: var(--vscode-descriptionForeground); }
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
    .btn:disabled { opacity: 0.5; cursor: default; }
    .btn-primary { background: #007acc; color: #fff; }
    .btn-primary:hover:enabled { background: #0062a3; }
    .btn-secondary { background: #6a6a6a; color: #fff; }
    .btn-secondary:hover:enabled { opacity: 0.9; }
    .btn-danger { background: #e57373; color: #fff; }
    .btn-danger:hover:enabled { opacity: 0.9; }
    .status { padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 12px; white-space: pre-wrap; }
    .status.ok { background: #e6f4ea; border: 1px solid #a3cf82; color: #1e7a3c; }
    .status.error { background: #fce8e6; border: 1px solid #e5a9a9; color: #b3261e; }
    .provider-description { font-size: 12px; color: var(--vscode-descriptionForeground); margin-bottom: 8px; }
`;

export interface SettingsHtmlOptions extends WebviewAssets {
  provider: string;
  model: string;
  /** provider id -> whether a key is saved in SecretStorage (never the key). */
  keyStates: Record<string, boolean>;
  /** provider id -> configured endpoint URL. */
  baseUrls: Record<string, string>;
}

export function buildSettingsHtml(options: SettingsHtmlOptions): string {
  const providerOptions = PROVIDERS.map(
    (p) =>
      `<option value="${escapeHtml(p.id)}"${p.id === options.provider ? ' selected' : ''}>${escapeHtml(p.label)}</option>`
  ).join('\n        ');

  return `${head('AI Provider Settings', options, SETTINGS_CSS)}
<body
  data-provider="${escapeHtml(options.provider)}"
  data-model="${escapeHtml(options.model)}"
  data-key-states="${escapeHtml(JSON.stringify(options.keyStates))}"
  data-base-urls="${escapeHtml(JSON.stringify(options.baseUrls))}">
  <h1>AI Provider Settings</h1>

  <div class="section">
    <div class="section-title">AI Provider</div>
    <div class="form-row label-first">
      <label for="providerSelect">Provider:</label>
      <select id="providerSelect">
        ${providerOptions}
      </select>
    </div>

    <div id="providerDescription" class="provider-description"></div>

    <div class="form-row label-first">
      <label for="modelInput">Model:</label>
      <input type="text" id="modelInput" list="modelList" placeholder="Enter model name" autocomplete="off" />
      <datalist id="modelList"></datalist>
    </div>

    <div class="form-row" id="apiKeySection" style="display: none;">
      <label for="apiKeyInput">API Key:</label>
      <input type="password" id="apiKeyInput" placeholder="Enter API key" autocomplete="off" />
      <div id="keyState" class="key-state"></div>
      <div class="note">Stored securely via VS Code SecretStorage; never written to a project file.</div>
    </div>

    <div class="form-row" id="baseUrlSection" style="display: none;">
      <label id="baseUrlLabel" for="baseUrlInput">URL:</label>
      <input type="text" id="baseUrlInput" autocomplete="off" />
      <div id="baseUrlNote" class="note"></div>
    </div>

    <div id="statusMessage" class="status" style="display: none;" role="status"></div>
  </div>

  <div class="section">
    <div class="section-title">Actions</div>
    <button class="btn btn-primary" id="saveBtn">Save Configuration</button>
    <button class="btn btn-secondary" id="testBtn">Test Connection</button>
    <button class="btn btn-danger" id="removeKeyBtn" style="display: none;">Remove API Key</button>
    <button class="btn btn-secondary" id="refreshModelsBtn">Refresh Models</button>
  </div>

  <script nonce="${options.nonce}" src="${escapeHtml(options.scriptUri)}"></script>
</body>
</html>`;
}
