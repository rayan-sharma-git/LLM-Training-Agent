import * as vscode from 'vscode';
import { ApiClient } from '../services/apiClient';
import { formatError } from '../utils';

/**
 * A WebviewViewProvider that renders a functional chat interface
 * in the sidebar. Messages are sent to the backend API and the
 * assistant's response is displayed in a scrollable conversation view.
 *
 * Supports action cards — interactive messages that present the user
 * with buttons (e.g. [Install Dependencies] [Cancel]).
 */
export class ChatWebviewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'llmTrainingAgent.chat';

  private _view?: vscode.WebviewView;
  private _disposables: vscode.Disposable[] = [];
  /** Resolve a pending action-button promise, keyed by action id. */
  private _actionResolvers: Map<string, (action: string) => void> = new Map();
  /**
   * Stable conversation id for this webview. History is additionally scoped
   * by project path on the backend, so switching workspaces never leaks
   * conversations between projects.
   */
  private readonly _sessionId = 'default';

  constructor(
    private readonly _extensionUri: vscode.Uri,
    private readonly _apiClient: ApiClient,
    private readonly _ensureBackend?: () => Promise<boolean>
  ) {}

  /**
   * Absolute path of the project the chat should talk about.
   *
   * Resolved on every call from the live workspace, so switching from Project A
   * to Project B immediately scopes the conversation to Project B. When no
   * folder is open the chat still works, just without project context.
   */
  private _getProjectPath(): string | undefined {
    return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  }

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      // Allow scripts so the chat UI can send messages.
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtml(webviewView.webview);

    // Handle messages coming from the webview.
    webviewView.webview.onDidReceiveMessage(
      async (message) => {
        if (!message || typeof message !== 'object' || typeof message.command !== 'string') {
          return;
        }
        switch (message.command) {
          case 'sendMessage': {
            const text = message.text;
            if (!text || !text.trim()) {
              return;
            }
            await this._handleSendMessage(text);
            break;
          }
          case 'ready': {
            // The webview finished loading — replay persisted history.
            await this._restoreHistory();
            break;
          }
          case 'actionButtonClicked': {
            const actionId = message.actionId as string;
            const action = message.action as string;
            const resolver = this._actionResolvers.get(actionId);
            if (resolver) {
              resolver(action);
              this._actionResolvers.delete(actionId);
            }
            break;
          }
        }
      },
      null,
      this._disposables
    );
  }

  /**
   * Post an assistant message to the chat view (used by other commands).
   */
  public postAssistantMessage(text: string): void {
    if (!this._view) {
      return;
    }
    this._view.webview.postMessage({
      command: 'appendMessage',
      role: 'assistant',
      text,
    });
  }

  /**
   * Post an action card — a message with clickable buttons — to the chat view.
   * Returns a promise that resolves to the action label the user clicked.
   * If the view is not available, the promise resolves to an empty string.
   */
  public postActionCard(options: {
    text: string;
    actionId: string;
    buttons: Array<{ label: string; action: string }>;
  }): Promise<string> {
    return new Promise((resolve) => {
      if (!this._view) {
        resolve('');
        return;
      }

      // Store the resolver so the webview message handler can call it.
      this._actionResolvers.set(options.actionId, resolve);

      this._view.webview.postMessage({
        command: 'showActionCard',
        actionId: options.actionId,
        text: options.text,
        buttons: options.buttons,
      });
    });
  }

  private async _handleSendMessage(text: string): Promise<void> {
    if (!this._view) {
      return;
    }

    // NOTE: the webview script already appended the user's message locally —
    // posting it back here would duplicate it in the conversation view.

    // Start the backend on first use rather than at activation.
    if (this._ensureBackend) {
      await this._ensureBackend();
    }

    try {
      const response = await this._apiClient.sendChatMessage(text, {
        projectPath: this._getProjectPath(),
        sessionId: this._sessionId,
      });
      this._view.webview.postMessage({
        command: 'appendMessage',
        role: 'assistant',
        text: response.assistantResponse,
        confidence: response.confidence,
        references: response.references,
      });
    } catch (error) {
      this._view.webview.postMessage({
        command: 'appendMessage',
        role: 'assistant',
        text: `⚠️ Failed to get a response from the backend: ${formatError(error)}`,
      });
    }
  }

  /**
   * Replay the backend's persisted conversation into the webview so history
   * survives webview reloads and extension restarts (within one backend run).
   */
  private async _restoreHistory(): Promise<void> {
    if (!this._view) {
      return;
    }
    if (this._ensureBackend) {
      await this._ensureBackend();
    }
    try {
      const { messages } = await this._apiClient.getChatHistory({
        projectPath: this._getProjectPath(),
        sessionId: this._sessionId,
      });
      if (!messages || messages.length === 0) {
        return;
      }
      this._view.webview.postMessage({
        command: 'restoreHistory',
        messages: messages.map((m) => ({ role: m.role, text: m.content })),
      });
    } catch {
      // History restore is best-effort; chat still works without it.
    }
  }

  private _getHtml(webview: vscode.Webview): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none';">
  <title>Chat</title>
  <style>
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
    .message .meta {
      font-size: 0.8em;
      opacity: 0.7;
      margin-bottom: 4px;
    }
    .message .refs {
      margin-top: 6px;
      font-size: 0.85em;
      opacity: 0.8;
    }
    .action-card {
      align-self: flex-start;
      background: var(--vscode-editor-inlineValues-background, var(--vscode-editor-background));
      color: var(--vscode-foreground);
      border: 1px solid var(--vscode-panel-border);
      border-radius: 6px;
      padding: 10px;
      max-width: 95%;
      word-wrap: break-word;
      white-space: pre-wrap;
      line-height: 1.4;
    }
    .action-card .card-text {
      margin-bottom: 10px;
    }
    .action-card .card-buttons {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .action-card .card-buttons button {
      padding: 6px 14px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-size: var(--vscode-font-size);
    }
    .action-card .card-buttons .primary-btn {
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    .action-card .card-buttons .primary-btn:hover {
      background: var(--vscode-button-hoverBackground);
    }
    .action-card .card-buttons .secondary-btn {
      background: var(--vscode-button-secondaryBackground, var(--vscode-editor-background));
      color: var(--vscode-button-secondaryForeground, var(--vscode-foreground));
      border: 1px solid var(--vscode-panel-border);
    }
    .action-card .card-buttons .secondary-btn:hover {
      background: var(--vscode-button-secondaryHoverBackground, var(--vscode-list-hoverBackground));
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
    .body h3 {
      font-size: 1.02em;
      margin: 12px 0 4px;
    }
    .body h4 {
      font-size: 0.95em;
      margin: 10px 0 4px;
      opacity: 0.9;
    }
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
    .body em { opacity: 0.75; }
  </style>
</head>
<body>
  <div id="messages">
    <div class="placeholder">Ask about your fine-tuning project…</div>
  </div>
  <div id="input-row">
    <textarea id="input" rows="2" placeholder="Type a message…"></textarea>
    <button id="send">Send</button>
  </div>
  <script>
    const vscode = acquireVsCodeApi();
    const messagesEl = document.getElementById('messages');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');

    function appendMessage(role, text, meta) {
      const placeholder = document.querySelector('.placeholder');
      if (placeholder) placeholder.remove();
      const div = document.createElement('div');
      div.className = 'message ' + role;
      if (meta) {
        const parts = [];
        if (meta.confidence) parts.push('Confidence: ' + meta.confidence);
        if (meta.refs && meta.refs.length) parts.push('References: ' + meta.refs.join(', '));
        if (parts.length) {
          const m = document.createElement('div');
          m.className = 'meta';
          m.textContent = parts.join(' · ');
          div.appendChild(m);
        }
      }
      const body = document.createElement('div');
      body.className = 'body';
      if (role === 'assistant') {
        renderMarkdown(body, text);
      } else {
        body.textContent = text;
      }
      div.appendChild(body);
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    /**
     * Minimal, safe Markdown renderer.
     *
     * The complete analysis is a long, heavily structured document, so plain
     * textContent would be unreadable. This handles the subset the backend
     * emits (headings, bold, inline code, blockquotes, lists) and — critically
     * — builds text nodes rather than injecting HTML, so project-supplied
     * content can never execute as markup (prompt-injection safe).
     */
    function renderMarkdown(root, text) {
      const lines = String(text).split('\n');
      let listEl = null;

      function inline(parent, content) {
        // Split on bold and code spans. The pattern is assembled with
        // String.fromCharCode so no backtick appears in this source: this JS
        // lives inside a TypeScript template literal, where a literal
        // backtick would terminate the surrounding string.
        var TICK = String.fromCharCode(96);
        var pattern = new RegExp(
          '(\\*\\*[^*]+\\*\\*|' + TICK + '[^' + TICK + ']+' + TICK + ')', 'g'
        );
        var lastIndex = 0;
        var match;
        while ((match = pattern.exec(content)) !== null) {
          if (match.index > lastIndex) {
            parent.appendChild(document.createTextNode(content.slice(lastIndex, match.index)));
          }
          var token = match[0];
          if (token.indexOf('**') === 0) {
            var strong = document.createElement('strong');
            strong.textContent = token.slice(2, -2);
            parent.appendChild(strong);
          } else {
            var code = document.createElement('code');
            code.textContent = token.slice(1, -1);
            parent.appendChild(code);
          }
          lastIndex = pattern.lastIndex;
        }
        if (lastIndex < content.length) {
          parent.appendChild(document.createTextNode(content.slice(lastIndex)));
        }
      }

      function closeList() {
        if (listEl) { listEl = null; }
      }

      lines.forEach(function(line) {
        const trimmed = line.trim();

        if (!trimmed) { closeList(); return; }

        if (trimmed.startsWith('### ')) {
          closeList();
          const h = document.createElement('h4');
          inline(h, trimmed.slice(4));
          root.appendChild(h);
          return;
        }
        if (trimmed.startsWith('## ')) {
          closeList();
          const h = document.createElement('h3');
          inline(h, trimmed.slice(3));
          root.appendChild(h);
          return;
        }
        if (trimmed.startsWith('# ')) {
          closeList();
          const h = document.createElement('h2');
          inline(h, trimmed.slice(2));
          root.appendChild(h);
          return;
        }
        if (trimmed.startsWith('> ')) {
          closeList();
          const q = document.createElement('blockquote');
          inline(q, trimmed.slice(2));
          root.appendChild(q);
          return;
        }
        if (trimmed === '---') {
          closeList();
          root.appendChild(document.createElement('hr'));
          return;
        }
        // List items, including the "  - " nested style used by recommendations.
        const listMatch = trimmed.match(/^([-*]|\d+\.)\s+(.*)$/);
        if (listMatch) {
          if (!listEl) {
            listEl = document.createElement('ul');
            root.appendChild(listEl);
          }
          const li = document.createElement('li');
          inline(li, listMatch[2]);
          listEl.appendChild(li);
          return;
        }

        closeList();
        const p = document.createElement('p');
        inline(p, trimmed);
        root.appendChild(p);
      });
    }

    function showActionCard(actionId, text, buttons) {
      const placeholder = document.querySelector('.placeholder');
      if (placeholder) placeholder.remove();
      const card = document.createElement('div');
      card.className = 'action-card';
      card.dataset.actionId = actionId;

      const textDiv = document.createElement('div');
      textDiv.className = 'card-text';
      textDiv.textContent = text;
      card.appendChild(textDiv);

      const btnRow = document.createElement('div');
      btnRow.className = 'card-buttons';
      buttons.forEach(function(btn, index) {
        const button = document.createElement('button');
        button.textContent = btn.label;
        button.className = index === 0 ? 'primary-btn' : 'secondary-btn';
        button.addEventListener('click', function() {
          // Disable all buttons to prevent double-clicks
          const allBtns = btnRow.querySelectorAll('button');
          allBtns.forEach(function(b) { b.disabled = true; });
          vscode.postMessage({
            command: 'actionButtonClicked',
            actionId: actionId,
            action: btn.action
          });
        });
        btnRow.appendChild(button);
      });
      card.appendChild(btnRow);
      messagesEl.appendChild(card);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function send() {
      const text = input.value.trim();
      if (!text) return;
      appendMessage('user', text);
      input.value = '';
      sendBtn.disabled = true;
      vscode.postMessage({ command: 'sendMessage', text });
    }

    sendBtn.addEventListener('click', send);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });

    window.addEventListener('message', (event) => {
      const message = event.data;
      switch (message.command) {
        case 'appendMessage':
          const meta = message.role === 'assistant'
            ? {
                confidence: message.confidence,
                refs: message.references,
              }
            : undefined;
          appendMessage(message.role, message.text, meta);
          sendBtn.disabled = false;
          break;
        case 'showActionCard':
          showActionCard(message.actionId, message.text, message.buttons);
          sendBtn.disabled = false;
          break;
        case 'restoreHistory':
          (message.messages || []).forEach(function(m) {
            appendMessage(m.role, m.text);
          });
          break;
      }
    });

    // Tell the extension we are ready to receive restored history.
    vscode.postMessage({ command: 'ready' });
  </script>
</body>
</html>`;
  }
}