import * as vscode from 'vscode';
import { ApiClient } from '../services/apiClient';
import { formatError } from '../utils';
import { buildChatHtml, createNonce } from './webviewHtml';

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
  /**
   * Messages posted while the view was closed or still loading (for example an
   * Analyze Project result). They are delivered once the page reports `ready`,
   * instead of being dropped.
   */
  private _pending: Array<Record<string, unknown>> = [];
  private _webviewReady = false;

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
    this._webviewReady = false;
    webviewView.onDidDispose(() => {
      if (this._view === webviewView) {
        this._view = undefined;
        this._webviewReady = false;
      }
    });

    webviewView.webview.options = {
      // Allow scripts so the chat UI can send messages.
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtml(webviewView.webview);

    // Handle messages coming from the webview. Failures are logged (without
    // message contents, which may contain keys) so a dead handler cannot
    // hide as "clicking Send does nothing" with an empty console.
    webviewView.webview.onDidReceiveMessage(
      async (message) => {
        try {
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
            // The webview finished loading — replay persisted history, then
            // anything that was queued while it was not available.
            this._webviewReady = true;
            await this._restoreHistory();
            this._flushPending();
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
        } catch (error) {
          console.error(
            `Chat webview handler failed for "${message?.command ?? 'unknown'}":`,
            error instanceof Error ? error.message : String(error)
          );
        }
      },
      null,
      this._disposables
    );
  }

  /** Deliver now when the page is ready, otherwise queue until it is. */
  private _post(message: Record<string, unknown>): void {
    if (this._view && this._webviewReady) {
      void this._view.webview.postMessage(message);
    } else {
      this._pending.push(message);
    }
  }

  private _flushPending(): void {
    const queued = this._pending;
    this._pending = [];
    for (const message of queued) {
      this._post(message);
    }
  }

  /**
   * Post an assistant message to the chat view (used by other commands). When
   * the view is not open yet the message is kept and shown once it opens.
   */
  public postAssistantMessage(text: string): void {
    this._post({ command: 'appendMessage', role: 'assistant', text });
  }

  /** Bring the chat view to the front so results are visible. */
  public async reveal(): Promise<void> {
    if (this._view) {
      this._view.show?.(true);
      return;
    }
    await vscode.commands.executeCommand(`${ChatWebviewProvider.viewType}.focus`);
  }

  /**
   * Post an action card — a message with clickable buttons — to the chat view.
   * Returns a promise that resolves to the action the user clicked, or '' when
   * the chat view cannot be shown (the caller then falls back to a dialog).
   */
  public async postActionCard(options: {
    text: string;
    actionId: string;
    buttons: Array<{ label: string; action: string }>;
  }): Promise<string> {
    await this.reveal().then(undefined, () => undefined);
    // Give a freshly created view a moment to load before giving up.
    for (let i = 0; i < 20 && !(this._view && this._webviewReady); i++) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (!this._view || !this._webviewReady) {
      return '';
    }
    return new Promise((resolve) => {
      this._actionResolvers.set(options.actionId, resolve);
      void this._view!.webview.postMessage({
        command: 'showActionCard',
        actionId: options.actionId,
        text: options.text,
        buttons: options.buttons,
      });
    });
  }

  private async _handleSendMessage(text: string): Promise<void> {
    // NOTE: the webview script already appended the user's message locally —
    // posting it back here would duplicate it in the conversation view.
    const fail = (reason: string) =>
      this._post({
        command: 'appendMessage',
        role: 'assistant',
        reply: true,
        text: `⚠️ ${reason}`,
      });

    try {
      // Start the backend on first use rather than at activation.
      if (this._ensureBackend && !(await this._ensureBackend())) {
        fail(
          'The Python backend is not running, so the message could not be sent. ' +
            'Open the "LLM Training Agent: Backend" output channel for the reason.'
        );
        return;
      }
      const response = await this._apiClient.sendChatMessage(text, {
        projectPath: this._getProjectPath(),
        sessionId: this._sessionId,
      });
      this._post({
        command: 'appendMessage',
        role: 'assistant',
        reply: true,
        text: response.assistantResponse,
        confidence: response.confidence,
        references: response.references,
      });
    } catch (error) {
      fail(`Failed to get a response: ${formatError(error)}`);
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
    // Deliberately does not start the backend: merely opening the view must not
    // spawn Python or raise install prompts. If it is not up, there is nothing
    // to restore yet.
    try {
      const { messages } = await this._apiClient.getChatHistory({
        projectPath: this._getProjectPath(),
        sessionId: this._sessionId,
      });
      if (!messages || messages.length === 0) {
        return;
      }
      void this._view.webview.postMessage({
        command: 'restoreHistory',
        messages: messages.map((m) => ({ role: m.role, text: m.content })),
      });
    } catch {
      // History restore is best-effort; chat still works without it.
    }
  }

  private _getHtml(webview: vscode.Webview): string {
    // The behaviour lives in media/chat.js (a real, syntax-checkable file) and
    // runs under a nonce-based CSP. See webviewHtml.ts for why it is never
    // inlined in a template literal.
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.js'));
    return buildChatHtml({
      cspSource: webview.cspSource,
      nonce: createNonce(),
      scriptUri: scriptUri.toString(),
    });
  }
}
