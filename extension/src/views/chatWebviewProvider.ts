import * as vscode from 'vscode';
import { ApiClient } from '../services/apiClient';
import { formatError } from '../utils';

/** VS Code commands a chat button may run. Anything else coming from the webview or a stored message is ignored. */
export const CHAT_COMMAND_WHITELIST: ReadonlySet<string> = new Set([
  'llmTrainingAgent.analyzeProject',
  'llmTrainingAgent.analyzeDataset',
  'llmTrainingAgent.configureProvider',
  'llmTrainingAgent.viewChanges',
]);

const MAX_COPY_CHARS = 2_000_000;

type Screen = 'list' | 'chat';

/**
 * The sidebar chat.
 *
 * Two screens: a scrollable list of previous chats (title, date and time) and a
 * conversation. Conversations are stored by the backend, so they survive closing VS Code
 * and switching chats; a chat is created automatically when the first message is sent.
 * The page itself lives in `media/chat.js` and `media/chat.css`.
 */
export class ChatWebviewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'llmTrainingAgent.chat';

  private _view?: vscode.WebviewView;
  private _disposables: vscode.Disposable[] = [];
  /** Resolve a pending local action-card promise, keyed by action id. */
  private _actionResolvers: Map<string, (action: string) => void> = new Map();
  /** The open conversation; undefined while a new, not yet created chat is on screen. */
  private _sessionId: string | undefined;
  private _screen: Screen = 'list';
  private _ready = false;
  private _queued: Array<() => void> = [];

  constructor(
    private readonly _extensionUri: vscode.Uri,
    private readonly _apiClient: ApiClient,
    private readonly _ensureBackend?: () => Promise<boolean>
  ) {}

  /** Project the chat belongs to, resolved from the live workspace on every call. */
  private _getProjectPath(): string | undefined {
    return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  }

  /** The conversation that new messages from commands (for example an analysis) are added to. */
  public getActiveSessionId(): string | undefined {
    return this._sessionId;
  }

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;
    this._ready = false;
    this._screen = 'list';

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this._extensionUri, 'media')],
    };
    webviewView.webview.html = this._getHtml(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(
      async (message) => {
        try {
          await this._onMessage(message);
        } catch (error) {
          // Never log message contents (they may hold keys); the command name is enough to diagnose.
          console.error(
            `Chat webview handler failed for "${message?.command ?? 'unknown'}":`,
            error instanceof Error ? error.message : String(error)
          );
          this._post({ command: 'busy', value: false });
          this._post({
            command: 'appendMessage',
            message: { role: 'assistant', content: `⚠️ ${formatError(error)}`, isError: true },
          });
        }
      },
      null,
      this._disposables
    );
    webviewView.onDidDispose(() => {
      this._view = undefined;
      this._ready = false;
    });
  }

  // ------------------------------------------------------------------
  // Webview → extension
  // ------------------------------------------------------------------
  private async _onMessage(message: any): Promise<void> {
    if (!message || typeof message !== 'object' || typeof message.command !== 'string') {
      return;
    }
    switch (message.command) {
      case 'ready':
        this._ready = true;
        await this._showList();
        this._queued.splice(0).forEach((fn) => fn());
        break;
      case 'listSessions':
      case 'backToList':
        await this._showList();
        break;
      case 'newChat':
        this._showNewChat();
        break;
      case 'openSession':
        if (typeof message.sessionId === 'string') {
          await this._openSession(message.sessionId);
        }
        break;
      case 'deleteSession':
        if (typeof message.sessionId === 'string') {
          await this._deleteSession(message.sessionId);
        }
        break;
      case 'sendMessage':
        if (typeof message.text === 'string' && message.text.trim()) {
          await this._handleSendMessage(message.text);
        }
        break;
      case 'respondAction':
        if (typeof message.actionId === 'string' && typeof message.choice === 'string') {
          await this._handleRespond(message.actionId, message.choice);
        }
        break;
      case 'copyText':
        if (typeof message.text === 'string') {
          await vscode.env.clipboard.writeText(message.text.slice(0, MAX_COPY_CHARS));
        }
        break;
      case 'runCommand':
        if (typeof message.id === 'string' && CHAT_COMMAND_WHITELIST.has(message.id)) {
          await vscode.commands.executeCommand(message.id);
        }
        break;
      case 'actionButtonClicked': {
        const resolver = this._actionResolvers.get(message.actionId as string);
        if (resolver) {
          resolver(String(message.action));
          this._actionResolvers.delete(message.actionId as string);
        }
        break;
      }
    }
  }

  // ------------------------------------------------------------------
  // Screens
  // ------------------------------------------------------------------
  private async _ready_backend(): Promise<boolean> {
    if (!this._ensureBackend) {
      return true;
    }
    return this._ensureBackend();
  }

  private async _showList(): Promise<void> {
    this._screen = 'list';
    this._sessionId = undefined;
    const projectPath = this._getProjectPath();
    let sessions: unknown[] = [];
    try {
      if (await this._ready_backend()) {
        sessions = (await this._apiClient.listChatSessions(projectPath)).sessions ?? [];
      }
    } catch {
      // The list is best-effort; chat itself still works and reports its own errors.
    }
    this._post({ command: 'showList', sessions, hasProject: !!projectPath });
  }

  private _showNewChat(): void {
    this._screen = 'chat';
    this._sessionId = undefined;
    this._post({ command: 'showChat', sessionId: null, title: 'New chat', messages: [] });
  }

  private async _openSession(sessionId: string): Promise<void> {
    await this._ready_backend();
    try {
      const data = await this._apiClient.getChatSession(sessionId, this._getProjectPath());
      this._screen = 'chat';
      this._sessionId = sessionId;
      this._post({ command: 'showChat', sessionId, title: data.title, messages: data.messages });
    } catch (error) {
      await this._showList();
      this._post({ command: 'appendMessage', message: { role: 'assistant', content: `⚠️ ${formatError(error)}`, isError: true } });
    }
  }

  private async _deleteSession(sessionId: string): Promise<void> {
    await this._ready_backend();
    try {
      await this._apiClient.deleteChatSession(sessionId, this._getProjectPath());
    } catch (error) {
      vscode.window.showErrorMessage(`Could not delete the chat: ${formatError(error)}`);
    }
    await this._showList();
  }

  // ------------------------------------------------------------------
  // Messages
  // ------------------------------------------------------------------
  private async _handleSendMessage(text: string): Promise<void> {
    // The webview already shows the user's message locally.
    try {
      if (!(await this._ready_backend())) {
        throw new Error('The Python backend could not be started. Check the "LLM Training Agent: Backend" output channel.');
      }
      const result = await this._apiClient.agentChat(text, {
        projectPath: this._getProjectPath(),
        sessionId: this._sessionId,
      });
      this._sessionId = result.sessionId;
      this._post({ command: 'setTitle', title: result.title });
      this._post({ command: 'appendMessage', sessionId: result.sessionId, message: result.message });
    } catch (error) {
      this._post({
        command: 'appendMessage',
        message: { role: 'assistant', content: `⚠️ Failed to get a response from the backend: ${formatError(error)}`, isError: true },
      });
    } finally {
      this._post({ command: 'busy', value: false });
    }
  }

  private async _handleRespond(actionId: string, choice: string): Promise<void> {
    try {
      const sessionId = this._sessionId;
      if (!sessionId) {
        throw new Error('This card belongs to a chat that is no longer open.');
      }
      const result = await this._apiClient.agentRespond(actionId, choice, {
        projectPath: this._getProjectPath(),
        sessionId,
      });
      if (result.resolvedMessageId && result.resolvedBlocks) {
        this._post({ command: 'replaceBlocks', messageId: result.resolvedMessageId, blocks: result.resolvedBlocks });
      }
      if (result.message) {
        this._post({ command: 'appendMessage', sessionId, message: result.message });
      }
    } catch (error) {
      this._post({
        command: 'appendMessage',
        message: { role: 'assistant', content: `⚠️ ${formatError(error)}`, isError: true },
      });
      // Re-open the conversation so a card that was already answered shows its real state.
      if (this._sessionId) {
        await this._openSession(this._sessionId);
      }
    } finally {
      this._post({ command: 'busy', value: false });
    }
  }

  // ------------------------------------------------------------------
  // API used by commands
  // ------------------------------------------------------------------

  /**
   * Add an assistant message to the open chat (creating and storing one if none is open), so it
   * is part of the persisted history rather than lost when the view closes.
   */
  public async postAssistantMessage(text: string, options: { title?: string } = {}): Promise<void> {
    let stored = false;
    try {
      if (await this._ready_backend()) {
        const previous = this._sessionId;
        const note = await this._apiClient.addChatNote({
          projectPath: this._getProjectPath(),
          sessionId: previous,
          role: 'assistant',
          content: text,
          title: options.title,
        });
        stored = true;
        this._sessionId = note.sessionId;
        if (this._view) {
          if (previous === note.sessionId && this._screen === 'chat') {
            this._post({ command: 'appendMessage', sessionId: note.sessionId, message: note.message });
            this._post({ command: 'setTitle', title: note.title });
          } else {
            await this._openSession(note.sessionId);
          }
        }
      }
    } catch (error) {
      console.error('Could not store a chat message:', error instanceof Error ? error.message : String(error));
    }
    if (!stored && this._view) {
      // Backend unavailable: still show the text in the open view (it will not be in the history).
      this._screenForMessage();
      this._post({ command: 'appendMessage', message: { role: 'assistant', content: text } });
    }
  }

  /**
   * Post an action card — a message with clickable buttons — to the chat view.
   * Resolves to the `action` of the clicked button, or '' when the view is unavailable.
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
      this._actionResolvers.set(options.actionId, resolve);
      this._screenForMessage();
      this._post({
        command: 'localCard',
        actionId: options.actionId,
        text: options.text,
        buttons: options.buttons,
      });
    });
  }

  /** Show the latest file changes inline in the chat (the View Changes command). */
  public async showChanges(): Promise<void> {
    await vscode.commands.executeCommand(`${ChatWebviewProvider.viewType}.focus`);
    const run = async () => {
      this._post({ command: 'busy', value: true });
      try {
        if (!(await this._ready_backend())) {
          throw new Error('The Python backend could not be started. Check the "LLM Training Agent: Backend" output channel.');
        }
        const result = await this._apiClient.showChanges({
          projectPath: this._getProjectPath(),
          sessionId: this._sessionId,
        });
        const previous = this._sessionId;
        this._sessionId = result.sessionId;
        if (previous === result.sessionId && this._screen === 'chat') {
          this._post({ command: 'appendMessage', sessionId: result.sessionId, message: result.message });
        } else {
          await this._openSession(result.sessionId);
        }
      } catch (error) {
        this._screenForMessage();
        this._post({
          command: 'appendMessage',
          message: { role: 'assistant', content: `⚠️ Could not show the changes: ${formatError(error)}`, isError: true },
        });
      } finally {
        this._post({ command: 'busy', value: false });
      }
    };
    if (this._ready) {
      await run();
    } else {
      this._queued.push(() => void run());
    }
  }

  /** Make sure a conversation (not the chat list) is on screen before a local message is posted. */
  private _screenForMessage(): void {
    if (this._screen !== 'chat') {
      this._showNewChat();
    }
  }

  private _post(message: Record<string, unknown>): void {
    if (!this._view) {
      return;
    }
    void this._view.webview.postMessage(message);
  }

  // ------------------------------------------------------------------
  // HTML
  // ------------------------------------------------------------------
  private _getHtml(webview: vscode.Webview): string {
    // VS Code serves every webview through an internal service worker. A CSP of
    // `default-src 'none'` with no explicit `worker-src` makes that registration fail on some
    // platforms ("The document is in an invalid state"), leaving a blank view whose buttons do
    // nothing. Allowing workers from the webview's own source keeps the bootstrap alive while
    // everything else stays locked down. The script and stylesheet are external files served
    // from the extension, so no inline script is needed.
    const mediaUri = (file: string) =>
      webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', file));
    const csp =
      `default-src 'none'; ` +
      `script-src ${webview.cspSource}; ` +
      `style-src ${webview.cspSource}; ` +
      `img-src ${webview.cspSource} data:; ` +
      `font-src ${webview.cspSource} data:; ` +
      `connect-src 'none'; ` +
      `worker-src ${webview.cspSource} blob:;`;
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <title>Chat</title>
  <link rel="stylesheet" href="${mediaUri('chat.css')}">
</head>
<body>
  <div id="app">
    <section id="list-screen">
      <div class="topbar">
        <span class="title">Chats</span>
        <button id="new-chat" class="btn" type="button">New chat</button>
      </div>
      <ul id="sessions"></ul>
    </section>
    <section id="chat-screen" hidden>
      <div class="topbar">
        <button id="back" class="btn icon" type="button" title="Back to chats" aria-label="Back to chats">&#8592;</button>
        <span class="title" id="chat-title">New chat</span>
        <button id="new-chat-2" class="btn secondary" type="button" title="New chat" aria-label="New chat">+</button>
      </div>
      <div id="messages"></div>
      <div id="composer">
        <textarea id="input" rows="2" placeholder="Type a message…"></textarea>
        <button id="send" class="btn" type="button">Send</button>
      </div>
    </section>
  </div>
  <script src="${mediaUri('chat.js')}"></script>
</body>
</html>`;
  }
}
