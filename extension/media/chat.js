/* LLM Training Agent chat view. Runs inside the webview; talks to the extension via postMessage only.
 * All text is inserted with textContent / text nodes, never as HTML, so project or model supplied
 * content cannot execute as markup. */
(function () {
  'use strict';
  var vscode = acquireVsCodeApi();

  var $ = function (id) { return document.getElementById(id); };
  var listScreen = $('list-screen'), chatScreen = $('chat-screen');
  var sessionsEl = $('sessions'), messagesEl = $('messages');
  var input = $('input'), sendBtn = $('send'), titleEl = $('chat-title');

  var state = { screen: 'list', busy: false, current: null, messages: [], localCounter: 0 };

  // ------------------------------------------------------------------ helpers
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function post(message) { try { vscode.postMessage(message); } catch (e) { /* host gone */ } }
  function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  function formatWhen(iso) {
    if (!iso) return '';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    var date = d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    var time = d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
    return date + ' · ' + time;
  }

  // ------------------------------------------------------------------ markdown
  var TICK = String.fromCharCode(96);
  var INLINE = new RegExp('(\\*\\*[^*]+\\*\\*|' + TICK + '[^' + TICK + ']+' + TICK + ')', 'g');

  function inline(parent, content) {
    var last = 0, match;
    INLINE.lastIndex = 0;
    while ((match = INLINE.exec(content)) !== null) {
      if (match.index > last) parent.appendChild(document.createTextNode(content.slice(last, match.index)));
      var token = match[0], node;
      if (token.indexOf('**') === 0) { node = el('strong', null, token.slice(2, -2)); }
      else { node = el('code', null, token.slice(1, -1)); }
      parent.appendChild(node);
      last = INLINE.lastIndex;
    }
    if (last < content.length) parent.appendChild(document.createTextNode(content.slice(last)));
  }

  function splitRow(line) {
    var cells = line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|');
    return cells.map(function (c) { return c.trim(); });
  }

  function renderMarkdown(root, text) {
    var lines = String(text).split('\n');
    var list = null, i = 0;
    var FENCE = TICK + TICK + TICK;
    while (i < lines.length) {
      var line = lines[i], t = line.trim();
      if (t.indexOf(FENCE) === 0) {
        var code = [];
        i++;
        while (i < lines.length && lines[i].trim().indexOf(FENCE) !== 0) { code.push(lines[i]); i++; }
        i++;
        var pre = el('pre'); pre.appendChild(el('code', null, code.join('\n'))); root.appendChild(pre);
        list = null; continue;
      }
      if (!t) { list = null; i++; continue; }
      if (t.charAt(0) === '|' && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
        var table = el('table'), head = el('tr');
        splitRow(t).forEach(function (c) { var th = el('th'); inline(th, c); head.appendChild(th); });
        table.appendChild(head);
        i += 2;
        while (i < lines.length && lines[i].trim().charAt(0) === '|') {
          var row = el('tr');
          splitRow(lines[i]).forEach(function (c) { var td = el('td'); inline(td, c); row.appendChild(td); });
          table.appendChild(row); i++;
        }
        root.appendChild(table); list = null; continue;
      }
      var node = null;
      if (t.indexOf('### ') === 0) { node = el('h4'); inline(node, t.slice(4)); }
      else if (t.indexOf('## ') === 0) { node = el('h3'); inline(node, t.slice(3)); }
      else if (t.indexOf('# ') === 0) { node = el('h2'); inline(node, t.slice(2)); }
      else if (t.indexOf('> ') === 0) { node = el('blockquote'); inline(node, t.slice(2)); }
      else if (t === '---') { node = el('hr'); }
      if (node) { root.appendChild(node); list = null; i++; continue; }
      var m = t.match(/^([-*]|\d+\.)\s+(.*)$/);
      if (m) {
        if (!list) { list = el('ul'); root.appendChild(list); }
        var li = el('li'); inline(li, m[2]); list.appendChild(li); i++; continue;
      }
      list = null;
      var p = el('p'); inline(p, t); root.appendChild(p); i++;
    }
  }

  // ------------------------------------------------------------------ blocks
  function renderDiff(block) {
    var box = el('div', 'diff');
    box.setAttribute('data-block', 'diff');
    var head = el('div', 'diff-head');
    head.appendChild(el('div', 'diff-path', block.filePath || '(unknown file)'));
    var sub = el('div', 'diff-sub');
    if (block.status) sub.appendChild(el('span', 'chip ' + String(block.status).toLowerCase(), block.note || block.status));
    else if (block.note) sub.appendChild(el('span', 'chip', block.note));
    sub.appendChild(el('span', 'stat-add', '+' + (block.additions || 0)));
    sub.appendChild(el('span', 'stat-del', '−' + (block.deletions || 0)));
    var n = (block.hunks || []).length;
    if (n) sub.appendChild(el('span', null, n + (n === 1 ? ' change' : ' changes')));
    head.appendChild(sub);
    box.appendChild(head);
    if (!n) box.appendChild(el('div', 'diff-note', 'No line changes to show.'));

    (block.hunks || []).forEach(function (hunk, index) {
      var h = el('div', 'hunk');
      h.setAttribute('data-hunk', String(index + 1));
      var hh = el('div', 'hunk-head');
      hh.appendChild(el('span', null, 'Change ' + (index + 1) + ' of ' + n));
      hh.appendChild(el('span', null, 'line ' + (hunk.oldStart || hunk.newStart || 1)));
      h.appendChild(hh);
      var code = el('div', 'hunk-code');
      var previous = null;
      (hunk.lines || []).forEach(function (line) {
        var type = line.type;
        if ((type === 'removed' || type === 'added') && type !== previous) {
          code.appendChild(el('div', 'run-label ' + type, type === 'removed' ? 'Removed' : 'Added'));
        }
        previous = type;
        if (type === 'note') { code.appendChild(el('div', 'dl note', line.text)); return; }
        var row = el('div', 'dl ' + type);
        var number = type === 'added' ? line.newNo : line.oldNo;
        row.appendChild(el('span', 'no', number === undefined || number === null ? '' : String(number)));
        row.appendChild(el('span', 'sign', type === 'removed' ? '−' : type === 'added' ? '+' : ' '));
        row.appendChild(el('span', 'txt', line.text === '' ? ' ' : line.text));
        code.appendChild(row);
      });
      h.appendChild(code);
      box.appendChild(h);
    });
    if (block.truncated) box.appendChild(el('div', 'diff-note', 'More changes exist than are shown here.'));
    return box;
  }

  function renderActions(block, messageId) {
    var card = el('div', 'card');
    card.setAttribute('data-block', 'actions');
    if (block.title) card.appendChild(el('div', 'card-title', block.title));
    if (block.text) card.appendChild(el('div', 'card-text', block.text));
    if (block.resolved) {
      card.appendChild(el('div', 'card-resolved', block.resolved.label || block.resolved.choice || 'Answered'));
      return card;
    }
    var row = el('div', 'card-buttons');
    (block.buttons || []).forEach(function (b, index) {
      var btn = el('button', 'btn' + (b.style === 'secondary' || (!b.style && index > 0) ? ' secondary' : ''), b.label);
      btn.type = 'button';
      btn.addEventListener('click', function () {
        row.querySelectorAll('button').forEach(function (x) { x.disabled = true; });
        setBusy(true);
        post({ command: 'respondAction', actionId: block.actionId, choice: b.choice, messageId: messageId });
      });
      row.appendChild(btn);
    });
    card.appendChild(row);
    return card;
  }

  function renderCommand(block) {
    var box = el('div', 'cmd');
    box.setAttribute('data-block', 'command');
    var head = el('div', 'cmd-head');
    head.appendChild(el('span', 'cmd-line', '$ ' + block.command));
    var failed = block.timedOut || (block.exitCode !== 0 && block.exitCode !== undefined && block.exitCode !== null);
    head.appendChild(el('span', 'cmd-status' + (failed ? ' fail' : ''),
      block.timedOut ? 'timed out' : 'exit ' + block.exitCode));
    box.appendChild(head);
    var out = el('pre', 'cmd-out', block.output ? block.output : '(no output)');
    box.appendChild(out);
    if (block.truncated) box.appendChild(el('div', 'diff-note', 'Output was shortened.'));
    return box;
  }

  function renderCommandButtons(block) {
    var row = el('div', 'card-buttons');
    row.setAttribute('data-block', 'commandButtons');
    (block.buttons || []).forEach(function (b) {
      var btn = el('button', 'btn', b.label);
      btn.type = 'button';
      btn.addEventListener('click', function () { post({ command: 'runCommand', id: b.command }); });
      row.appendChild(btn);
    });
    return row;
  }

  function renderBlocks(container, blocks, messageId) {
    clear(container);
    (blocks || []).forEach(function (block) {
      var node = null;
      switch (block && block.type) {
        case 'diff': node = renderDiff(block); break;
        case 'actions': node = renderActions(block, messageId); break;
        case 'command': node = renderCommand(block); break;
        case 'commandButtons': node = renderCommandButtons(block); break;
      }
      if (node) container.appendChild(node);
    });
  }

  // ------------------------------------------------------------------ copy
  /** Plain-text version of a message, including change viewers and command output. */
  function messageText(msg) {
    var parts = [msg.content || ''];
    (msg.blocks || []).forEach(function (b) {
      if (b.type === 'diff') {
        var lines = ['', b.filePath || ''];
        (b.hunks || []).forEach(function (h) {
          (h.lines || []).forEach(function (l) {
            if (l.type === 'removed') lines.push('- ' + l.text);
            else if (l.type === 'added') lines.push('+ ' + l.text);
            else if (l.type === 'context') lines.push('  ' + l.text);
          });
          lines.push('');
        });
        parts.push(lines.join('\n').replace(/\n+$/, ''));
      } else if (b.type === 'command') {
        parts.push('\n$ ' + b.command + '\n' + (b.output || ''));
      } else if (b.type === 'actions') {
        var text = [b.title, b.text].filter(Boolean).join('\n');
        if (text) parts.push(text);
        if (b.resolved) parts.push('[' + (b.resolved.label || b.resolved.choice) + ']');
      }
    });
    return parts.filter(function (p) { return p !== ''; }).join('\n').trim();
  }

  function copyButton(msg) {
    var btn = el('button', 'copy-btn', 'Copy Message');
    btn.type = 'button';
    btn.title = 'Copy Message';
    btn.setAttribute('aria-label', 'Copy Message');
    btn.addEventListener('click', function () {
      post({ command: 'copyText', text: messageText(msg) });
      btn.textContent = 'Copied';
      btn.classList.add('done');
      setTimeout(function () { btn.textContent = 'Copy Message'; btn.classList.remove('done'); }, 1500);
    });
    return btn;
  }

  // ------------------------------------------------------------------ messages
  function addMessage(msg) {
    var key = msg.messageId || ('local-' + (++state.localCounter));
    msg.messageId = key;
    state.messages.push(msg);
    var wrap = el('div', 'msg ' + msg.role + (msg.isError ? ' error' : ''));
    wrap.setAttribute('data-message-id', key);
    if (msg.content || msg.role === 'user') {
      var bubble = el('div', 'bubble');
      var body = el('div', 'body');
      if (msg.role === 'assistant') renderMarkdown(body, msg.content || ''); else body.textContent = msg.content || '';
      bubble.appendChild(body);
      wrap.appendChild(bubble);
    }
    var blocks = el('div', 'blocks');
    renderBlocks(blocks, msg.blocks, key);
    if (blocks.firstChild) wrap.appendChild(blocks);
    // Every message gets a copy button, including messages that only carry action buttons.
    var actions = el('div', 'msg-actions');
    actions.appendChild(copyButton(msg));
    wrap.appendChild(actions);
    messagesEl.appendChild(wrap);
    scrollDown();
    return wrap;
  }

  function replaceBlocks(messageId, blocks) {
    var msg = state.messages.filter(function (m) { return m.messageId === messageId; })[0];
    if (msg) msg.blocks = blocks;
    var wrap = messagesEl.querySelector('[data-message-id="' + String(messageId).replace(/"/g, '') + '"]');
    if (!wrap) return;
    var container = wrap.querySelector('.blocks');
    if (!container) {
      container = el('div', 'blocks');
      wrap.insertBefore(container, wrap.querySelector('.msg-actions'));
    }
    renderBlocks(container, blocks, messageId);
  }

  function showEmptyChat() {
    if (messagesEl.children.length === 0) {
      messagesEl.appendChild(el('div', 'empty', 'Ask about your fine-tuning project, or say what to change, for example “change epochs from 3 to 5”.'));
    }
  }
  function clearEmptyHint() {
    var hint = messagesEl.querySelector('.empty');
    if (hint) hint.remove();
  }

  function setBusy(on) {
    state.busy = on;
    sendBtn.disabled = on;
    var existing = messagesEl.querySelector('.busy');
    if (on && !existing) { messagesEl.appendChild(el('div', 'busy', 'Working…')); scrollDown(); }
    if (!on && existing) existing.remove();
  }

  // ------------------------------------------------------------------ screens
  function showScreen(name) {
    state.screen = name;
    listScreen.hidden = name !== 'list';
    chatScreen.hidden = name !== 'chat';
  }

  function renderSessions(sessions, hasProject) {
    showScreen('list');
    clear(sessionsEl);
    if (!sessions.length) {
      sessionsEl.appendChild(el('li', 'empty', hasProject
        ? 'No chats yet. Choose New chat to start one.'
        : 'Open a project folder, then choose New chat.'));
      return;
    }
    sessions.forEach(function (s) {
      var li = el('li', 'session');
      li.setAttribute('data-session-id', s.sessionId);
      var open = el('button', 'session-open');
      open.type = 'button';
      open.appendChild(el('span', 'session-title', s.title || 'New chat'));
      open.appendChild(el('span', 'session-meta', formatWhen(s.updatedAt || s.createdAt)));
      open.addEventListener('click', function () { post({ command: 'openSession', sessionId: s.sessionId }); });
      li.appendChild(open);
      var del = el('button', 'btn icon session-delete', '✕');
      del.type = 'button';
      del.title = 'Delete chat';
      del.setAttribute('aria-label', 'Delete chat');
      del.addEventListener('click', function (e) { e.stopPropagation(); askDelete(li, s); });
      li.appendChild(del);
      sessionsEl.appendChild(li);
    });
  }

  function askDelete(li, s) {
    li.classList.add('confirming');
    clear(li);
    li.appendChild(el('div', 'confirm-text', 'Delete “' + (s.title || 'New chat') + '”? This cannot be undone.'));
    var yes = el('button', 'btn danger', 'Delete'); yes.type = 'button';
    yes.addEventListener('click', function () { post({ command: 'deleteSession', sessionId: s.sessionId }); });
    var no = el('button', 'btn secondary', 'Cancel'); no.type = 'button';
    no.addEventListener('click', function () { post({ command: 'listSessions' }); });
    li.appendChild(yes); li.appendChild(no);
  }

  function showChat(sessionId, title, messages) {
    state.current = sessionId;
    state.messages = [];
    clear(messagesEl);
    titleEl.textContent = title || 'New chat';
    showScreen('chat');
    (messages || []).forEach(addMessage);
    showEmptyChat();
    setBusy(false);
    scrollDown();
    input.focus();
  }

  // ------------------------------------------------------------------ input
  function send() {
    var text = input.value.trim();
    if (!text || state.busy) return;
    clearEmptyHint();
    addMessage({ role: 'user', content: text });
    input.value = '';
    setBusy(true);
    post({ command: 'sendMessage', text: text });
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $('new-chat').addEventListener('click', function () { post({ command: 'newChat' }); });
  $('new-chat-2').addEventListener('click', function () { post({ command: 'newChat' }); });
  $('back').addEventListener('click', function () { post({ command: 'backToList' }); });

  // ------------------------------------------------------------------ host messages
  window.addEventListener('message', function (event) {
    var m = event.data || {};
    switch (m.command) {
      case 'showList': renderSessions(m.sessions || [], m.hasProject !== false); break;
      case 'showChat': showChat(m.sessionId, m.title, m.messages); break;
      case 'setTitle': titleEl.textContent = m.title || 'New chat'; break;
      case 'appendMessage':
        clearEmptyHint();
        if (m.sessionId) state.current = m.sessionId;
        addMessage(m.message);
        break;
      case 'replaceBlocks': replaceBlocks(m.messageId, m.blocks || []); break;
      case 'busy': setBusy(!!m.value); break;
      case 'localCard': {
        clearEmptyHint();
        var buttons = (m.buttons || []).map(function (b) { return { label: b.label, choice: b.action, style: undefined }; });
        var block = { type: 'actions', actionId: m.actionId, text: m.text, buttons: buttons, local: true };
        var msg = { role: 'assistant', content: '', blocks: [block] };
        var wrap = addMessage(msg);
        // Local cards answer through the extension's own resolver, not the backend.
        wrap.querySelectorAll('.card-buttons button').forEach(function (btn, index) {
          var clone = btn.cloneNode(true);
          btn.parentNode.replaceChild(clone, btn);
          clone.addEventListener('click', function () {
            wrap.querySelectorAll('.card-buttons button').forEach(function (x) { x.disabled = true; });
            post({ command: 'actionButtonClicked', actionId: m.actionId, action: m.buttons[index].action });
            block.resolved = { label: m.buttons[index].label };
            block.buttons = [];
            replaceBlocks(msg.messageId, [block]);
          });
        });
        break;
      }
    }
  });

  post({ command: 'ready' });
})();
