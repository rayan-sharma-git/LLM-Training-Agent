/*
 * Chat view script.
 *
 * This is a real file (not a string inside a TypeScript template literal) on
 * purpose: inside a template literal, sequences such as '\n', \d and \s are
 * rewritten before the browser ever sees them, which silently produced a script
 * that failed to parse and left the Send button dead. As a file it can be
 * syntax-checked (npm run verify:wiring) and is exactly what the webview runs.
 *
 * Security: assistant and project-supplied text is only ever inserted with
 * textContent / text nodes, never as HTML.
 */
(function () {
  'use strict';

  var vscode = acquireVsCodeApi();
  var messagesEl = document.getElementById('messages');
  var input = document.getElementById('input');
  var sendBtn = document.getElementById('send');
  var thinkingEl = null;

  function removePlaceholder() {
    var placeholder = document.querySelector('.placeholder');
    if (placeholder) {
      placeholder.remove();
    }
  }

  function scrollToEnd() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ----- Minimal, safe Markdown renderer ---------------------------------
  // Handles the subset the backend emits: headings, **bold**, `code`,
  // blockquotes, lists, horizontal rules and fenced code blocks.
  var TICK = String.fromCharCode(96);
  var FENCE = TICK + TICK + TICK;
  var INLINE_PATTERN = new RegExp('(\\*\\*[^*]+\\*\\*|' + TICK + '[^' + TICK + ']+' + TICK + ')', 'g');
  var LIST_PATTERN = /^([-*]|\d+\.)\s+(.*)$/;

  function renderInline(parent, content) {
    INLINE_PATTERN.lastIndex = 0;
    var lastIndex = 0;
    var match;
    while ((match = INLINE_PATTERN.exec(content)) !== null) {
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
      lastIndex = INLINE_PATTERN.lastIndex;
    }
    if (lastIndex < content.length) {
      parent.appendChild(document.createTextNode(content.slice(lastIndex)));
    }
  }

  // A code block with a header: the language on the left, a Copy button on the
  // right that copies just this snippet.
  function appendCodeBlock(root, code, language) {
    var wrapper = document.createElement('div');
    wrapper.className = 'code-block';
    var header = document.createElement('div');
    header.className = 'code-header';
    var lang = document.createElement('span');
    lang.className = 'code-lang';
    lang.textContent = language || 'code';
    header.appendChild(lang);
    header.appendChild(makeCopyButton('Copy', function () { return code; }));
    wrapper.appendChild(header);
    var pre = document.createElement('pre');
    var codeEl = document.createElement('code');
    codeEl.textContent = code;
    pre.appendChild(codeEl);
    wrapper.appendChild(pre);
    root.appendChild(wrapper);
  }

  function renderMarkdown(root, text) {
    var lines = String(text).split('\n');
    var listEl = null;
    var fenceLines = null;
    var fenceLanguage = '';

    function closeList() {
      listEl = null;
    }

    function addBlock(tag, content) {
      closeList();
      var el = document.createElement(tag);
      renderInline(el, content);
      root.appendChild(el);
    }

    lines.forEach(function (line) {
      var trimmed = line.trim();

      if (trimmed.indexOf(FENCE) === 0) {
        if (fenceLines === null) {
          closeList();
          fenceLines = [];
          fenceLanguage = trimmed.slice(FENCE.length).trim();
        } else {
          appendCodeBlock(root, fenceLines.join('\n'), fenceLanguage);
          fenceLines = null;
        }
        return;
      }
      if (fenceLines !== null) {
        fenceLines.push(line);
        return;
      }

      if (!trimmed) {
        closeList();
        return;
      }
      if (trimmed.indexOf('### ') === 0) {
        addBlock('h4', trimmed.slice(4));
        return;
      }
      if (trimmed.indexOf('## ') === 0) {
        addBlock('h3', trimmed.slice(3));
        return;
      }
      if (trimmed.indexOf('# ') === 0) {
        addBlock('h2', trimmed.slice(2));
        return;
      }
      if (trimmed.indexOf('> ') === 0) {
        addBlock('blockquote', trimmed.slice(2));
        return;
      }
      if (trimmed === '---') {
        closeList();
        root.appendChild(document.createElement('hr'));
        return;
      }
      var listMatch = trimmed.match(LIST_PATTERN);
      if (listMatch) {
        if (!listEl) {
          listEl = document.createElement('ul');
          root.appendChild(listEl);
        }
        var li = document.createElement('li');
        renderInline(li, listMatch[2]);
        listEl.appendChild(li);
        return;
      }
      addBlock('p', trimmed);
    });

    // An unterminated fence still shows what it collected.
    if (fenceLines !== null && fenceLines.length) {
      appendCodeBlock(root, fenceLines.join('\n'), fenceLanguage);
    }
  }

  // ----- Copy buttons -------------------------------------------------------
  // The extension writes to the clipboard (it always can; a webview may not),
  // then confirms with a 'copied' message so the button can say so.
  var copyButtons = {};
  var copyCounter = 0;

  function makeCopyButton(label, getText) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-btn';
    button.textContent = label;
    var id = 'copy-' + (++copyCounter);
    copyButtons[id] = { button: button, label: label };
    button.addEventListener('click', function () {
      vscode.postMessage({ command: 'copy', id: id, text: getText() });
    });
    return button;
  }

  function onCopied(id) {
    var entry = copyButtons[id];
    if (!entry) {
      return;
    }
    entry.button.textContent = 'Copied ✓';
    setTimeout(function () {
      entry.button.textContent = entry.label;
    }, 1500);
  }

  // ----- Messages ---------------------------------------------------------
  function appendMessage(role, text, meta) {
    removePlaceholder();
    var div = document.createElement('div');
    div.className = 'message ' + role;
    if (meta) {
      var parts = [];
      if (meta.confidence) {
        parts.push('Confidence: ' + meta.confidence);
      }
      if (meta.refs && meta.refs.length) {
        parts.push('References: ' + meta.refs.join(', '));
      }
      if (parts.length) {
        var metaEl = document.createElement('div');
        metaEl.className = 'meta';
        metaEl.textContent = parts.join(' · ');
        div.appendChild(metaEl);
      }
    }
    var body = document.createElement('div');
    body.className = 'body';
    if (role === 'assistant') {
      renderMarkdown(body, text);
    } else {
      body.textContent = text;
    }
    div.appendChild(body);
    if (role === 'assistant') {
      var bar = document.createElement('div');
      bar.className = 'message-actions';
      bar.appendChild(makeCopyButton('Copy answer', function () { return String(text); }));
      div.appendChild(bar);
    }
    messagesEl.appendChild(div);
    scrollToEnd();
  }

  // Interactive card with buttons (for example "Install Dependencies" / "Cancel").
  function showActionCard(actionId, text, buttons) {
    removePlaceholder();
    var card = document.createElement('div');
    card.className = 'action-card';

    var textEl = document.createElement('div');
    textEl.className = 'card-text';
    textEl.textContent = text;
    card.appendChild(textEl);

    var row = document.createElement('div');
    row.className = 'card-buttons';
    (buttons || []).forEach(function (spec, index) {
      var button = document.createElement('button');
      button.textContent = spec.label;
      button.className = index === 0 ? 'primary-btn' : 'secondary-btn';
      button.addEventListener('click', function () {
        Array.prototype.forEach.call(row.querySelectorAll('button'), function (b) {
          b.disabled = true;
        });
        vscode.postMessage({ command: 'actionButtonClicked', actionId: actionId, action: spec.action });
      });
      row.appendChild(button);
    });
    card.appendChild(row);
    messagesEl.appendChild(card);
    scrollToEnd();
  }

  function setBusy(busy) {
    sendBtn.disabled = busy;
    if (busy && !thinkingEl) {
      removePlaceholder();
      thinkingEl = document.createElement('div');
      thinkingEl.className = 'message assistant thinking';
      thinkingEl.textContent = 'Thinking…';
      messagesEl.appendChild(thinkingEl);
      scrollToEnd();
    } else if (!busy && thinkingEl) {
      thinkingEl.remove();
      thinkingEl = null;
    }
  }

  function send() {
    var text = input.value.trim();
    if (!text || sendBtn.disabled) {
      return;
    }
    appendMessage('user', text);
    input.value = '';
    setBusy(true);
    vscode.postMessage({ command: 'sendMessage', text: text });
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  window.addEventListener('message', function (event) {
    var message = event.data;
    if (!message || typeof message !== 'object') {
      return;
    }
    switch (message.command) {
      case 'appendMessage':
        // Only the answer to a chat message ends the "Thinking…" state; an
        // unrelated message (for example an Analyze Project result) must not.
        if (message.reply) {
          setBusy(false);
        }
        appendMessage(
          message.role,
          message.text,
          message.role === 'assistant'
            ? { confidence: message.confidence, refs: message.references }
            : undefined
        );
        break;
      case 'copied':
        onCopied(message.id);
        break;
      case 'idle':
        setBusy(false);
        break;
      case 'showActionCard':
        showActionCard(message.actionId, message.text, message.buttons);
        break;
      case 'restoreHistory':
        (message.messages || []).forEach(function (m) {
          appendMessage(m.role, m.text);
        });
        break;
      default:
        break;
    }
  });

  // Tell the extension the listeners above are attached, so it can replay
  // history and deliver anything that was queued while the view was closed.
  vscode.postMessage({ command: 'ready' });
})();
