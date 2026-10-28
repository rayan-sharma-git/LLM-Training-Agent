/**
 * Browser check of the chat page (media/chat.js + media/chat.css) with a stubbed VS Code API.
 *
 * It builds the real HTML from the compiled provider (`npm run compile` first), loads it in
 * Chromium under the same kind of CSP, plays host messages into it and asserts on the DOM:
 * inline red/green change blocks, Copy Message on every message, cards, the chat list and
 * back/new-chat/delete flows, and a narrow sidebar.
 *
 * Needs Playwright (not a dependency of the extension):  npm i --no-save playwright
 * Usage: node scripts/verify-chat-ui.js [screenshotDir]
 */
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { pathToFileURL } = require('url');

const root = path.resolve(__dirname, '..');
const shotDir = process.argv[2] || path.join(os.tmpdir(), 'chat-ui-shots');
fs.mkdirSync(shotDir, { recursive: true });

// ---- build the page from the compiled provider ------------------------------------------
const realLoad = Module._load;
Module._load = function (request, ...rest) {
  if (request === 'vscode') {
    return { Uri: { joinPath: (base, ...parts) => ({ fsPath: path.join(base.fsPath, ...parts) }) } };
  }
  return realLoad.call(this, request, ...rest);
};
const { ChatWebviewProvider } = require(path.join(root, 'out', 'views', 'chatWebviewProvider.js'));
Module._load = realLoad;
const provider = new ChatWebviewProvider({ fsPath: root }, {}, undefined);
const fakeWebview = {
  cspSource: 'file:',
  asWebviewUri: (uri) => pathToFileURL(uri.fsPath).toString(),
};
// VS Code injects its theme as CSS variables; approximate Dark+ so screenshots look like the real thing.
const themePath = path.join(root, 'media', '.ui-theme.css');
fs.writeFileSync(themePath, `:root{--vscode-font-family:'Segoe UI',system-ui,sans-serif;--vscode-font-size:13px;
    --vscode-foreground:#cccccc;--vscode-editor-background:#1e1e1e;--vscode-panel-border:#454545;
    --vscode-button-background:#0e639c;--vscode-button-foreground:#fff;--vscode-button-hoverBackground:#1177bb;
    --vscode-button-secondaryBackground:#3a3d41;--vscode-button-secondaryForeground:#fff;
    --vscode-input-background:#3c3c3c;--vscode-input-foreground:#ccc;--vscode-descriptionForeground:#9d9d9d;
    --vscode-list-hoverBackground:#2a2d2e;--vscode-editorWidget-background:#252526;
    --vscode-editor-font-family:Consolas,monospace;--vscode-editor-font-size:12px;--vscode-focusBorder:#007fd4}
    body{background:#252526}`);
const html = provider._getHtml(fakeWebview).replace('</head>', `<link rel="stylesheet" href="${pathToFileURL(themePath)}"></head>`);
const pagePath = path.join(root, 'media', '.ui-check.html');
fs.writeFileSync(pagePath, html);

const failures = [];
function check(name, ok, detail) {
  if (ok) console.log(`ok   ${name}`);
  else { console.error(`FAIL ${name}${detail ? ' - ' + detail : ''}`); failures.push(name); }
}

const DIFF = {
  type: 'diff', changeId: 'c1', filePath: 'train.py', operation: 'modify', status: 'APPLIED',
  additions: 2, deletions: 2, truncated: false, note: 'Applied',
  hunks: [
    { id: 'h1', oldStart: 3, newStart: 3, removed: 1, added: 1, lines: [
      { type: 'context', text: 'from transformers import TrainingArguments', oldNo: 1, newNo: 1 },
      { type: 'removed', text: 'EPOCHS = 3', oldNo: 3 },
      { type: 'added', text: 'EPOCHS = 5', newNo: 3 },
      { type: 'context', text: 'LR = 2e-4', oldNo: 4, newNo: 4 } ] },
    { id: 'h2', oldStart: 20, newStart: 20, removed: 1, added: 1, lines: [
      { type: 'removed', text: 'batch_size = 4', oldNo: 21 },
      { type: 'added', text: 'batch_size = 8', newNo: 21 } ] },
  ],
};

(async () => {
  const { chromium } = require('playwright');
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 380, height: 760 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.addInitScript(() => {
    window.__posted = [];
    window.acquireVsCodeApi = () => ({ postMessage: (m) => window.__posted.push(m), getState: () => null, setState: () => {} });
  });
  await page.goto(pathToFileURL(pagePath).toString());
  const host = (data) => page.evaluate((d) => window.dispatchEvent(new MessageEvent('message', { data: d })), data);
  const posted = () => page.evaluate(() => window.__posted.slice());
  const last = async (command) => (await posted()).filter((m) => m.command === command).pop();

  check('page tells the extension it is ready', !!(await last('ready')));

  // ---- chat list ----
  await host({ command: 'showList', hasProject: true, sessions: [
    { sessionId: 's1', title: 'Change epochs from 3 to 5', updatedAt: '2026-10-28T14:05:00Z', messageCount: 4 },
    { sessionId: 's2', title: 'Project analysis', updatedAt: '2026-10-27T09:30:00Z', messageCount: 2 },
    { sessionId: 's3', title: 'Clean training data', updatedAt: '2026-10-26T18:45:00Z', messageCount: 6 },
  ] });
  check('list shows every session', (await page.locator('.session').count()) === 3);
  const meta = await page.locator('.session-meta').first().innerText();
  check('each session shows a date and time', /2026/.test(meta) && /\d{1,2}:\d{2}/.test(meta), meta);
  check('list scrolls inside its own container', await page.evaluate(() => getComputedStyle(document.getElementById('sessions')).overflowY === 'auto'));
  await page.screenshot({ path: path.join(shotDir, '1-list.png') });

  await page.locator('.session-open').first().click();
  check('opening a session asks the extension for it', (await last('openSession'))?.sessionId === 's1');

  // ---- conversation with an inline change viewer ----
  await host({ command: 'showChat', sessionId: 's1', title: 'Change epochs from 3 to 5', messages: [
    { messageId: 'm1', role: 'user', content: 'change epochs from 3 to 5' },
    { messageId: 'm2', role: 'assistant', content: 'Done - **Set epochs: 3 → 5**. The file was re-parsed and still validates.',
      blocks: [DIFF, { type: 'actions', actionId: 'a1', kind: 'revert_offer', title: '', text: '', resolved: null,
        buttons: [{ label: 'Revert', choice: 'revert', style: 'secondary' }] }] },
    { messageId: 'm3', role: 'assistant', content: '', blocks: [{ type: 'actions', actionId: 'a2', kind: 'permission',
      title: 'Permission needed', text: 'Allow writing train.py?', resolved: null,
      buttons: [{ label: 'Allow', choice: 'allow', style: 'primary' }, { label: 'Deny', choice: 'deny', style: 'secondary' }] }] },
  ] });
  check('title is shown', (await page.locator('#chat-title').innerText()).includes('epochs'));
  check('file path is shown', (await page.locator('.diff-path').innerText()) === 'train.py');
  check('changes are separate blocks (not two giant blocks)', (await page.locator('.hunk').count()) === 2);
  const removedLabels = await page.locator('.run-label.removed').allInnerTexts();
  const addedLabels = await page.locator('.run-label.added').allInnerTexts();
  check('removed lines are labelled "Removed"', removedLabels.length === 2 && removedLabels.every((t) => t === 'Removed'), removedLabels.join());
  check('replacement lines are labelled "Added"', addedLabels.length === 2 && addedLabels.every((t) => t === 'Added'), addedLabels.join());
  const colors = await page.evaluate(() => {
    const rgb = (el) => getComputedStyle(el).backgroundColor.match(/[\d.]+/g).map(Number);
    return { removed: rgb(document.querySelector('.dl.removed')), added: rgb(document.querySelector('.dl.added')) };
  });
  check('removed lines are red', colors.removed[0] > colors.removed[1] + 40, JSON.stringify(colors.removed));
  check('added lines are green', colors.added[1] > colors.added[0] + 40, JSON.stringify(colors.added));
  const removedRow = await page.locator('.dl.removed').first().innerText();
  const addedRow = await page.locator('.dl.added').first().innerText();
  check('removed row shows old line number, sign and code', /3\s*−\s*EPOCHS = 3/.test(removedRow), JSON.stringify(removedRow));
  check('added row shows new line number, sign and code', /3\s*\+\s*EPOCHS = 5/.test(addedRow), JSON.stringify(addedRow));
  check('stats are shown', (await page.locator('.stat-add').first().innerText()) === '+2');
  await page.screenshot({ path: path.join(shotDir, '2-diff.png') });

  // ---- Copy Message on every message, including card-only ones ----
  const messages = await page.locator('.msg').count();
  const copies = await page.locator('.msg .copy-btn').count();
  check('every message has a Copy Message button', messages === 3 && copies === 3, `${copies}/${messages}`);
  check('copy button is labelled Copy Message', (await page.locator('.copy-btn').first().getAttribute('aria-label')) === 'Copy Message');
  await page.locator('.msg').nth(1).locator('.copy-btn').click();
  const copied = await last('copyText');
  check('copy sends message text including the change', !!copied && copied.text.includes('Set epochs') &&
    copied.text.includes('train.py') && copied.text.includes('- EPOCHS = 3') && copied.text.includes('+ EPOCHS = 5'), JSON.stringify(copied));
  check('copy button confirms', (await page.locator('.msg').nth(1).locator('.copy-btn').innerText()) === 'Copied');
  await page.locator('.msg').nth(2).locator('.copy-btn').click();
  const copiedCard = await last('copyText');
  check('copying a card message copies its text', copiedCard.text.includes('Allow writing train.py?'), JSON.stringify(copiedCard));

  // ---- cards ----
  await page.getByRole('button', { name: 'Allow' }).click();
  const answered = await last('respondAction');
  check('Allow answers the card by id', answered && answered.actionId === 'a2' && answered.choice === 'allow', JSON.stringify(answered));
  check('buttons are disabled after a click', (await page.getByRole('button', { name: 'Deny' }).isDisabled()));
  await host({ command: 'replaceBlocks', messageId: 'm3', blocks: [{ type: 'actions', actionId: 'a2', title: 'Permission needed',
    text: 'Allow writing train.py?', buttons: [{ label: 'Allow', choice: 'allow' }], resolved: { choice: 'allow', label: 'Allowed' } }] });
  check('answered card shows its outcome and no buttons', (await page.locator('[data-message-id="m3"] .card-resolved').innerText()) === 'Allowed' &&
    (await page.locator('[data-message-id="m3"] .card-buttons').count()) === 0);
  await host({ command: 'busy', value: false });

  // ---- sending ----
  await page.fill('#input', 'what does warmup do?');
  await page.click('#send');
  const sent = await last('sendMessage');
  check('Send posts the text', sent && sent.text === 'what does warmup do?');
  check('Send is disabled while working', await page.locator('#send').isDisabled());
  check('working indicator is shown', (await page.locator('.busy').count()) === 1);
  await host({ command: 'appendMessage', sessionId: 's1', message: { messageId: 'm5', role: 'assistant',
    content: '## Warmup\n\nWarmup **ramps** the `learning_rate`.\n\n- item one\n- item two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\nprint("hi")\n```' } });
  await host({ command: 'busy', value: false });
  check('markdown renders (heading, list, table, code)', (await page.locator('.msg:last-of-type h3').count()) === 1 &&
    (await page.locator('.msg:last-of-type li').count()) === 2 && (await page.locator('.msg:last-of-type td').count()) === 2 &&
    (await page.locator('.msg:last-of-type pre code').innerText()).includes('print("hi")'));
  check('Send is enabled again', await page.locator('#send').isEnabled());

  // ---- untrusted text is never markup ----
  await host({ command: 'appendMessage', message: { messageId: 'm6', role: 'assistant', content: '<img src=x onerror="window.__xss=1"><script>window.__xss=1</script>' } });
  await page.waitForTimeout(100);
  check('HTML in messages is shown as text, not executed', !(await page.evaluate(() => window.__xss)) &&
    (await page.locator('[data-message-id="m6"] img').count()) === 0);

  // ---- command output ----
  await host({ command: 'appendMessage', message: { messageId: 'm7', role: 'assistant', content: 'Ran `git status` - exit code 0.',
    blocks: [{ type: 'command', command: 'git status', tier: 'safe', exitCode: 0, timedOut: false, output: 'On branch main', truncated: false, durationMs: 5 }] } });
  check('command output block renders', (await page.locator('[data-message-id="m7"] .cmd-out').innerText()).includes('On branch main'));

  // ---- local (extension-made) card ----
  await host({ command: 'localCard', actionId: 'dep-1', text: 'Install dependencies?', buttons: [{ label: 'Install', action: 'install' }, { label: 'Cancel', action: 'cancel' }] });
  const card = page.locator('.msg').last();
  check('local card has Copy Message too', (await card.locator('.copy-btn').count()) === 1);
  await card.getByRole('button', { name: 'Install' }).click();
  const local = await last('actionButtonClicked');
  check('local card reports the chosen action', local && local.actionId === 'dep-1' && local.action === 'install');
  check('local card shows what was chosen', (await card.locator('.card-resolved').innerText()) === 'Install');

  // ---- command buttons from stored messages go through the extension ----
  await host({ command: 'appendMessage', message: { messageId: 'm8', role: 'assistant', content: 'Use the button.',
    blocks: [{ type: 'commandButtons', buttons: [{ label: 'Analyze Project', command: 'llmTrainingAgent.analyzeProject' }] }] } });
  await page.getByRole('button', { name: 'Analyze Project' }).click();
  check('command buttons post the command id', (await last('runCommand'))?.id === 'llmTrainingAgent.analyzeProject');

  // ---- navigation ----
  await page.click('#back');
  check('back control returns to the list', !!(await last('backToList')));
  await host({ command: 'showList', hasProject: true, sessions: [{ sessionId: 's2', title: 'Project analysis', updatedAt: '2026-10-27T09:30:00Z' }] });
  check('list is shown again', await page.locator('#list-screen').isVisible() && !(await page.locator('#chat-screen').isVisible()));
  await page.locator('.session-delete').click();
  check('delete asks for confirmation', (await page.locator('.confirm-text').innerText()).includes('Project analysis'));
  check('nothing is deleted before confirming', !(await posted()).some((m) => m.command === 'deleteSession'));
  await page.getByRole('button', { name: 'Delete', exact: true }).click();
  check('confirming deletes', (await last('deleteSession'))?.sessionId === 's2');
  await host({ command: 'showList', hasProject: true, sessions: [] });
  check('empty list explains what to do', (await page.locator('#sessions .empty').innerText()).includes('New chat'));
  await page.click('#new-chat');
  check('New chat button asks for a new chat', !!(await last('newChat')));
  await host({ command: 'showChat', sessionId: null, title: 'New chat', messages: [] });
  check('new chat is empty and ready to type', (await page.locator('.msg').count()) === 0 && (await page.locator('#input').isVisible()));

  // ---- narrow sidebar ----
  await page.setViewportSize({ width: 220, height: 600 });
  await host({ command: 'showChat', sessionId: 's1', title: 'A rather long chat title that must be clipped nicely', messages: [
    { messageId: 'n1', role: 'user', content: 'change epochs from 3 to 5 please and thanks very much' },
    { messageId: 'n2', role: 'assistant', content: 'Done.', blocks: [DIFF] } ] });
  const overflow = await page.evaluate(() => ({ page: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    top: document.querySelector('#chat-screen .topbar').scrollWidth - document.querySelector('#chat-screen .topbar').clientWidth }));
  check('no horizontal page scroll at 220px', overflow.page <= 0, JSON.stringify(overflow));
  check('header fits at 220px', overflow.top <= 0, JSON.stringify(overflow));
  check('code blocks scroll inside themselves', await page.evaluate(() => getComputedStyle(document.querySelector('.hunk-code')).overflowX === 'auto'));
  check('composer is visible at 220px', await page.locator('#input').isVisible() && await page.locator('#send').isVisible());
  await page.screenshot({ path: path.join(shotDir, '3-narrow.png') });
  await page.setViewportSize({ width: 220, height: 600 });
  await host({ command: 'showList', hasProject: true, sessions: [{ sessionId: 's1', title: 'A rather long chat title that must be clipped nicely', updatedAt: '2026-10-28T14:05:00Z' }] });
  await page.screenshot({ path: path.join(shotDir, '4-narrow-list.png') });

  check('no script errors', errors.length === 0, errors.join(' | '));
  await browser.close();
  fs.rmSync(pagePath, { force: true });
  fs.rmSync(themePath, { force: true });
  if (failures.length) { console.error(`\n${failures.length} check(s) failed`); process.exit(1); }
  console.log(`\nAll checks passed. Screenshots: ${shotDir}`);
})().catch((e) => { fs.rmSync(pagePath, { force: true }); console.error(e); process.exit(1); });
