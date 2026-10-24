/**
 * Static wiring checks for the two webviews. Run with `npm run verify:wiring`
 * (compiles first). It checks what actually broke before, not strings:
 *
 *  - media/chat.js and media/settings.js must PARSE. A script that fails to
 *    parse leaves every button dead and the console empty.
 *  - The HTML the extension really builds must load exactly one external,
 *    nonce-tagged script under a CSP without 'unsafe-inline' scripts.
 *  - The providers must load those scripts, and extension.ts must restore the
 *    saved configuration into every new backend process.
 *
 * It does not click anything; the clicking is covered by running the pages in
 * a browser engine (see the report that accompanies a release).
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
let failed = false;
const ok = (message) => console.log(`ok   ${message}`);
const fail = (message) => {
  console.error(`FAIL ${message}`);
  failed = true;
};
const check = (condition, message) => (condition ? ok(message) : fail(message));

for (const file of ['chat.js', 'settings.js']) {
  const source = fs.readFileSync(path.join(root, 'media', file), 'utf8');
  try {
    new vm.Script(source, { filename: file });
    ok(`media/${file} parses`);
  } catch (error) {
    fail(`media/${file} does not parse: ${error.message}`);
  }
}

const html = require(path.join(root, 'out', 'views', 'webviewHtml.js'));
const assets = { cspSource: 'vscode-webview://x', nonce: 'abc', scriptUri: 'vscode-webview://x/media/a.js' };
const pages = {
  chat: html.buildChatHtml(assets),
  settings: html.buildSettingsHtml({ ...assets, provider: 'ollama', model: 'm', keyStates: {}, baseUrls: {} }),
};
for (const [name, page] of Object.entries(pages)) {
  const scripts = page.match(/<script\b[^>]*>/g) || [];
  check(scripts.length === 1 && /nonce="abc"/.test(scripts[0]) && /src="/.test(scripts[0]), `${name} html loads one external nonce script`);
  check(!/<script[^>]*>[^<]+<\/script>/.test(page), `${name} html has no inline script`);
  check(!/script-src[^;"]*unsafe-inline/.test(page), `${name} CSP has no script 'unsafe-inline'`);
}

const read = (...parts) => fs.readFileSync(path.join(root, 'out', ...parts), 'utf8');
check(read('views', 'chatWebviewProvider.js').includes('chat.js'), 'chat provider loads media/chat.js');
check(read('views', 'settingsWebviewProvider.js').includes('settings.js'), 'settings provider loads media/settings.js');
check(/syncBackendConfig/.test(read('extension.js')), 'extension restores the saved configuration into new backends');
check(!/startupPromise/.test(read('extension.js')), 'extension does not cache a past backend start forever');

process.exit(failed ? 1 : 0);
