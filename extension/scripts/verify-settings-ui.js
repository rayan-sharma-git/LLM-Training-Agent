/**
 * Browser check of the Settings view: Permissions toggles and Training Hardware form.
 * Needs Playwright (npm i --no-save playwright) and a compiled extension (npm run compile).
 * Usage: node scripts/verify-settings-ui.js [screenshotDir]
 */
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { pathToFileURL } = require('url');

const root = path.resolve(__dirname, '..');
const shotDir = process.argv[2] || path.join(os.tmpdir(), 'settings-ui-shots');
fs.mkdirSync(shotDir, { recursive: true });

const realLoad = Module._load;
Module._load = function (request, ...rest) {
  if (request === 'vscode') return { ConfigurationTarget: { Global: 1 }, workspace: {}, window: {}, commands: {}, Uri: {} };
  return realLoad.call(this, request, ...rest);
};
const { SettingsWebviewProvider } = require(path.join(root, 'out', 'views', 'settingsWebviewProvider.js'));
Module._load = realLoad;

const failures = [];
const check = (name, ok, detail) => { if (ok) console.log(`ok   ${name}`); else { console.error(`FAIL ${name}${detail ? ' - ' + detail : ''}`); failures.push(name); } };

(async () => {
  const { chromium } = require('playwright');
  const browser = await chromium.launch();
  for (const trusted of [true, false]) {
    const provider = new SettingsWebviewProvider({}, {}, {}, undefined);
    const html = provider._getHtml({ cspSource: 'file:' }, 'ollama', 'llama3.2', {},
      { readFiles: true, writeFiles: false, safeCommands: false, unrestrictedCommands: false },
      { mode: 'cloud', gpuName: '', hourlyRateUsd: 2.5, provider: 'RunPod', gpuCount: 1 }, trusted);
    const file = path.join(os.tmpdir(), `settings-ui-${trusted}.html`);
    fs.writeFileSync(file, html);
    const page = await browser.newPage({ viewport: { width: 300, height: 1100 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.addInitScript(() => {
      window.__posted = [];
      window.acquireVsCodeApi = () => ({ postMessage: (m) => window.__posted.push(m) });
    });
    await page.goto(pathToFileURL(file).toString());
    const posted = () => page.evaluate(() => window.__posted.slice());
    const tag = trusted ? 'trusted' : 'untrusted';

    check(`[${tag}] four permission controls exist`, (await page.locator('input[id^="perm-"]').count()) === 4);
    check(`[${tag}] defaults: read on, others off`, await page.evaluate(() => ['readFiles', 'writeFiles', 'safeCommands', 'unrestrictedCommands']
      .map((k) => document.getElementById('perm-' + k).checked).join() === 'true,false,false,false'));
    if (trusted) {
      await page.check('#perm-writeFiles');
      const m = (await posted()).filter((x) => x.command === 'setPermission').pop();
      check('toggling Write posts only that key', m && m.key === 'writeFiles' && m.value === true, JSON.stringify(m));
      check('other permissions stay independent', !(await page.isChecked('#perm-safeCommands')) && !(await page.isChecked('#perm-unrestrictedCommands')));
      await page.check('#perm-safeCommands');
      check('safe commands toggles on its own', (await posted()).filter((x) => x.key === 'safeCommands').pop().value === true);

      const before = (await posted()).length;
      await page.click('#perm-unrestrictedCommands');
      check('unrestricted needs a second, explicit confirmation', (await posted()).length === before && await page.locator('#unrestrictedConfirm').isVisible());
      check('unrestricted stays off until confirmed', !(await page.isChecked('#perm-unrestrictedCommands')));
      await page.click('#unrestrictedNo');
      check('cancelling changes nothing', (await posted()).length === before);
      await page.click('#perm-unrestrictedCommands');
      await page.click('#unrestrictedYes');
      check('confirming enables it', (await posted()).pop().key === 'unrestrictedCommands' && await page.isChecked('#perm-unrestrictedCommands'));
      await page.click('#perm-unrestrictedCommands');
      check('turning it off needs no confirmation', (await posted()).pop().value === false);
      await page.screenshot({ path: path.join(shotDir, 'permissions.png'), fullPage: true });

      check('cloud mode shows rate fields', await page.locator('#hwRate').isVisible());
      await page.selectOption('#hwMode', 'local');
      check('local mode hides rate fields (no rental cost)', !(await page.locator('#hwRate').isVisible()));
      await page.selectOption('#hwMode', 'cloud');
      await page.fill('#hwRate', '3.2');
      await page.fill('#hwGpu', 'A100');
      await page.click('#saveHardwareBtn');
      const hw = (await posted()).filter((x) => x.command === 'saveHardware').pop();
      check('hardware form posts what was typed', hw && hw.hardware.mode === 'cloud' && hw.hardware.hourlyRateUsd === 3.2 && hw.hardware.gpuName === 'A100', JSON.stringify(hw));
      await page.fill('#hwRate', '');
      await page.click('#saveHardwareBtn');
      check('an empty rate is sent as null, never a guess', (await posted()).filter((x) => x.command === 'saveHardware').pop().hardware.hourlyRateUsd === null);
    } else {
      check('untrusted: warning shown', await page.locator('#untrustedWarning').isVisible());
      check('untrusted: write and commands are disabled', await page.locator('#perm-writeFiles').isDisabled() &&
        await page.locator('#perm-safeCommands').isDisabled() && await page.locator('#perm-unrestrictedCommands').isDisabled());
      check('untrusted: read stays adjustable', await page.locator('#perm-readFiles').isEnabled());
    }
    check(`[${tag}] no horizontal scroll at 300px`, await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth));
    check(`[${tag}] no script errors`, errors.length === 0, errors.join(' | '));
    await page.close();
    fs.rmSync(file, { force: true });
  }
  await browser.close();
  if (failures.length) { console.error(`\n${failures.length} check(s) failed`); process.exit(1); }
  console.log(`\nAll checks passed. Screenshots: ${shotDir}`);
})().catch((e) => { console.error(e); process.exit(1); });
