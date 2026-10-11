const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const compiledDir = path.join(root, 'out', 'views');

function check(file, needles) {
  const body = fs.readFileSync(path.join(compiledDir, file), 'utf8');
  for (const needle of needles) {
    if (!body.includes(needle)) {
      console.error(`MISSING in ${file}: ${needle}`);
      process.exitCode = 1;
    } else {
      console.log(`ok ${file}: ${needle}`);
    }
  }
}

// Settings wiring: immediate init + backend startup before API calls.
check('settingsWebviewProvider.js', [
  'initSettingsUi()',
  '_ensureBackendReady',
  'selectProvider',
  "getElementById('saveBtn').addEventListener",
  "getElementById('testBtn').addEventListener",
  "getElementById('refreshModelsBtn').addEventListener",
]);
const settings = fs.readFileSync(path.join(compiledDir, 'settingsWebviewProvider.js'), 'utf8');
if (/addEventListener\(\s*['"]DOMContentLoaded['"]/.test(settings)) {
  console.error('MISSING: settings still defers wiring on DOMContentLoaded');
  process.exitCode = 1;
} else {
  console.log('ok settingsWebviewProvider.js: no DOMContentLoaded deferral');
}

// Chat wiring: direct bindings for send.
check('chatWebviewProvider.js', [
  "getElementById('send')",
  "getElementById('input')",
]);

// Activation retry behavior.
const ext = fs.readFileSync(path.join(root, 'out', 'extension.js'), 'utf8');
if (!ext.includes('startupPromise = null')) {
  console.error('MISSING in extension.js: startupPromise = null');
  process.exitCode = 1;
} else {
  console.log('ok extension.js: startupPromise = null');
}
