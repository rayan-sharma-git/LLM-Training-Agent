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

// Chat wiring: the page is an external script/stylesheet in media/, referenced from the provider.
check('chatWebviewProvider.js', [
  "chat.js",
  "chat.css",
  "id=\"send\"",
  "id=\"input\"",
  "id=\"new-chat\"",
  "id=\"back\"",
]);
for (const file of ['chat.js', 'chat.css']) {
  const full = path.join(root, 'media', file);
  if (!fs.existsSync(full)) {
    console.error(`MISSING media/${file}`);
    process.exitCode = 1;
  } else {
    console.log(`ok media/${file} present`);
  }
}
const chatJs = fs.readFileSync(path.join(root, 'media', 'chat.js'), 'utf8');
for (const needle of ["$('send')", "$('input')", "'Copy Message'", "'Removed'", "'Added'"]) {
  if (!chatJs.includes(needle)) {
    console.error(`MISSING in media/chat.js: ${needle}`);
    process.exitCode = 1;
  } else {
    console.log(`ok media/chat.js: ${needle}`);
  }
}

// Activation retry behavior.
const ext = fs.readFileSync(path.join(root, 'out', 'extension.js'), 'utf8');
if (!ext.includes('startupPromise = null')) {
  console.error('MISSING in extension.js: startupPromise = null');
  process.exitCode = 1;
} else {
  console.log('ok extension.js: startupPromise = null');
}
