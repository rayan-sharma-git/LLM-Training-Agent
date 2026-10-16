/**
 * Promote the freshly built VSIX to the repository root.
 *
 * `vsce package` always writes the file into `extension/` (the build
 * directory). The authoritative distributable that users download lives at
 * the repository root (`llm-training-agent-<version>.vsix`) and is the only
 * VSIX tracked by git (`extension/*.vsix` is git-ignored).
 *
 * This script runs automatically as `postpackage` so the root copy can never
 * silently remain stale after source changes: every `npm run package` ends
 * with the root VSIX rebuilt from the latest verified source, and no VSIX
 * left behind in `extension/`.
 *
 * No third-party dependencies: plain Node.js only.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const EXTENSION_DIR = path.resolve(__dirname, '..');
const ROOT_DIR = path.resolve(EXTENSION_DIR, '..');

function findBuiltVsix() {
  const candidates = fs
    .readdirSync(EXTENSION_DIR)
    .filter((name) => name.endsWith('.vsix'))
    .map((name) => path.join(EXTENSION_DIR, name));

  if (candidates.length === 0) {
    console.error(
      `promote-vsix: no .vsix found in ${EXTENSION_DIR} — did 'vsce package' succeed?`
    );
    process.exit(1);
  }
  if (candidates.length > 1) {
    console.error(
      `promote-vsix: multiple .vsix files found: ${candidates.join(', ')}`
    );
    process.exit(1);
  }
  return candidates[0];
}

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function main() {
  const built = findBuiltVsix();
  const fileName = path.basename(built);
  const dest = path.join(ROOT_DIR, fileName);

  fs.copyFileSync(built, dest);
  fs.rmSync(built);

  const hash = sha256(dest);
  const size = fs.statSync(dest).size;
  console.log(
    `promote-vsix: promoted ${fileName} to repository root (${size} bytes, sha256 ${hash}); removed build copy in extension/`
  );
}

main();
