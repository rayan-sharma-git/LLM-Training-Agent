/**
 * Copy the Python backend into the extension folder so `vsce package`
 * produces a self-contained .vsix.
 *
 * This MUST run before packaging. It is wired into both `vscode:prepublish`
 * and `prepackage` so the bundled backend can never drift from the source
 * backend (a stale copy is what previously shipped a non-importable backend).
 *
 * The extension resolves this folder at runtime from `context.extensionUri`,
 * i.e. from the VS Code installation directory — never from the user's project.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const EXTENSION_DIR = path.resolve(__dirname, '..');
const BACKEND_SRC = path.resolve(EXTENSION_DIR, '..', 'backend');
const BACKEND_DEST = path.join(EXTENSION_DIR, 'backend');

/** Directories that must never reach the VSIX. */
const EXCLUDED_DIRS = new Set([
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.ruff_cache',
  '.git',
  'tests',
  '.venv',
  'venv',
  'venv310',
  'node_modules',
]);

/** Files that must never reach the VSIX (secrets, local data, artifacts). */
const EXCLUDED_FILES = new Set([
  '.env',
  '.env.local',
  'llm_training_agent.db',
  'llm_training_agent.db-wal',
  'llm_training_agent.db-shm',
]);

const EXCLUDED_EXTENSIONS = new Set(['.pyc', '.pyo', '.log', '.db', '.db-wal', '.db-shm']);

function isExcludedDir(name) {
  return EXCLUDED_DIRS.has(name);
}

function isExcludedFile(name) {
  if (EXCLUDED_FILES.has(name)) {
    return true;
  }
  // Any dotenv-style file, at any depth.
  if (name.startsWith('.env')) {
    return true;
  }
  return EXCLUDED_EXTENSIONS.has(path.extname(name));
}

function copyTree(srcDir, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  let copied = 0;

  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const src = path.join(srcDir, entry.name);
    const dest = path.join(destDir, entry.name);

    if (entry.isDirectory()) {
      if (isExcludedDir(entry.name)) {
        continue;
      }
      copied += copyTree(src, dest);
      continue;
    }

    if (!entry.isFile() || isExcludedFile(entry.name)) {
      continue;
    }

    fs.copyFileSync(src, dest);
    copied += 1;
  }

  return copied;
}

function main() {
  if (!fs.existsSync(BACKEND_SRC)) {
    console.error(`Backend source not found: ${BACKEND_SRC}`);
    process.exit(1);
  }

  if (fs.existsSync(BACKEND_DEST)) {
    fs.rmSync(BACKEND_DEST, { recursive: true, force: true });
  }

  const count = copyTree(BACKEND_SRC, BACKEND_DEST);
  console.log(
    `Bundled backend: ${count} files copied\n  from ${BACKEND_SRC}\n  to   ${BACKEND_DEST}`
  );
}

main();
