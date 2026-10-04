import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Every suite under tests/suite runs here. All of them are pure Node
    // (no `import * as vscode from 'vscode'`), so no Electron host is needed.
    include: ['tests/suite/**/*.test.ts'],
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
});