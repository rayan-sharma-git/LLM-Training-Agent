/**
 * Packaging and manifest guards.
 *
 * These tests encode the "install once from a .vsix" contract at the source
 * level: the manifest must activate normally, settings must be user-level, and
 * the packaging rules must keep runtime dependencies while excluding
 * development-only and secret files.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

const EXTENSION_DIR = path.resolve(__dirname, '..', '..');
const read = (relative: string) => fs.readFileSync(path.join(EXTENSION_DIR, relative), 'utf8');
const manifest = JSON.parse(read('package.json'));

describe('extension manifest', () => {
  it('is a normal VS Code extension manifest', () => {
    expect(manifest.name).toBe('llm-training-agent');
    expect(manifest.publisher).toBeTruthy();
    expect(manifest.engines.vscode).toBeTruthy();
    expect(manifest.main).toBe('./out/extension.js');
  });

  it('declares every contributed command', () => {
    const declared = manifest.contributes.commands.map((c: { command: string }) => c.command);
    const expected = [
      'llmTrainingAgent.analyzeProject',
      'llmTrainingAgent.analyzeDataset',
      'llmTrainingAgent.openChat',
      'llmTrainingAgent.viewReport',
      'llmTrainingAgent.configureProvider',
      'llmTrainingAgent.viewChanges',
    ];
    for (const command of expected) {
      expect(declared).toContain(command);
    }
  });

  it('activates on its own commands and views, not on a project structure', () => {
    const events: string[] = manifest.activationEvents;
    // Lazy activation is preserved: nothing forces startup work.
    expect(events).not.toContain('onStartupFinished');
    // No project-shape activation that would require the agent inside a repo.
    expect(events.some((e: string) => e.startsWith('workspaceContains'))).toBe(false);
    expect(events.some((e: string) => e.startsWith('workspaceFileContains'))).toBe(false);
    expect(events.some((e: string) => e.startsWith('onLanguage:'))).toBe(false);
    // Every command has an activation event, so the palette always works.
    for (const command of manifest.contributes.commands) {
      expect(events).toContain(`onCommand:${command.command}`);
    }
  });

  it('keeps commands visible in the palette regardless of workspace state', () => {
    // The commands report the no-workspace state themselves, so they must not
    // be hidden by a `when` clause.
    const gated = manifest.contributes.menus.commandPalette.filter(
      (entry: { when?: string }) => entry.when
    );
    expect(gated).toEqual([]);
  });

  it('scopes LLM provider settings to the user, not to a project', () => {
    const props = manifest.contributes.configuration.properties;
    expect(props['llmTrainingAgent.provider'].scope).toBe('application');
    expect(props['llmTrainingAgent.model'].scope).toBe('application');
  });

  it('exposes a user-level python interpreter override', () => {
    const prop = manifest.contributes.configuration.properties['llmTrainingAgent.pythonPath'];
    expect(prop).toBeDefined();
    expect(prop.type).toBe('string');
    expect(prop.default).toBe('');
  });

  it('keeps the existing LLM provider architecture intact', () => {
    const providers =
      manifest.contributes.configuration.properties['llmTrainingAgent.provider'].enum;
    expect(providers).toContain('ollama');
    expect(providers).toEqual(
      expect.arrayContaining(['openai', 'anthropic', 'gemini', 'openrouter'])
    );
  });
});

describe('packaging rules (.vscodeignore)', () => {
  const ignore = read('.vscodeignore');
  const lines = ignore.split(/\r?\n/).map((l) => l.trim());

  it('does not exclude node_modules wholesale', () => {
    // Excluding node_modules would strip axios and break activation, because
    // vsce only ships the production dependency tree.
    expect(lines).not.toContain('node_modules');
    expect(lines).not.toContain('node_modules/**');
  });

  it('excludes development-only files', () => {
    for (const pattern of ['src/**', 'tests/**', 'scripts/**', '.vscode/**', '*.vsix']) {
      expect(lines).toContain(pattern);
    }
  });

  it('excludes Python caches, logs, databases and secrets', () => {
    for (const pattern of [
      '**/__pycache__/**',
      '**/*.pyc',
      '**/.mypy_cache/**',
      '**/.pytest_cache/**',
      '**/.ruff_cache/**',
      '**/tests/**',
      '**/.env',
      '**/.env.*',
    ]) {
      expect(lines).toContain(pattern);
    }
  });

  it('does not exclude the bundled backend or resources', () => {
    expect(lines).not.toContain('backend/**');
    expect(lines).not.toContain('resources/**');
    expect(lines).not.toContain('out/**');
  });
});

describe('permissions, hardware and icon in the manifest', () => {
  const props = manifest.contributes.configuration.properties;

  it('declares the four independent permissions with safe defaults', () => {
    expect(props['llmTrainingAgent.permissions.readFiles'].default).toBe(true);
    for (const key of ['writeFiles', 'safeCommands', 'unrestrictedCommands']) {
      expect(props[`llmTrainingAgent.permissions.${key}`].default).toBe(false);
    }
  });

  it('keeps permissions application-scoped so a repository cannot grant itself access', () => {
    for (const key of Object.keys(props).filter((k) => k.startsWith('llmTrainingAgent.permissions.'))) {
      expect(props[key].scope).toBe('application');
    }
  });

  it('has a real marketplace icon and a monochrome activity-bar icon', () => {
    expect(manifest.icon).toBe('resources/icon.png');
    expect(fs.existsSync(path.join(EXTENSION_DIR, manifest.icon))).toBe(true);
    const svg = read('resources/icon.svg');
    expect(svg).toContain('currentColor');
    expect(svg).not.toContain('#007acc');
  });

  it('ships the chat page files (not excluded by .vscodeignore)', () => {
    const ignore = read('.vscodeignore');
    expect(ignore).not.toMatch(/^media\b/m);
    expect(fs.existsSync(path.join(EXTENSION_DIR, 'media', 'chat.js'))).toBe(true);
    expect(fs.existsSync(path.join(EXTENSION_DIR, 'media', 'chat.css'))).toBe(true);
  });
});

