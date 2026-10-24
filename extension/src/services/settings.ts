import * as vscode from 'vscode';
import { DEFAULT_BASE_URLS } from './providers';

export interface Settings {
  backendUrl: string;
  provider: string;
  model: string;
  /**
   * User-level path to the Python interpreter used for the bundled backend.
   * Empty means "use the interpreter bundled with the extension, or `python`
   * from PATH". It is a machine setting, never a project setting.
   */
  pythonPath: string;
  /** Ollama server URL (user-level). */
  ollamaBaseUrl: string;
  /** Base URL of the OpenAI-compatible endpoint (user-level). */
  openaiCompatibleBaseUrl: string;
}

/**
 * Reads and writes the extension's configuration.
 *
 * All settings are declared with a user-level (or machine-level) scope in
 * `package.json`, so installing the extension once is enough: opening a new
 * project never requires reconfiguring or reinstalling anything.
 */
export class SettingsManager {
  private readonly context: vscode.ExtensionContext;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
  }

  /** Read the effective settings for the current window. */
  getSettings(): Settings {
    const config = vscode.workspace.getConfiguration('llmTrainingAgent');
    return {
      backendUrl: config.get<string>('backendUrl', 'http://127.0.0.1:8000'),
      provider: config.get<string>('provider', 'ollama'),
      model: config.get<string>('model', 'llama3.2'),
      pythonPath: config.get<string>('pythonPath', ''),
      ollamaBaseUrl: config.get<string>('ollamaBaseUrl', DEFAULT_BASE_URLS.ollama),
      openaiCompatibleBaseUrl: config.get<string>(
        'openaiCompatibleBaseUrl',
        DEFAULT_BASE_URLS.openai_compatible
      ),
    };
  }

  /**
   * Persist settings at the user level so they apply to every project.
   * Writing to `ConfigurationTarget.Global` guarantees the values are never
   * written into a `.vscode/settings.json` inside a user project.
   */
  async updateSettings(settings: Partial<Settings>): Promise<void> {
    const config = vscode.workspace.getConfiguration('llmTrainingAgent');
    for (const [key, value] of Object.entries(settings)) {
      await config.update(key as never, value, vscode.ConfigurationTarget.Global);
    }
  }
}
