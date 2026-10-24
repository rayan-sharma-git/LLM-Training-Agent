/**
 * Push the user's saved AI configuration into the running backend.
 *
 * The backend deliberately keeps API keys in memory only (nothing secret is ever
 * written to disk), so it forgets everything whenever it restarts. VS Code's
 * SecretStorage and settings are therefore the source of truth, and this runs
 * every time a backend process is new to us: without it, a key that was saved
 * yesterday is silently missing today and chat answers "no provider configured".
 */
import { BASE_URL_PROVIDERS, DEFAULT_BASE_URLS, PROVIDER_IDS } from './providers';

export interface BackendSyncSettings {
  provider: string;
  model: string;
  ollamaBaseUrl: string;
  openaiCompatibleBaseUrl: string;
}

export interface BackendSyncDeps {
  getApiKey(provider: string): Promise<string | undefined>;
  updateConfig(config: Record<string, unknown>): Promise<unknown>;
  selectProvider(provider: string, model: string): Promise<unknown>;
  /** Receives secret-free diagnostics only. */
  log(message: string): void;
}

export interface BackendSyncResult {
  ok: boolean;
  /** Secret-free description of the first failure, when `ok` is false. */
  error?: string;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function baseUrlFor(provider: string, settings: BackendSyncSettings): string {
  const configured =
    provider === 'ollama'
      ? settings.ollamaBaseUrl
      : provider === 'openai_compatible'
        ? settings.openaiCompatibleBaseUrl
        : '';
  return (configured || '').trim() || DEFAULT_BASE_URLS[provider] || '';
}

export async function syncBackendConfig(
  settings: BackendSyncSettings,
  deps: BackendSyncDeps
): Promise<BackendSyncResult> {
  let firstError: string | undefined;
  const fail = (what: string, error: unknown) => {
    const message = `${what}: ${describe(error)}`;
    deps.log(`Backend sync problem — ${message}`);
    firstError = firstError ?? message;
  };

  // Keys first. `updateConfig` also selects the provider it is given, so the
  // real provider/model are applied last and win.
  for (const provider of PROVIDER_IDS) {
    let key: string | undefined;
    try {
      key = await deps.getApiKey(provider);
    } catch (error) {
      fail(`could not read the saved key for ${provider}`, error);
      continue;
    }
    if (!key) {
      continue;
    }
    try {
      await deps.updateConfig({ provider, apiKey: key });
      deps.log(`Backend sync: restored the saved API key for ${provider}.`);
    } catch (error) {
      fail(`could not send the key for ${provider}`, error);
    }
  }

  for (const provider of BASE_URL_PROVIDERS) {
    const baseUrl = baseUrlFor(provider, settings);
    if (!baseUrl) {
      continue;
    }
    try {
      await deps.updateConfig({ provider, baseUrl });
    } catch (error) {
      fail(`could not apply the ${provider} URL`, error);
    }
  }

  try {
    await deps.selectProvider(settings.provider, settings.model);
    deps.log(`Backend sync: active provider "${settings.provider}", model "${settings.model}".`);
  } catch (error) {
    fail(`could not select provider "${settings.provider}"`, error);
  }

  return firstError ? { ok: false, error: firstError } : { ok: true };
}
