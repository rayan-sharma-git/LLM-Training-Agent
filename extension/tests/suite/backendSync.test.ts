import { describe, it, expect, vi } from 'vitest';
import { syncBackendConfig, baseUrlFor, BackendSyncSettings } from '../../src/services/backendSync';

const settings: BackendSyncSettings = {
  provider: 'openai',
  model: 'gpt-4o-mini',
  ollamaBaseUrl: 'http://localhost:11434',
  openaiCompatibleBaseUrl: '',
};

function makeDeps(keys: Record<string, string>) {
  const calls: Array<[string, unknown]> = [];
  return {
    calls,
    deps: {
      getApiKey: vi.fn(async (provider: string) => keys[provider]),
      updateConfig: vi.fn(async (config: Record<string, unknown>) => {
        calls.push(['updateConfig', config]);
      }),
      selectProvider: vi.fn(async (provider: string, model: string) => {
        calls.push(['selectProvider', { provider, model }]);
      }),
      log: vi.fn(),
    },
  };
}

describe('syncBackendConfig', () => {
  it('re-sends every saved key, then applies the chosen provider and model last', async () => {
    const { deps, calls } = makeDeps({ anthropic: 'sk-ant-1', openai: 'sk-oa-2' });
    const result = await syncBackendConfig(settings, deps);
    expect(result).toEqual({ ok: true });
    const keyCalls = calls.filter(([, payload]) => (payload as { apiKey?: string }).apiKey);
    expect(keyCalls.length).toBe(2);
    expect(calls[calls.length - 1]).toEqual(['selectProvider', { provider: 'openai', model: 'gpt-4o-mini' }]);
  });

  it('never writes a key into a log line', async () => {
    const { deps } = makeDeps({ openai: 'sk-SECRET-VALUE' });
    await syncBackendConfig(settings, deps);
    const logged = deps.log.mock.calls.map((c: unknown[]) => String(c[0])).join('\n');
    expect(logged).not.toContain('sk-SECRET-VALUE');
  });

  it('reports a failure without secrets and still applies the provider', async () => {
    const { deps, calls } = makeDeps({ openai: 'sk-SECRET-VALUE' });
    deps.updateConfig.mockImplementation(async (config: Record<string, unknown>) => {
      if (config.apiKey) {
        throw new Error('connect ECONNREFUSED');
      }
    });
    const result = await syncBackendConfig(settings, deps);
    expect(result.ok).toBe(false);
    expect(result.error).toContain('ECONNREFUSED');
    expect(result.error).not.toContain('sk-SECRET-VALUE');
    expect(calls.some(([name]) => name === 'selectProvider')).toBe(true);
  });

  it('skips providers with no saved key', async () => {
    const { deps } = makeDeps({});
    await syncBackendConfig({ ...settings, provider: 'ollama', model: 'llama3.2' }, deps);
    expect(deps.updateConfig.mock.calls.every((c: unknown[]) => !(c[0] as { apiKey?: string }).apiKey)).toBe(true);
  });
});

describe('baseUrlFor', () => {
  it('uses the configured URL, falling back to the provider default', () => {
    expect(baseUrlFor('ollama', settings)).toBe('http://localhost:11434');
    expect(baseUrlFor('openai_compatible', settings)).toBe('http://localhost:1234/v1');
    expect(baseUrlFor('openai', settings)).toBe('');
  });
});
