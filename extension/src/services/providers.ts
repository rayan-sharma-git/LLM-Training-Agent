/**
 * The AI providers the extension and backend both know about.
 *
 * Kept in one dependency-free module so the settings view, the backend sync and
 * the tests cannot drift apart. The backend's `list_providers()` is the other
 * half of this contract.
 */
export interface ProviderDescriptor {
  id: string;
  label: string;
  /** False for providers that run locally and need no credentials. */
  requiresKey: boolean;
}

export const PROVIDERS: readonly ProviderDescriptor[] = [
  { id: 'ollama', label: 'Ollama — Local (no API key)', requiresKey: false },
  { id: 'openai', label: 'OpenAI', requiresKey: true },
  { id: 'anthropic', label: 'Anthropic', requiresKey: true },
  { id: 'gemini', label: 'Google Gemini', requiresKey: true },
  { id: 'deepseek', label: 'DeepSeek', requiresKey: true },
  { id: 'cohere', label: 'Cohere', requiresKey: true },
  { id: 'openrouter', label: 'OpenRouter', requiresKey: true },
  { id: 'openai_compatible', label: 'OpenAI-compatible (custom endpoint)', requiresKey: true },
];

export const PROVIDER_IDS: readonly string[] = PROVIDERS.map((provider) => provider.id);

export function isKnownProvider(value: unknown): value is string {
  return typeof value === 'string' && PROVIDER_IDS.includes(value);
}

/** Providers whose endpoint URL is user-configurable. */
export const BASE_URL_PROVIDERS: readonly string[] = ['ollama', 'openai_compatible'];

export const DEFAULT_BASE_URLS: Readonly<Record<string, string>> = {
  ollama: 'http://localhost:11434',
  openai_compatible: 'http://localhost:1234/v1',
};
