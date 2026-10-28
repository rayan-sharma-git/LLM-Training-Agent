import type { PermissionPolicy, HardwareSettings } from './apiClient';

/**
 * Pure permission/hardware helpers (no `vscode` import, so they are unit-testable in Node).
 */
export const PERMISSION_KEYS = ['readFiles', 'writeFiles', 'safeCommands', 'unrestrictedCommands'] as const;
export type PermissionKey = (typeof PERMISSION_KEYS)[number];

export const PERMISSION_DEFAULTS: PermissionPolicy = {
  readFiles: true,
  writeFiles: false,
  safeCommands: false,
  unrestrictedCommands: false,
};

/** Only a literal `true` enables a permission; anything else falls back to the safe default. */
export function normalizePermissions(raw: Partial<Record<string, unknown>> | undefined): PermissionPolicy {
  const out: PermissionPolicy = { ...PERMISSION_DEFAULTS };
  for (const key of PERMISSION_KEYS) {
    const value = raw?.[key];
    if (typeof value === 'boolean') {
      out[key] = value;
    }
  }
  return out;
}

/**
 * In an untrusted workspace nothing may modify files or run commands, no matter what the
 * settings say.
 */
export function effectivePermissions(policy: PermissionPolicy, workspaceTrusted: boolean): PermissionPolicy {
  if (workspaceTrusted) {
    return policy;
  }
  return { readFiles: policy.readFiles, writeFiles: false, safeCommands: false, unrestrictedCommands: false };
}


export function normalizeHardware(raw: Partial<Record<string, unknown>> | undefined): HardwareSettings {
  const mode = raw?.mode;
  const rate = raw?.hourlyRateUsd;
  const count = typeof raw?.gpuCount === 'number' ? Math.floor(raw.gpuCount) : 1;
  return {
    mode: mode === 'local' || mode === 'cloud' ? mode : 'auto',
    gpuName: typeof raw?.gpuName === 'string' ? raw.gpuName.trim().slice(0, 80) : '',
    hourlyRateUsd: typeof rate === 'number' && isFinite(rate) && rate > 0 ? rate : null,
    provider: typeof raw?.provider === 'string' ? raw.provider.trim().slice(0, 60) : '',
    gpuCount: Math.min(Math.max(count || 1, 1), 64),
  };
}
