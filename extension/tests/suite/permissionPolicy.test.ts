import { describe, it, expect } from 'vitest';
import {
  PERMISSION_DEFAULTS,
  normalizePermissions,
  effectivePermissions,
  normalizeHardware,
} from '../../src/services/permissionPolicy';

describe('permission settings', () => {
  it('defaults to reading only', () => {
    expect(PERMISSION_DEFAULTS).toEqual({
      readFiles: true, writeFiles: false, safeCommands: false, unrestrictedCommands: false,
    });
    expect(normalizePermissions(undefined)).toEqual(PERMISSION_DEFAULTS);
  });

  it('only accepts real booleans, so odd values can never turn a permission on', () => {
    const policy = normalizePermissions({ writeFiles: 'true', safeCommands: 1, unrestrictedCommands: {} });
    expect(policy.writeFiles).toBe(false);
    expect(policy.safeCommands).toBe(false);
    expect(policy.unrestrictedCommands).toBe(false);
  });

  it('keeps each permission independent', () => {
    expect(normalizePermissions({ safeCommands: true }).writeFiles).toBe(false);
    expect(normalizePermissions({ unrestrictedCommands: true }).safeCommands).toBe(false);
    expect(normalizePermissions({ readFiles: false }).readFiles).toBe(false);
  });

  it('forces writing and commands off in an untrusted workspace', () => {
    const all = { readFiles: true, writeFiles: true, safeCommands: true, unrestrictedCommands: true };
    expect(effectivePermissions(all, true)).toEqual(all);
    expect(effectivePermissions(all, false)).toEqual({
      readFiles: true, writeFiles: false, safeCommands: false, unrestrictedCommands: false,
    });
  });
});

describe('hardware settings', () => {
  it('falls back to auto and never invents a rate', () => {
    expect(normalizeHardware(undefined)).toEqual({
      mode: 'auto', gpuName: '', hourlyRateUsd: null, provider: '', gpuCount: 1,
    });
    expect(normalizeHardware({ mode: 'weird', hourlyRateUsd: -3 }).mode).toBe('auto');
    expect(normalizeHardware({ hourlyRateUsd: -3 }).hourlyRateUsd).toBeNull();
    expect(normalizeHardware({ hourlyRateUsd: NaN }).hourlyRateUsd).toBeNull();
  });

  it('keeps what the user entered, within bounds', () => {
    const hw = normalizeHardware({ mode: 'cloud', gpuName: ' A100 ', hourlyRateUsd: 1.5, provider: 'X', gpuCount: 500 });
    expect(hw).toEqual({ mode: 'cloud', gpuName: 'A100', hourlyRateUsd: 1.5, provider: 'X', gpuCount: 64 });
  });
});
