import * as vscode from 'vscode';
import { ApiClient, PermissionPolicy, HardwareSettings } from './apiClient';
import { PERMISSION_KEYS, normalizePermissions, effectivePermissions, normalizeHardware } from './permissionPolicy';

export function readPermissionSettings(): PermissionPolicy {
  const config = vscode.workspace.getConfiguration('llmTrainingAgent.permissions');
  const raw: Record<string, unknown> = {};
  for (const key of PERMISSION_KEYS) {
    raw[key] = config.get(key);
  }
  return effectivePermissions(normalizePermissions(raw), vscode.workspace.isTrusted !== false);
}

export function readHardwareSettings(): HardwareSettings {
  const config = vscode.workspace.getConfiguration('llmTrainingAgent.hardware');
  return normalizeHardware({
    mode: config.get('mode'),
    gpuName: config.get('gpuName'),
    hourlyRateUsd: config.get('hourlyRateUsd'),
    provider: config.get('provider'),
    gpuCount: config.get('gpuCount'),
  });
}

/** Push the user's permissions for the open project(s) to the backend that enforces them. */
export async function syncPermissions(apiClient: ApiClient): Promise<void> {
  const policy = readPermissionSettings();
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    await apiClient.setPermissions(folder.uri.fsPath, policy);
  }
}
