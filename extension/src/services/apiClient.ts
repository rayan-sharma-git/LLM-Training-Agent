import axios, { AxiosInstance, AxiosError } from 'axios';
import * as vscode from 'vscode';

/**
 * The complete end-to-end analysis produced by one Analyze run.
 *
 * The backend composes every real analyzer output into this single object and
 * renders `markdown` for display in the chat window. The structured fields are
 * kept so the webview (and history) can read individual values without
 * re-parsing the rendered text.
 */
export interface CompleteAnalysis {
  project_name: string;
  analysis_status: string;
  partial_failures: Array<Record<string, any>>;
  use_case: Record<string, any> | null;
  dataset_analysis: Record<string, any> | null;
  prompt_analysis: Record<string, any> | null;
  model_analysis: Record<string, any> | null;
  hardware: Record<string, any> | null;
  training_time_estimate: Record<string, any> | null;
  cost_estimate: Record<string, any> | null;
  budget: Record<string, any> | null;
  risks: Array<Record<string, any>>;
  recommendations: Array<Record<string, any>>;
  unavailable_sections: Array<Record<string, any>>;
  /** Pre-rendered Markdown shown in the chat window. */
  markdown: string;
}

export interface AnalysisResult {
  project: any;
  report: any;
  /** The single authoritative complete analysis (rendered in the chat). */
  analysis?: CompleteAnalysis;
  analysisStatus?: string;
  partialFailures?: Array<Record<string, any>>;
}

export interface ChatResponse {
  assistantResponse: string;
  references: string[];
  confidence: string;
  sessionId?: string;
  model?: string;
  error?: string;
}

export interface ProviderInfo {
  requiresKey: boolean;
  models: string[];
  description: string;
  configured: boolean;
}

export interface TestConnectionResult {
  success: boolean;
  message: string;
  provider?: string;
  model?: string;
  error?: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

const SECRET_KEY_PREFIX = 'llmTrainingAgent.apiKey.';

export class ApiClient {
  private client: AxiosInstance;
  private getDynamicBaseUrl?: () => string;
  private context: vscode.ExtensionContext | undefined;

  constructor(
    private baseUrl: string,
    getDynamicBaseUrl?: () => string,
    context?: vscode.ExtensionContext
  ) {
    this.getDynamicBaseUrl = getDynamicBaseUrl;
    this.context = context;
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 120000,
    });

    if (getDynamicBaseUrl) {
      this.client.interceptors.request.use((config) => {
        config.baseURL = getDynamicBaseUrl();
        return config;
      });
    }

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.code === 'ECONNREFUSED') {
          throw new ApiError(
            'Could not connect to the backend. The Python backend may not be running. ' +
            'Try reloading the VS Code window, or start it manually: ' +
            'cd backend && python -m uvicorn main:app --port 8000',
            0,
            { reason: 'connection_refused' }
          );
        }
        if (error.response) {
          const data = error.response.data as Record<string, any> | undefined;
          const apiError = data as { detail?: any; message?: string; errorCode?: string } | undefined;
          let message: string;
          if (apiError?.detail) {
            message = typeof apiError.detail === 'object' ? JSON.stringify(apiError.detail) : String(apiError.detail);
          } else if (apiError?.message) {
            message = String(apiError.message);
          } else {
            message = error.message;
          }
          throw new ApiError(message, error.response.status, data);
        } else if (error.request) {
          throw new ApiError(
            'No response from backend — the backend may still be starting. ' +
            'Check the Output panel (LLM Training Agent: Backend) for details.',
            0,
            { reason: 'network' }
          );
        } else {
          throw new ApiError(error.message, 0, { reason: 'client' });
        }
      }
    );
  }

  // ------------------------------------------------------------------
  // SecretStorage-backed API-key helpers
  // ------------------------------------------------------------------

  private getSecretStorage(): vscode.SecretStorage | undefined {
    return this.context?.secrets;
  }

  private getSecretKey(provider: string): string {
    return `${SECRET_KEY_PREFIX}${provider}`;
  }

  /**
   * Retrieve a stored API key for *provider* from VS Code SecretStorage.
   * Never logs or returns the key to the user interface directly.
   */
  async getApiKey(provider: string): Promise<string | undefined> {
    try {
      const secretStore = this.getSecretStorage();
      if (!secretStore) return undefined;
      const secret = await secretStore.get(this.getSecretKey(provider));
      return secret ?? undefined;
    } catch {
      return undefined;
    }
  }

  /**
   * Store an API key for *provider* in VS Code SecretStorage.
   */
  async setApiKey(provider: string, apiKey: string): Promise<void> {
    const secretStore = this.getSecretStorage();
    if (!secretStore) {
      throw new ApiError('VS Code SecretStorage is not available.', 0);
    }
    await secretStore.store(this.getSecretKey(provider), apiKey);
  }

  /**
   * Remove a stored API key for *provider* from VS Code SecretStorage.
   */
  async removeApiKey(provider: string): Promise<void> {
    const secretStore = this.getSecretStorage();
    if (!secretStore) {
      throw new ApiError('VS Code SecretStorage is not available.', 0);
    }
    await secretStore.delete(this.getSecretKey(provider));
  }

  // ------------------------------------------------------------------
  // API methods
  // ------------------------------------------------------------------

  async healthCheck(): Promise<{ status: string; version: string }> {
    const response = await this.client.get('/api/v1/health');
    return response.data;
  }

  async analyzeProject(projectPath: string): Promise<AnalysisResult> {
    if (!projectPath || !projectPath.trim()) {
      throw new ApiError('No project/workspace is currently open. Please open a folder first.', 0);
    }
    const response = await this.client.post('/api/v1/project/analyze', {
      projectPath,
    });
    return response.data as AnalysisResult;
  }

  async analyzeDataset(datasetPath: string): Promise<any> {
    const response = await this.client.post('/api/v1/dataset/analyze', {
      datasetPath,
    });
    return response.data;
  }

  /**
   * Clean every dataset file of a project in LLM-sized chunks.
   *
   * The backend discovers all dataset files (a directory entry is expanded
   * recursively), splits each file into sequential chunks and sends one
   * request per chunk. Cleaned files are written per source file together with
   * a manifest inside `<project>/.llm-training-agent/cleaned`.
   */
  async cleanDatasets(options: {
    projectPath?: string;
    datasetPaths?: string[];
    chunkSize?: number;
    maxChunkChars?: number;
    outputDirectory?: string;
    maxFiles?: number;
    useLlm?: boolean;
  } = {}): Promise<any> {
    const response = await this.client.post('/api/v1/dataset/clean', options);
    return response.data;
  }

  async sendChatMessage(
    message: string,
    options: { projectPath?: string; sessionId?: string; provider?: string } = {}
  ): Promise<ChatResponse> {
    const payload: Record<string, unknown> = { message };
    if (options.projectPath) {
      payload.projectPath = options.projectPath;
    }
    if (options.sessionId) {
      payload.sessionId = options.sessionId;
    }
    if (options.provider) {
      payload.provider = options.provider;
    }
    const response = await this.client.post('/api/v1/chat/message', payload);
    return response.data as ChatResponse;
  }

  async getChatHistory(
    options: { projectPath?: string; sessionId?: string } = {}
  ): Promise<{ messages: Array<{ role: string; content: string }> }> {
    const params: Record<string, string> = {};
    if (options.projectPath) {
      params.projectPath = options.projectPath;
    }
    if (options.sessionId) {
      params.sessionId = options.sessionId;
    }
    const response = await this.client.get('/api/v1/chat/history', { params });
    return response.data;
  }

  async getReport(projectPath?: string): Promise<any> {
    const response = await this.client.get('/api/v1/report', {
      params: projectPath ? { projectPath } : undefined,
    });
    return response.data;
  }

  async getRecommendations(projectPath?: string): Promise<{ recommendations: any[] }> {
    const response = await this.client.get('/api/v1/recommendations', {
      params: projectPath ? { projectPath } : undefined,
    });
    return response.data;
  }

  async getConfig(): Promise<{ default_provider: string }> {
    const response = await this.client.get('/api/v1/config');
    return response.data;
  }

  async updateConfig(config: Record<string, any>): Promise<any> {
    const response = await this.client.put('/api/v1/config', config);
    return response.data;
  }

  async listProviders(): Promise<{ providers: Record<string, ProviderInfo> }> {
    const response = await this.client.get('/api/v1/providers');
    return response.data;
  }

  async getProviderModels(provider: string): Promise<{ provider: string; models: string[]; error?: string }> {
    const response = await this.client.get('/api/v1/provider/models', {
      params: { provider },
    });
    return response.data;
  }

  async selectProvider(provider: string, model: string): Promise<any> {
    const response = await this.client.post('/api/v1/provider/select', {
      provider,
      model,
    });
    return response.data;
  }

  async testConnection(config: {
    provider: string;
    model?: string;
    apiKey?: string;
    baseUrl?: string;
  }): Promise<TestConnectionResult> {
    const response = await this.client.post('/api/v1/provider/test', config);
    return response.data as TestConnectionResult;
  }

  async removeApiKeyBackend(provider: string): Promise<any> {
    const response = await this.client.delete('/api/v1/config/key', {
      params: { provider },
    });
    return response.data;
  }

  // ------------------------------------------------------------------
  // GPU detection & time estimation
  // ------------------------------------------------------------------

  async detectGpu(): Promise<any> {
    const response = await this.client.get('/api/v1/gpu/detect');
    return response.data;
  }

  async listGpuSpecs(): Promise<{ gpus: string[] }> {
    const response = await this.client.get('/api/v1/gpu/specs');
    return response.data;
  }

  async estimateGpuTime(request: Record<string, any>): Promise<any> {
    const response = await this.client.post('/api/v1/gpu/estimate', request);
    return response.data;
  }

  async calibrateGpu(request: Record<string, any>): Promise<any> {
    const response = await this.client.post('/api/v1/gpu/calibrate', request);
    return response.data;
  }

  // ------------------------------------------------------------------
  // File changes: propose → review → approve → apply → verify → rollback
  // ------------------------------------------------------------------

  async proposeFileChange(
    filePath: string,
    proposedContent: string | null,
    projectRoot?: string,
    operation: 'create' | 'modify' | 'delete' = 'modify',
    reason?: string
  ): Promise<any> {
    const response = await this.client.post('/api/v1/files/propose', {
      filePath,
      proposedContent,
      operation,
      reason,
      projectRoot,
    });
    return response.data;
  }

  async listFileChanges(projectRoot?: string, includeTerminal = true): Promise<{ changes: any[] }> {
    const response = await this.client.get('/api/v1/files/changes', {
      params: { projectRoot, includeTerminal },
    });
    return response.data;
  }

  async getFileChange(changeId: string, projectRoot?: string, includeContents = true): Promise<any> {
    const response = await this.client.get(`/api/v1/files/changes/${changeId}`, {
      params: { includeContents, projectRoot },
    });
    return response.data;
  }

  /** Explicitly approve a proposal. Required before it can be applied. */
  async approveFileChange(changeId: string, projectRoot?: string): Promise<any> {
    const response = await this.client.post(`/api/v1/files/changes/${changeId}/approve`, {
      projectRoot,
    });
    return response.data;
  }

  async rejectFileChange(changeId: string, projectRoot?: string): Promise<any> {
    const response = await this.client.post(`/api/v1/files/changes/${changeId}/reject`, {
      projectRoot,
    });
    return response.data;
  }

  async cancelFileChange(changeId: string, projectRoot?: string): Promise<any> {
    const response = await this.client.post(`/api/v1/files/changes/${changeId}/cancel`, {
      projectRoot,
    });
    return response.data;
  }

  /** Apply an approved change. Fails with 409 when not approved or conflicted. */
  async applyFileChange(changeId: string, projectRoot?: string): Promise<any> {
    const response = await this.client.post(`/api/v1/files/changes/${changeId}/apply`, {
      projectRoot,
    });
    return response.data;
  }

  async verifyFileChange(changeId: string, projectRoot?: string): Promise<any> {
    const response = await this.client.get(`/api/v1/files/changes/${changeId}/verify`, {
      params: { projectRoot },
    });
    return response.data;
  }

  async rollbackFileChange(changeId: string, projectRoot?: string): Promise<any> {
    const response = await this.client.post(`/api/v1/files/changes/${changeId}/rollback`, {
      projectRoot,
    });
    return response.data;
  }
}
