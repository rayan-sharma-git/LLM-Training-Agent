import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios, { AxiosError } from 'axios';
import { ApiClient, ApiError, ChatResponse, permissionRequiredAction } from '../../src/services/apiClient';

vi.mock('axios');

describe('ApiClient', () => {
  let client: ApiClient;
  let get: ReturnType<typeof vi.fn>;
  let post: ReturnType<typeof vi.fn>;
  let put: ReturnType<typeof vi.fn>;
  let del: ReturnType<typeof vi.fn>;

  const createClient = () => {
    get = vi.fn();
    post = vi.fn();
    put = vi.fn();
    del = vi.fn();

    const clientInstance = {
      get,
      post,
      put,
      delete: del,
      interceptors: {
        response: {
          use: vi.fn(() => 0),
        },
        request: { use: vi.fn() },
      },
    } as any;

    (axios.create as any) = vi.fn(() => clientInstance);
    client = new ApiClient('http://127.0.0.1:8000');
  };

  beforeEach(() => {
    vi.clearAllMocks();
    createClient();
  });

  it('should health check successfully', async () => {
    get.mockResolvedValueOnce({ data: { status: 'ok', version: '1.0.0' } });
    const result = await client.healthCheck();
    expect(result).toEqual({ status: 'ok', version: '1.0.0' });
    expect(get).toHaveBeenCalledWith('/api/v1/health');
  });

  it('should analyze project', async () => {
    const mockResponse = { project: { name: 'test' }, report: {} };
    post.mockResolvedValueOnce({ data: mockResponse });
    const result = await client.analyzeProject('/tmp/test');
    expect(result).toEqual(mockResponse);
    expect(post).toHaveBeenCalledWith('/api/v1/project/analyze', { projectPath: '/tmp/test' });
  });

  it('should send chat message', async () => {
    const mockResponse: ChatResponse = {
      assistantResponse: 'hello',
      references: [],
      confidence: 'high',
    };
    post.mockResolvedValueOnce({ data: mockResponse });
    const result = await client.sendChatMessage('hello');
    expect(result).toEqual(mockResponse);
    expect(post).toHaveBeenCalledWith('/api/v1/chat/message', { message: 'hello' });
  });

  it('should get a project report with its project scope', async () => {
    get.mockResolvedValueOnce({ data: { report: 'test' } });
    const result = await client.getReport('/tmp/project');
    expect(result).toEqual({ report: 'test' });
    expect(get).toHaveBeenCalledWith('/api/v1/report', { params: { projectPath: '/tmp/project' } });
  });

  it('should get project recommendations with their project scope', async () => {
    get.mockResolvedValueOnce({ data: { recommendations: [] } });
    const result = await client.getRecommendations('/tmp/project');
    expect(result).toEqual({ recommendations: [] });
    expect(get).toHaveBeenCalledWith('/api/v1/recommendations', { params: { projectPath: '/tmp/project' } });
  });

  it('should reject on network errors', async () => {
    const error = new Error('Network Error') as AxiosError;
    error.request = {};
    get.mockRejectedValueOnce(error);
    await expect(client.healthCheck()).rejects.toThrow('Network Error');
  });

  it('should handle API errors', async () => {
    const error = new Error('API Error') as AxiosError;
    error.response = { status: 500, data: { detail: 'Server Error' } } as any;
    get.mockRejectedValueOnce(error);
    await expect(client.healthCheck()).rejects.toThrow('API Error');
  });

  it('should analyze dataset', async () => {
    post.mockResolvedValueOnce({ data: { dataset: 'analysis' } });
    const result = await client.analyzeDataset('/tmp/dataset');
    expect(result).toEqual({ dataset: 'analysis' });
    expect(post).toHaveBeenCalledWith('/api/v1/dataset/analyze', { datasetPath: '/tmp/dataset' });
  });

  it('should list providers', async () => {
    get.mockResolvedValueOnce({ data: { providers: ['openai', 'ollama'] } });
    const result = await client.listProviders();
    expect(result).toEqual({ providers: ['openai', 'ollama'] });
  });

  it('sends hardware settings with the analysis request only when given', async () => {
    post.mockResolvedValue({ data: {} });
    await client.analyzeProject('/p', { mode: 'cloud', hourlyRateUsd: 2, gpuName: 'A100' });
    expect(post).toHaveBeenCalledWith('/api/v1/project/analyze', {
      projectPath: '/p',
      hardware: { mode: 'cloud', hourlyRateUsd: 2, gpuName: 'A100' },
    });
  });

  describe('chat agent', () => {
    it('sends a message with a long timeout and no session so the backend creates the chat', async () => {
      post.mockResolvedValueOnce({ data: { sessionId: 's1', title: 'T', message: { messageId: 'm' } } });
      const result = await client.agentChat('change epochs to 5', { projectPath: '/p' });
      expect(result.sessionId).toBe('s1');
      const [url, body, config] = post.mock.calls[0];
      expect(url).toBe('/api/v1/agent/chat');
      expect(body).toEqual({ message: 'change epochs to 5', projectPath: '/p' });
      expect(config.timeout).toBeGreaterThan(10 * 60 * 1000);
    });

    it('answers a card by action id', async () => {
      post.mockResolvedValueOnce({ data: { sessionId: 's1' } });
      await client.agentRespond('a/b', 'allow', { projectPath: '/p', sessionId: 's1' });
      const [url, body] = post.mock.calls[0];
      expect(url).toBe('/api/v1/agent/actions/a%2Fb/respond');
      expect(body).toEqual({ choice: 'allow', projectPath: '/p', sessionId: 's1' });
    });

    it('lists, opens and deletes chats within the project scope', async () => {
      get.mockResolvedValueOnce({ data: { sessions: [] } });
      await client.listChatSessions('/p');
      expect(get).toHaveBeenCalledWith('/api/v1/chat/sessions', { params: { projectPath: '/p' } });
      get.mockResolvedValueOnce({ data: { sessionId: 's', title: 't', messages: [] } });
      await client.getChatSession('s', '/p');
      expect(get).toHaveBeenLastCalledWith('/api/v1/chat/sessions/s', { params: { projectPath: '/p' } });
      del.mockResolvedValueOnce({ data: {} });
      await client.deleteChatSession('s', '/p');
      expect(del).toHaveBeenCalledWith('/api/v1/chat/sessions/s', { params: { projectPath: '/p' } });
    });

    it('shows changes, stores notes and manages permissions', async () => {
      post.mockResolvedValue({ data: {} });
      await client.showChanges({ projectPath: '/p', sessionId: 's' });
      expect(post).toHaveBeenLastCalledWith('/api/v1/agent/changes/show', { projectPath: '/p', sessionId: 's' });
      await client.addChatNote({ projectPath: '/p', content: 'x', title: 'Project analysis' });
      expect(post).toHaveBeenLastCalledWith('/api/v1/agent/chat/note', {
        projectPath: '/p', content: 'x', title: 'Project analysis',
      });
      await client.allowOnce('/p', 'read');
      expect(post).toHaveBeenLastCalledWith('/api/v1/agent/permissions/allow-once', { projectPath: '/p', action: 'read' });
      put.mockResolvedValueOnce({ data: {} });
      const policy = { readFiles: true, writeFiles: false, safeCommands: false, unrestrictedCommands: false };
      await client.setPermissions('/p', policy);
      expect(put).toHaveBeenCalledWith('/api/v1/agent/permissions', { projectPath: '/p', permissions: policy });
    });
  });

  describe('permission errors', () => {
    it('recognises a PERMISSION_REQUIRED answer and names the action', () => {
      const error = new ApiError('Permission needed', 403, {
        detail: { errorCode: 'PERMISSION_REQUIRED', action: 'read', message: 'Permission needed' },
      });
      expect(permissionRequiredAction(error)).toBe('read');
    });

    it('ignores other errors', () => {
      expect(permissionRequiredAction(new ApiError('x', 500, { detail: 'boom' }))).toBeUndefined();
      expect(permissionRequiredAction(new ApiError('x', 403, { detail: 'forbidden' }))).toBeUndefined();
      expect(permissionRequiredAction(new Error('x'))).toBeUndefined();
    });
  });
});
