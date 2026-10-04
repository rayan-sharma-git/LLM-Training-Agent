import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios, { AxiosError } from 'axios';
import { ApiClient, ApiError, ChatResponse } from '../../src/services/apiClient';

vi.mock('axios');

describe('ApiClient', () => {
  let client: ApiClient;
  let get: ReturnType<typeof vi.fn>;
  let post: ReturnType<typeof vi.fn>;
  let put: ReturnType<typeof vi.fn>;

  const createClient = () => {
    get = vi.fn();
    post = vi.fn();
    put = vi.fn();

    const clientInstance = {
      get,
      post,
      put,
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
});