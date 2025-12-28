/**
 * API client for RoninNVR backend.
 */

import axios, { type AxiosInstance, type AxiosError } from 'axios';
import type {
  Camera,
  CameraCreate,
  CameraUpdate,
  CameraTestResult,
  RecordingStatus,
  StorageStats,
  RecordingFile,
  DayRecordings,
  ExportRequest,
  ExportResponse,
  MLStatus,
  MLJobListResponse,
  MLJob,
  MLDetectionSummary,
  MLModelListResponse,
  TimelineEventsResponse,
} from '../types/camera';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api';
const TOKEN_KEY = 'auth_token';

// Auth types
export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: number;
  username: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
}

// Event for auth state changes
type AuthChangeCallback = (isAuthenticated: boolean) => void;
let authChangeCallback: AuthChangeCallback | null = null;

export function setAuthChangeCallback(callback: AuthChangeCallback): void {
  authChangeCallback = callback;
}

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor: add auth token
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem(TOKEN_KEY);
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Response interceptor: handle 401 errors
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expired or invalid - clear it and notify
          localStorage.removeItem(TOKEN_KEY);
          if (authChangeCallback) {
            authChangeCallback(false);
          }
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth methods
  async login(credentials: LoginRequest): Promise<TokenResponse> {
    const response = await this.client.post('/auth/login', credentials);
    const data: TokenResponse = response.data;
    localStorage.setItem(TOKEN_KEY, data.access_token);
    if (authChangeCallback) {
      authChangeCallback(true);
    }
    return data;
  }

  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    if (authChangeCallback) {
      authChangeCallback(false);
    }
  }

  async getMe(): Promise<User> {
    const response = await this.client.get('/auth/me');
    return response.data;
  }

  isAuthenticated(): boolean {
    return !!localStorage.getItem(TOKEN_KEY);
  }

  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  // Health check
  async getHealth(): Promise<{ status: string; database: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // Camera CRUD
  async listCameras(): Promise<Camera[]> {
    const response = await this.client.get('/cameras');
    return response.data.cameras;
  }

  async getCamera(id: number): Promise<Camera> {
    const response = await this.client.get(`/cameras/${id}`);
    return response.data;
  }

  async createCamera(data: CameraCreate): Promise<Camera> {
    const response = await this.client.post('/cameras', data);
    return response.data;
  }

  async updateCamera(id: number, data: CameraUpdate): Promise<Camera> {
    const response = await this.client.put(`/cameras/${id}`, data);
    return response.data;
  }

  async deleteCamera(id: number): Promise<void> {
    await this.client.delete(`/cameras/${id}`);
  }

  async testCamera(id: number): Promise<CameraTestResult> {
    const response = await this.client.post(`/cameras/${id}/test`);
    return response.data;
  }

  // Recording control
  async startRecording(id: number): Promise<RecordingStatus> {
    const response = await this.client.post(`/cameras/${id}/recording/start`);
    return response.data;
  }

  async stopRecording(id: number): Promise<RecordingStatus> {
    const response = await this.client.post(`/cameras/${id}/recording/stop`);
    return response.data;
  }

  async getRecordingStatus(id: number): Promise<RecordingStatus> {
    const response = await this.client.get(`/cameras/${id}/recording/status`);
    return response.data;
  }

  async getAllRecordingStatus(): Promise<RecordingStatus[]> {
    const response = await this.client.get('/cameras/recording/status');
    return response.data;
  }

  // Storage
  async getStorageStats(): Promise<StorageStats> {
    const response = await this.client.get('/storage/stats');
    return response.data;
  }

  async runStorageCleanup(): Promise<{
    files_scanned: number;
    files_deleted: number;
    bytes_freed: number;
    gb_freed: number;
  }> {
    const response = await this.client.post('/storage/cleanup');
    return response.data;
  }

  // Get HLS stream URL for a camera
  getStreamUrl(id: number): string {
    return `${API_BASE}/cameras/${id}/stream/hls/playlist.m3u8`;
  }

  // Stream control
  async startStream(id: number): Promise<{ camera_id: number; streaming: boolean }> {
    const response = await this.client.post(`/cameras/${id}/stream/start`);
    return response.data;
  }

  async stopStream(id: number): Promise<{ camera_id: number; streaming: boolean }> {
    const response = await this.client.post(`/cameras/${id}/stream/stop`);
    return response.data;
  }

  // Playback API
  async getCamerasWithRecordings(): Promise<string[]> {
    const response = await this.client.get('/playback/cameras');
    return response.data.cameras;
  }

  async getAvailableDates(cameraName: string): Promise<string[]> {
    const response = await this.client.get(`/playback/cameras/${cameraName}/dates`);
    return response.data.dates;
  }

  async getDayRecordings(cameraName: string, date: string): Promise<DayRecordings> {
    const response = await this.client.get(
      `/playback/cameras/${cameraName}/recordings`,
      { params: { date } }
    );
    return response.data;
  }

  async listRecordings(params?: {
    camera_name?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ recordings: RecordingFile[]; total: number }> {
    const response = await this.client.get('/playback/recordings', { params });
    return response.data;
  }

  async getRecording(recordingId: string): Promise<RecordingFile> {
    const response = await this.client.get(`/playback/recordings/${recordingId}`);
    return response.data;
  }

  getRecordingStreamUrl(recordingId: string): string {
    return `${API_BASE}/playback/recordings/${recordingId}/stream`;
  }

  getRecordingDownloadUrl(recordingId: string): string {
    return `${API_BASE}/playback/recordings/${recordingId}/download`;
  }

  async exportClip(request: ExportRequest): Promise<ExportResponse> {
    const response = await this.client.post('/playback/export', request);
    return response.data;
  }

  getExportDownloadUrl(exportId: string): string {
    return `${API_BASE}/playback/exports/${exportId}`;
  }

  // ML API
  async getMLStatus(): Promise<MLStatus> {
    const response = await this.client.get('/ml/status');
    return response.data;
  }

  async getMLJobs(params?: {
    status?: string;
    recording_id?: number;
    limit?: number;
    offset?: number;
  }): Promise<MLJobListResponse> {
    const response = await this.client.get('/ml/jobs', { params });
    return response.data;
  }

  async getMLJob(jobId: number): Promise<MLJob> {
    const response = await this.client.get(`/ml/jobs/${jobId}`);
    return response.data;
  }

  async createMLJob(recordingId: number, modelName?: string, priority?: number): Promise<MLJob> {
    const response = await this.client.post('/ml/jobs', {
      recording_id: recordingId,
      model_name: modelName,
      priority: priority ?? 0,
    });
    return response.data;
  }

  async cancelMLJob(jobId: number): Promise<void> {
    await this.client.delete(`/ml/jobs/${jobId}`);
  }

  async getMLDetectionSummary(params?: {
    camera_id?: number;
    recording_id?: number;
    group_by?: 'class_name' | 'camera' | 'hour';
  }): Promise<MLDetectionSummary> {
    const response = await this.client.get('/ml/detections/summary', { params });
    return response.data;
  }

  async getMLModels(): Promise<MLModelListResponse> {
    const response = await this.client.get('/ml/models');
    return response.data;
  }

  getMLEventsUrl(): string {
    const token = this.getToken();
    return `${API_BASE}/ml/events${token ? `?token=${token}` : ''}`;
  }

  async startMLSystem(): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post('/ml/start');
    return response.data;
  }

  async stopMLSystem(): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post('/ml/stop');
    return response.data;
  }

  async processAllRecordings(params?: {
    camera_name?: string;
    limit?: number;
  }): Promise<{ success: boolean; queued: number; message: string }> {
    const response = await this.client.post('/ml/process-all', null, { params });
    return response.data;
  }

  async retryFailedJobs(params?: {
    error_filter?: string;
  }): Promise<{ success: boolean; reset_count: number; message: string }> {
    const response = await this.client.post('/ml/retry-failed', null, { params });
    return response.data;
  }

  async resetStuckJobs(): Promise<{ success: boolean; reset_count: number; message: string }> {
    const response = await this.client.post('/ml/reset-stuck');
    return response.data;
  }

  async getTimelineEvents(params: {
    camera_name: string;
    date: string;
    class_filter?: string;
    min_confidence?: number;
  }): Promise<TimelineEventsResponse> {
    const response = await this.client.get('/ml/detections/timeline', { params });
    return response.data;
  }
}

export const api = new ApiClient();
export default api;
