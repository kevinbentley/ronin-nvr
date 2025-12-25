/**
 * Camera types matching backend API schemas.
 */

export interface Camera {
  id: number;
  name: string;
  host: string;
  port: number;
  path: string;
  username?: string;
  password?: string;
  transport: 'tcp' | 'udp';
  recording_enabled: boolean;
  status: 'online' | 'offline' | 'unknown';
  last_seen?: string;
  created_at: string;
  updated_at: string;
}

export interface CameraCreate {
  name: string;
  host: string;
  port?: number;
  path?: string;
  username?: string;
  password?: string;
  transport?: 'tcp' | 'udp';
  recording_enabled?: boolean;
}

export interface CameraUpdate {
  name?: string;
  host?: string;
  port?: number;
  path?: string;
  username?: string;
  password?: string;
  transport?: 'tcp' | 'udp';
  recording_enabled?: boolean;
}

export interface CameraTestResult {
  success: boolean;
  message: string;
  codec?: string;
  resolution?: string;
  fps?: number;
}

export interface RecordingStatus {
  camera_id: number;
  camera_name: string;
  is_recording: boolean;
  started_at?: string;
  current_file?: string;
  error?: string;
}

export interface StorageStats {
  total_size_bytes: number;
  total_size_gb: number;
  total_size_mb: number;
  total_files: number;
  oldest_file?: string;
  newest_file?: string;
  cameras: CameraStorageStats[];
}

export interface CameraStorageStats {
  name: string;
  size_bytes: number;
  size_gb: number;
  file_count: number;
}

export type GridLayout = '1x1' | '2x2' | '3x3' | '4x4';
