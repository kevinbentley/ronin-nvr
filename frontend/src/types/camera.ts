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

// Playback types
export interface RecordingFile {
  id: string;
  camera_name: string;
  date: string;
  start_time: string;
  duration_seconds: number | null;
  size_bytes: number;
  filename: string;
}

export interface DayRecordings {
  camera_name: string;
  date: string;
  files: RecordingFile[];
  total_duration_seconds: number;
  total_size_bytes: number;
  start_time: string | null;
  end_time: string | null;
}

export interface ExportRequest {
  camera_name: string;
  start_time: string;
  end_time: string;
}

export interface ExportResponse {
  success: boolean;
  message: string;
  download_url?: string;
}

// ML Types
export interface MLJob {
  id: number;
  recording_id: number;
  model_name: string;
  status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  priority: number;
  progress_percent: number;
  frames_processed: number;
  total_frames: number;
  detections_count: number;
  started_at?: string;
  completed_at?: string;
  processing_time_seconds?: number;
  error_message?: string;
  created_at: string;
}

export interface MLJobListResponse {
  jobs: MLJob[];
  total: number;
}

export interface MLDetection {
  id: number;
  recording_id: number;
  camera_id: number;
  class_name: string;
  confidence: number;
  timestamp_ms: number;
  frame_number: number;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  model_name: string;
  model_version?: string;
  created_at: string;
}

export interface MLDetectionListResponse {
  detections: MLDetection[];
  total: number;
}

export interface MLDetectionSummaryItem {
  label: string;
  count: number;
  avg_confidence: number;
}

export interface MLDetectionSummary {
  total_detections: number;
  items: MLDetectionSummaryItem[];
}

export interface MLModel {
  id: number;
  name: string;
  display_name: string;
  version: string;
  file_path: string;
  model_type: string;
  class_names: string[];
  input_size: number[];
  default_confidence_threshold: number;
  default_nms_threshold: number;
  is_enabled: boolean;
  is_default: boolean;
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface MLModelListResponse {
  models: MLModel[];
}

export interface MLWorkerStatus {
  id: number;
  running: boolean;
  current_job: number | null;
}

export interface MLQueueStatus {
  pending: number;
  active: number;
  max_size: number;
  active_jobs: number[];
}

export interface MLStatus {
  running: boolean;
  workers: number;
  worker_status: MLWorkerStatus[];
  queue: MLQueueStatus;
  models_loaded: string[];
}

// Timeline events
export interface TimelineEvent {
  timestamp_ms: number;  // Milliseconds from start of day
  class_name: string;
  confidence: number;
  recording_id: number;
  count: number;
}

export interface TimelineEventsResponse {
  events: TimelineEvent[];
  total: number;
  class_counts: Record<string, number>;
}

// Live Detection types
export interface LiveDetectionConfig {
  fps: number;
  cooldown: number;
  confidence: number;
  classes: string[];
  model: string;
}

export interface LiveDetectionStatus {
  enabled: boolean;
  config: LiveDetectionConfig;
  detections_last_hour: number;
  active_cameras: number[];
}

export interface LiveDetection {
  id: number;
  camera_id: number;
  camera_name: string;
  class_name: string;
  confidence: number;
  detected_at: string | null;
  snapshot_url: string | null;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface LiveDetectionsResponse {
  detections: LiveDetection[];
  count: number;
}
