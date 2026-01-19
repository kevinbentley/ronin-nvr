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
  // ONVIF settings
  onvif_port?: number;
  onvif_enabled?: boolean;
  onvif_profile_token?: string;
  onvif_device_info?: ONVIFDeviceInfo;
  onvif_events_enabled?: boolean;
  // VLLM Activity Characterization
  scene_description?: string;
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
  // ONVIF settings
  onvif_port?: number;
  onvif_enabled?: boolean;
  onvif_events_enabled?: boolean;
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
  // ONVIF settings
  onvif_port?: number;
  onvif_enabled?: boolean;
  onvif_events_enabled?: boolean;
  // VLLM Activity Characterization
  scene_description?: string;
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
  is_in_progress?: boolean;  // True if recording is currently being written
  recording_id?: number;  // Database recording ID for detection queries
  camera_id?: number;  // Database camera ID for detection queries
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

// Event source type for unified timeline
export type EventSource = 'ml' | 'onvif_motion' | 'onvif_analytics';

// Timeline events
export interface TimelineEvent {
  timestamp_ms: number;  // Milliseconds from start of day
  class_name: string;
  confidence: number;
  recording_id: number | null;  // null for live detections
  count: number;
  event_source?: EventSource;  // Source of the detection
}

export interface TimelineEventsResponse {
  events: TimelineEvent[];
  total: number;
  class_counts: Record<string, number>;
}

// Detection types for bounding box overlay
export interface Detection {
  id: number;
  recording_id: number | null;
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
  detected_at?: string | null;  // Actual detection time (for live detections)
  snapshot_url?: string;
}

export interface DetectionListResponse {
  detections: Detection[];
  total: number;
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
  // VLLM activity characterization
  llm_description: string | null;
  concern_level: 'none' | 'low' | 'medium' | 'high' | 'emergency' | null;
  activity_type: string | null;
}

export interface LiveDetectionsResponse {
  detections: LiveDetection[];
  count: number;
  total: number;
  offset: number;
  limit: number;
}

// Global ML Settings types
export interface MLSettings {
  live_detection_enabled: boolean;
  live_detection_fps: number;
  live_detection_cooldown: number;
  live_detection_confidence: number;
  live_detection_classes: string[];
  class_thresholds: Record<string, number>;
  updated_at: string | null;
}

export interface MLSettingsUpdate {
  live_detection_enabled?: boolean;
  live_detection_fps?: number;
  live_detection_cooldown?: number;
  live_detection_confidence?: number;
  live_detection_classes?: string[];
  class_thresholds?: Record<string, number>;
}

// Object Event types (FSM state transitions)
export interface ObjectEvent {
  id: number;
  event_type: string;
  class_name: string;
  track_id: number;
  old_state: string | null;
  new_state: string | null;
  confidence: number;
  duration_seconds: number;
  snapshot_url: string | null;
  camera_id: number;
  camera_name: string | null;
  event_time: string;
}

export interface ObjectEventListResponse {
  events: ObjectEvent[];
  total: number;
  offset: number;
  limit: number;
}

// Transcode types
export interface TranscodeQueueStatus {
  pending_files: number;
  pending_size_bytes: number;
  pending_size_gb: number;
}

export interface TranscodeWorkerInfo {
  worker_id: string;
  is_active: boolean;
  current_file: string | null;
  last_seen: string | null;
}

export interface TranscodeStatus {
  enabled: boolean;
  files_transcoded: number;
  files_failed: number;
  total_original_gb: number;
  total_new_gb: number;
  total_savings_gb: number;
  average_savings_percent: number;
  by_encoder: Record<string, number>;
  queue: TranscodeQueueStatus;
  workers: TranscodeWorkerInfo[];
}

// Retention Settings types
export interface RetentionSettings {
  retention_days: number | null;
  retention_max_gb: number | null;
  retention_check_interval_minutes: number;
}

export interface RetentionSettingsUpdate {
  retention_days?: number | null;
  retention_max_gb?: number | null;
}

// ONVIF Types

export interface ONVIFDeviceInfo {
  manufacturer?: string;
  model?: string;
  firmware?: string;
  serial?: string;
  hardware_id?: string;
}

export interface ONVIFProfile {
  token: string;
  name: string;
  rtsp_url: string;
  encoding?: string;
  resolution?: string;
  fps?: number;
}

export interface ONVIFProbeRequest {
  host: string;
  onvif_port?: number;
  username?: string;
  password?: string;
  timeout?: number;
}

export interface ONVIFProbeResponse {
  success: boolean;
  host: string;
  device_info: ONVIFDeviceInfo;
  profiles: ONVIFProfile[];
  has_events: boolean;
  has_analytics: boolean;
  has_ptz: boolean;
  error?: string;
}

export interface ONVIFProfilesResponse {
  camera_id: number;
  profiles: ONVIFProfile[];
}

export interface ONVIFApplyProfileRequest {
  profile_token: string;
  rtsp_url: string;
}

export interface ONVIFApplyProfileResponse {
  success: boolean;
  camera_id: number;
  new_path: string;
  new_port: number;
  profile_token: string;
}

export interface ONVIFEventsResponse {
  success: boolean;
  camera_id: number;
  events_enabled: boolean;
}
