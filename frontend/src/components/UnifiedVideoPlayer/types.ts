/**
 * Shared types for the UnifiedVideoPlayer component.
 */

export type PlayerMode = 'live' | 'playback';

export type ConnectionState = 'connecting' | 'connected' | 'error' | 'reconnecting';

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
}

export interface TimelineEvent {
  timestamp_ms: number;
  class_name: string;
  confidence: number;
  recording_id: number | null;
  count: number;
}

export interface UnifiedVideoPlayerProps {
  /** Player mode: 'live' for HLS streams, 'playback' for recorded files */
  mode: PlayerMode;
  /** Video source URL */
  src: string;
  /** Camera ID for API calls */
  cameraId: number;
  /** Camera name for display */
  cameraName?: string;
  /** Camera online status (for live mode) */
  status?: 'online' | 'offline' | 'unknown';
  /** Whether camera is currently recording (for live mode) */
  isRecording?: boolean;
  /** Recording ID for fetching detections (for playback mode) */
  recordingId?: number;
  /** Recording ID as string for thumbnail API (for playback mode) */
  recordingIdString?: string;
  /** Recording start timestamp (for playback mode detection queries) */
  recordingStartTime?: Date;
  /** Whether to show controls */
  showControls?: boolean;
  /** Whether to show the mini-timeline */
  showTimeline?: boolean;
  /** Initial playback speed */
  initialSpeed?: number;
  /** Callback when playback position changes */
  onTimeUpdate?: (time: number) => void;
  /** Callback when seek is requested */
  onSeek?: (time: number) => void;
  /** Custom class name */
  className?: string;
}

export interface PlayerControlsProps {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  playbackSpeed: number;
  isLive: boolean;
  isAtLiveEdge: boolean;
  timeBehindLive: number;
  showDetectionOverlay: boolean;
  onPlayPause: () => void;
  onSeek: (time: number) => void;
  onVolumeChange: (volume: number) => void;
  onMuteToggle: () => void;
  onSpeedChange: (speed: number) => void;
  onFullscreen: () => void;
  onReturnToLive: () => void;
  onToggleDetectionOverlay: () => void;
}

export interface ThumbnailSprite {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ThumbnailData {
  spriteUrl: string;
  intervalSeconds: number;
  getThumbnailForTime: (time: number) => ThumbnailSprite | null;
}

export interface MiniTimelineProps {
  currentTime: number;
  duration: number;
  events: TimelineEvent[];
  isLive: boolean;
  liveWindowDuration: number;
  onSeek: (time: number) => void;
  onHover?: (time: number | null) => void;
  thumbnailData?: ThumbnailData | null;
}

export interface VideoCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  detections: Detection[];
  visible: boolean;
}

export interface LiveIndicatorProps {
  isAtLiveEdge: boolean;
  timeBehindLive: number;
  onReturnToLive: () => void;
}

export interface ThumbnailPreviewProps {
  visible: boolean;
  spriteUrl: string | null;
  spriteX: number;
  spriteY: number;
  spriteWidth: number;
  spriteHeight: number;
  time: number;
  positionX: number;
}

export const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4, 8, 16];

export const EVENT_COLORS: Record<string, string> = {
  motion: '#f59e0b',
  person: '#22c55e',
  car: '#3b82f6',
  truck: '#3b82f6',
  bus: '#3b82f6',
  motorcycle: '#3b82f6',
  bicycle: '#06b6d4',
  dog: '#a855f7',
  cat: '#a855f7',
  bird: '#a855f7',
  default: '#fbbf24',
};

export const getEventColor = (className: string): string => {
  return EVENT_COLORS[className.toLowerCase()] || EVENT_COLORS.default;
};
