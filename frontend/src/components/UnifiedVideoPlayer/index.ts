/**
 * UnifiedVideoPlayer component exports.
 */

export { UnifiedVideoPlayer } from './UnifiedVideoPlayer';
export { PlayerControls } from './PlayerControls';
export { LiveIndicator } from './LiveIndicator';
export { MiniTimeline } from './MiniTimeline';
export { ThumbnailPreview } from './ThumbnailPreview';

// Hooks
export { useHlsPlayer } from './hooks/useHlsPlayer';
export { usePlaybackSpeed, applyPlaybackSpeed } from './hooks/usePlaybackSpeed';
export { useTimelineEvents } from './hooks/useTimelineEvents';
export { useThumbnails } from './hooks/useThumbnails';

// Types
export type {
  PlayerMode,
  ConnectionState,
  Detection,
  TimelineEvent,
  ThumbnailSprite,
  ThumbnailData,
  UnifiedVideoPlayerProps,
  PlayerControlsProps,
  MiniTimelineProps,
  LiveIndicatorProps,
  ThumbnailPreviewProps,
} from './types';

export { PLAYBACK_SPEEDS, EVENT_COLORS, getEventColor } from './types';
