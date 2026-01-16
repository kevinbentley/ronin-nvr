/**
 * Player controls component with play/pause, seek, volume, speed, and fullscreen.
 */

import { useCallback } from 'react';
import type { PlayerControlsProps } from './types';
import { PLAYBACK_SPEEDS } from './types';
import { ObjectTypeFilter } from './ObjectTypeFilter';
import './PlayerControls.css';

export function PlayerControls({
  isPlaying,
  currentTime,
  duration,
  volume,
  isMuted,
  playbackSpeed,
  isLive,
  isAtLiveEdge,
  timeBehindLive,
  showDetectionOverlay,
  visibleObjectTypes,
  typeCounts,
  onPlayPause,
  onSeek,
  onVolumeChange,
  onMuteToggle,
  onSpeedChange,
  onFullscreen,
  onReturnToLive,
  onToggleDetectionOverlay,
  onToggleObjectType,
  onToggleAllTypes,
}: PlayerControlsProps) {
  const formatTime = useCallback((seconds: number): string => {
    if (!isFinite(seconds) || isNaN(seconds)) return '0:00';

    const absSeconds = Math.abs(seconds);
    const hours = Math.floor(absSeconds / 3600);
    const mins = Math.floor((absSeconds % 3600) / 60);
    const secs = Math.floor(absSeconds % 60);

    const sign = seconds < 0 ? '-' : '';

    if (hours > 0) {
      return `${sign}${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${sign}${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  const formatTimeBehindLive = useCallback((seconds: number): string => {
    if (seconds < 1) return 'LIVE';
    return `-${formatTime(seconds)}`;
  }, [formatTime]);

  const handleSeekChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const time = parseFloat(e.target.value);
      onSeek(time);
    },
    [onSeek]
  );

  const handleVolumeChangeInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const vol = parseFloat(e.target.value);
      onVolumeChange(vol);
    },
    [onVolumeChange]
  );

  const handleSpeedSelect = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const speed = parseFloat(e.target.value);
      onSpeedChange(speed);
    },
    [onSpeedChange]
  );

  // Calculate seek bar progress percentage
  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="player-controls">
      <div className="controls-row controls-main">
        {/* Play/Pause */}
        <button
          className="control-btn play-pause-btn"
          onClick={onPlayPause}
          title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
        >
          {isPlaying ? (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        {/* Live indicator / Time display */}
        {isLive ? (
          <div className="live-time-display">
            {isAtLiveEdge ? (
              <span className="live-badge active">LIVE</span>
            ) : (
              <button
                className="live-badge behind"
                onClick={onReturnToLive}
                title="Return to live (L)"
              >
                {formatTimeBehindLive(timeBehindLive)}
              </button>
            )}
          </div>
        ) : (
          <span className="time-display">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        )}

        {/* Seek bar */}
        <div className="seek-container">
          <input
            type="range"
            className="seek-bar"
            min={0}
            max={duration || 0}
            value={currentTime}
            onChange={handleSeekChange}
            step={0.1}
            style={{
              background: `linear-gradient(to right, var(--accent-color) ${progressPercent}%, var(--track-color) ${progressPercent}%)`,
            }}
          />
        </div>

        {/* Speed selector (hidden for live at edge) */}
        {!isLive || !isAtLiveEdge ? (
          <select
            className="speed-select"
            value={playbackSpeed}
            onChange={handleSpeedSelect}
            title="Playback speed"
          >
            {PLAYBACK_SPEEDS.map((speed) => (
              <option key={speed} value={speed}>
                {speed}x
              </option>
            ))}
          </select>
        ) : null}

        {/* Detection overlay toggle */}
        <button
          className={`control-btn detection-btn ${showDetectionOverlay ? 'active' : ''}`}
          onClick={onToggleDetectionOverlay}
          title={showDetectionOverlay ? 'Hide detections (D)' : 'Show detections (D)'}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <rect x="7" y="7" width="10" height="10" rx="1" strokeDasharray="2 2" />
          </svg>
        </button>

        {/* Object type filter (only visible when detection overlay is on) */}
        <ObjectTypeFilter
          visible={showDetectionOverlay}
          visibleTypes={visibleObjectTypes}
          typeCounts={typeCounts}
          onToggleType={onToggleObjectType}
          onToggleAll={onToggleAllTypes}
        />

        {/* Volume controls */}
        <button
          className="control-btn volume-btn"
          onClick={onMuteToggle}
          title={isMuted ? 'Unmute (M)' : 'Mute (M)'}
        >
          {isMuted || volume === 0 ? (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z" />
            </svg>
          ) : volume < 0.5 ? (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M18.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM5 9v6h4l5 5V4L9 9H5z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
            </svg>
          )}
        </button>

        <input
          type="range"
          className="volume-bar"
          min={0}
          max={1}
          value={isMuted ? 0 : volume}
          onChange={handleVolumeChangeInput}
          step={0.05}
          title="Volume"
        />

        {/* Fullscreen */}
        <button
          className="control-btn fullscreen-btn"
          onClick={onFullscreen}
          title="Fullscreen (F)"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z" />
          </svg>
        </button>
      </div>
    </div>
  );
}
