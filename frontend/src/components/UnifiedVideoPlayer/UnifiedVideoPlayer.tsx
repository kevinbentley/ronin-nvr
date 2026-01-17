/**
 * Unified video player component supporting both live HLS streams and recorded playback.
 * Features: DVR/timeshift, playback speed control, timeline with detection events, keyboard shortcuts.
 */

import { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import type { UnifiedVideoPlayerProps, ThumbnailData } from './types';
import { useHlsPlayer } from './hooks/useHlsPlayer';
import { usePlaybackSpeed, applyPlaybackSpeed } from './hooks/usePlaybackSpeed';
import { useTimelineEvents } from './hooks/useTimelineEvents';
import { useThumbnails } from './hooks/useThumbnails';
import { PlayerControls } from './PlayerControls';
import { LiveIndicator } from './LiveIndicator';
import { MiniTimeline } from './MiniTimeline';
import './UnifiedVideoPlayer.css';

const DVR_WINDOW_DURATION = 900; // 15 minutes in seconds

export function UnifiedVideoPlayer({
  mode,
  src,
  cameraId,
  cameraName,
  status = 'online',
  isRecording = false,
  recordingId,
  recordingIdString,
  recordingStartTime,
  recordingDuration,
  showControls = true,
  showTimeline = true,
  initialSpeed = 1,
  onTimeUpdate,
  onSeek,
  className,
}: UnifiedVideoPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(true);
  const [controlsVisible, setControlsVisible] = useState(false);
  const controlsTimeoutRef = useRef<number | null>(null);

  const {
    videoRef,
    connectionState,
    errorMessage,
    isPlaying,
    currentTime,
    duration,
    isAtLiveEdge,
    timeBehindLive,
    togglePlay,
    seek,
    seekToLive,
    reconnect,
  } = useHlsPlayer({
    mode,
    src,
    cameraId,
    status,
    onTimeUpdate,
  });

  const { speed, setSpeed } = usePlaybackSpeed({ initialSpeed });

  const isLive = mode === 'live';

  // Timeline events
  const { events: timelineEvents } = useTimelineEvents({
    recordingId,
    cameraId,
    cameraName,
    recordingStartTime,
    recordingDuration,
    duration: isLive ? DVR_WINDOW_DURATION : duration,
    isLive,
    enabled: showTimeline,
  });

  // Thumbnail preview (playback mode only)
  const { thumbnailData: rawThumbnailData, getThumbnailForTime } = useThumbnails({
    recordingId: recordingIdString,
    enabled: mode === 'playback' && showTimeline,
  });

  // Memoize thumbnail data for MiniTimeline
  const thumbnailData: ThumbnailData | null = useMemo(() => {
    if (!rawThumbnailData) return null;
    return {
      spriteUrl: rawThumbnailData.spriteUrl,
      intervalSeconds: rawThumbnailData.intervalSeconds,
      getThumbnailForTime,
    };
  }, [rawThumbnailData, getThumbnailForTime]);

  // Apply playback speed when it changes or when video becomes connected
  useEffect(() => {
    if (connectionState === 'connected') {
      applyPlaybackSpeed(videoRef.current, speed);
    }
  }, [speed, videoRef, connectionState]);

  // Volume control
  const handleVolumeChange = useCallback((newVolume: number) => {
    const video = videoRef.current;
    if (video) {
      video.volume = newVolume;
      setVolume(newVolume);
      if (newVolume > 0) {
        video.muted = false;
        setIsMuted(false);
      }
    }
  }, [videoRef]);

  const handleMuteToggle = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      video.muted = !video.muted;
      setIsMuted(video.muted);
    }
  }, [videoRef]);

  // Fullscreen
  const handleFullscreen = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      container.requestFullscreen();
    }
  }, []);

  // Seek handler that also notifies parent
  const handleSeek = useCallback((time: number) => {
    seek(time);
    onSeek?.(time);
  }, [seek, onSeek]);

  // Show controls on mouse movement
  const showControlsTemporarily = useCallback(() => {
    setControlsVisible(true);
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current);
    }
    controlsTimeoutRef.current = window.setTimeout(() => {
      setControlsVisible(false);
    }, 3000);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key.toLowerCase()) {
        case ' ':
          e.preventDefault();
          togglePlay();
          showControlsTemporarily();
          break;
        case 'k':
          togglePlay();
          showControlsTemporarily();
          break;
        case 'f':
          handleFullscreen();
          break;
        case 'm':
          handleMuteToggle();
          showControlsTemporarily();
          break;
        case 'l':
          if (mode === 'live') {
            seekToLive();
          }
          showControlsTemporarily();
          break;
        case 'arrowleft':
          handleSeek(Math.max(0, currentTime - 10));
          showControlsTemporarily();
          break;
        case 'arrowright':
          handleSeek(Math.min(duration, currentTime + 10));
          showControlsTemporarily();
          break;
        case 'arrowup':
          handleVolumeChange(Math.min(1, volume + 0.1));
          showControlsTemporarily();
          break;
        case 'arrowdown':
          handleVolumeChange(Math.max(0, volume - 0.1));
          showControlsTemporarily();
          break;
        case ',':
          // Previous frame (when paused)
          if (!isPlaying && videoRef.current) {
            handleSeek(Math.max(0, currentTime - 1 / 30));
          }
          break;
        case '.':
          // Next frame (when paused)
          if (!isPlaying && videoRef.current) {
            handleSeek(Math.min(duration, currentTime + 1 / 30));
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    mode,
    togglePlay,
    handleFullscreen,
    handleMuteToggle,
    handleVolumeChange,
    handleSeek,
    seekToLive,
    showControlsTemporarily,
    currentTime,
    duration,
    volume,
    isPlaying,
    videoRef,
  ]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (controlsTimeoutRef.current) {
        clearTimeout(controlsTimeoutRef.current);
      }
    };
  }, []);

  const showReconnectButton = connectionState === 'error';

  return (
    <div
      ref={containerRef}
      className={`unified-player ${className || ''} ${controlsVisible ? 'controls-visible' : ''} ${showTimeline ? 'has-timeline' : ''}`}
      onMouseMove={showControlsTemporarily}
      onMouseLeave={() => setControlsVisible(false)}
    >
      {/* Header */}
      {cameraName && (
        <div className="player-header">
          <span className="camera-name">{cameraName}</span>
          <div className="header-indicators">
            {isRecording && <span className="recording-indicator">REC</span>}
            {isLive && (
              <LiveIndicator
                isAtLiveEdge={isAtLiveEdge}
                timeBehindLive={timeBehindLive}
                onReturnToLive={seekToLive}
              />
            )}
          </div>
        </div>
      )}

      {/* Video container */}
      <div className="video-wrapper" onClick={togglePlay}>
        {status === 'online' || mode === 'playback' ? (
          <>
            <video
              ref={videoRef}
              muted={isMuted}
              playsInline
              className="video-element"
              preload={mode === 'playback' ? 'metadata' : 'auto'}
            />

            {/* Loading overlay */}
            {connectionState === 'connecting' && (
              <div className="video-overlay">
                <div className="loading-spinner" />
                <span className="overlay-text">Connecting...</span>
              </div>
            )}

            {/* Reconnecting overlay */}
            {connectionState === 'reconnecting' && (
              <div className="video-overlay">
                <div className="loading-spinner" />
                <span className="overlay-text">{errorMessage || 'Reconnecting...'}</span>
              </div>
            )}

            {/* Error overlay with reconnect button */}
            {showReconnectButton && (
              <div className="video-overlay error-overlay" onClick={(e) => { e.stopPropagation(); reconnect(); }}>
                <span className="overlay-text error">{errorMessage}</span>
                <button className="reconnect-btn">Reconnect</button>
              </div>
            )}

            {/* Large play button overlay */}
            {!isPlaying && connectionState === 'connected' && (
              <button className="play-overlay-btn" onClick={(e) => { e.stopPropagation(); togglePlay(); }}>
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
            )}
          </>
        ) : (
          <div className="video-placeholder">
            <span className="placeholder-text">
              {status === 'offline' ? 'Camera Offline' : 'Connecting...'}
            </span>
          </div>
        )}
      </div>

      {/* Player controls */}
      {showControls && (
        <PlayerControls
          isPlaying={isPlaying}
          currentTime={currentTime}
          duration={duration}
          volume={volume}
          isMuted={isMuted}
          playbackSpeed={speed}
          isLive={isLive}
          isAtLiveEdge={isAtLiveEdge}
          timeBehindLive={timeBehindLive}
          onPlayPause={togglePlay}
          onSeek={handleSeek}
          onVolumeChange={handleVolumeChange}
          onMuteToggle={handleMuteToggle}
          onSpeedChange={setSpeed}
          onFullscreen={handleFullscreen}
          onReturnToLive={seekToLive}
        />
      )}

      {/* Mini timeline with events */}
      {showTimeline && (
        <MiniTimeline
          currentTime={currentTime}
          duration={duration}
          events={timelineEvents}
          isLive={isLive}
          liveWindowDuration={DVR_WINDOW_DURATION}
          onSeek={handleSeek}
          thumbnailData={thumbnailData}
        />
      )}
    </div>
  );
}

export default UnifiedVideoPlayer;
