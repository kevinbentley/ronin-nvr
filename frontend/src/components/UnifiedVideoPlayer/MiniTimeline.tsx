/**
 * Compact timeline component for the unified video player.
 * Shows playback progress and detection event markers.
 */

import { useRef, useState, useCallback, useMemo } from 'react';
import type { MiniTimelineProps } from './types';
import { getEventColor } from './types';
import { ThumbnailPreview } from './ThumbnailPreview';
import './MiniTimeline.css';

export function MiniTimeline({
  currentTime,
  duration,
  events,
  isLive,
  liveWindowDuration,
  onSeek,
  onHover,
  thumbnailData,
}: MiniTimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverX, setHoverX] = useState<number>(0);
  const [isDragging, setIsDragging] = useState(false);

  // For live mode, duration represents the DVR window
  const effectiveDuration = isLive ? liveWindowDuration : duration;

  // Calculate progress percentage
  const progressPercent = effectiveDuration > 0
    ? (currentTime / effectiveDuration) * 100
    : 0;

  // Map events to timeline positions
  const eventMarkers = useMemo(() => {
    if (effectiveDuration <= 0) return [];

    return events.map((event, index) => {
      // For playback: timestamp_ms is ms from recording start
      // For live: timestamp_ms might need adjustment based on DVR window
      const eventTimeSeconds = event.timestamp_ms / 1000;
      const positionPercent = (eventTimeSeconds / effectiveDuration) * 100;

      // Only show events within the visible range
      if (positionPercent < 0 || positionPercent > 100) return null;

      return {
        key: `${event.recording_id}-${event.timestamp_ms}-${event.class_name}-${index}`,
        event,
        left: `${positionPercent}%`,
        color: getEventColor(event.class_name),
        tooltip: `${event.class_name}${event.count > 1 ? ` (${event.count})` : ''}`,
      };
    }).filter(Boolean);
  }, [events, effectiveDuration]);

  // Handle mouse position to time conversion
  const getTimeFromMouseEvent = useCallback((e: React.MouseEvent | MouseEvent) => {
    const timeline = timelineRef.current;
    if (!timeline || effectiveDuration <= 0) return null;

    const rect = timeline.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = Math.max(0, Math.min(1, x / rect.width));
    return percent * effectiveDuration;
  }, [effectiveDuration]);

  // Handle click to seek
  const handleClick = useCallback((e: React.MouseEvent) => {
    const time = getTimeFromMouseEvent(e);
    if (time !== null) {
      onSeek(time);
    }
  }, [getTimeFromMouseEvent, onSeek]);

  // Handle mouse move for hover preview
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const time = getTimeFromMouseEvent(e);
    setHoverTime(time);
    onHover?.(time);

    // Track mouse X position for thumbnail preview positioning
    const timeline = timelineRef.current;
    if (timeline) {
      const rect = timeline.getBoundingClientRect();
      setHoverX(e.clientX - rect.left);
    }

    // If dragging, seek to position
    if (isDragging && time !== null) {
      onSeek(time);
    }
  }, [getTimeFromMouseEvent, onHover, isDragging, onSeek]);

  const handleMouseLeave = useCallback(() => {
    setHoverTime(null);
    onHover?.(null);
    setIsDragging(false);
  }, [onHover]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    const time = getTimeFromMouseEvent(e);
    if (time !== null) {
      onSeek(time);
    }
  }, [getTimeFromMouseEvent, onSeek]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Format time for display
  const formatTime = useCallback((seconds: number | null): string => {
    if (seconds === null || !isFinite(seconds)) return '';

    const absSeconds = Math.abs(seconds);
    const mins = Math.floor(absSeconds / 60);
    const secs = Math.floor(absSeconds % 60);

    if (isLive) {
      // Show as time behind live (negative)
      const behind = effectiveDuration - seconds;
      if (behind < 1) return 'LIVE';
      const behindMins = Math.floor(behind / 60);
      const behindSecs = Math.floor(behind % 60);
      return `-${behindMins}:${behindSecs.toString().padStart(2, '0')}`;
    }

    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, [isLive, effectiveDuration]);

  // Calculate hover position percent
  const hoverPercent = hoverTime !== null && effectiveDuration > 0
    ? (hoverTime / effectiveDuration) * 100
    : null;

  return (
    <div
      ref={timelineRef}
      className={`mini-timeline ${isDragging ? 'dragging' : ''}`}
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
    >
      {/* Progress track */}
      <div className="mini-timeline-track">
        {/* Buffered/available region would go here for live */}
        <div
          className="mini-timeline-progress"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Event markers */}
      <div className="mini-timeline-events">
        {eventMarkers.map((marker) => marker && (
          <div
            key={marker.key}
            className="mini-event-marker"
            style={{
              left: marker.left,
              backgroundColor: marker.color,
            }}
            title={marker.tooltip}
          />
        ))}
      </div>

      {/* Playhead */}
      <div
        className="mini-timeline-playhead"
        style={{ left: `${progressPercent}%` }}
      />

      {/* Hover indicator */}
      {hoverPercent !== null && (
        <div
          className="mini-timeline-hover"
          style={{ left: `${hoverPercent}%` }}
        >
          {/* Show hover time only if no thumbnail */}
          {!thumbnailData && (
            <div className="mini-hover-time">
              {formatTime(hoverTime)}
            </div>
          )}
        </div>
      )}

      {/* Thumbnail preview on hover */}
      {thumbnailData && hoverTime !== null && (() => {
        const sprite = thumbnailData.getThumbnailForTime(hoverTime);
        return sprite ? (
          <ThumbnailPreview
            visible={true}
            spriteUrl={thumbnailData.spriteUrl}
            spriteX={sprite.x}
            spriteY={sprite.y}
            spriteWidth={sprite.width}
            spriteHeight={sprite.height}
            time={hoverTime}
            positionX={hoverX}
          />
        ) : null;
      })()}

      {/* Time labels */}
      <div className="mini-timeline-labels">
        <span className="mini-time-start">{isLive ? `-${formatDuration(effectiveDuration)}` : '0:00'}</span>
        <span className="mini-time-end">{isLive ? 'LIVE' : formatDuration(duration)}</span>
      </div>
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return '0:00';

  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
