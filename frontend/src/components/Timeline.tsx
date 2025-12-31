/**
 * Timeline scrubber showing recordings and detection events throughout a day.
 */

import { useMemo } from 'react';
import type { RecordingFile, TimelineEvent } from '../types/camera';
import './Timeline.css';

// Event type colors for visual differentiation
const EVENT_COLORS: Record<string, string> = {
  motion: '#ff5722',     // Deep orange for motion
  person: '#4caf50',     // Green for person
  car: '#2196f3',        // Blue for vehicles
  truck: '#2196f3',
  bus: '#2196f3',
  motorcycle: '#2196f3',
  bicycle: '#00bcd4',    // Cyan for bicycle
  dog: '#9c27b0',        // Purple for animals
  cat: '#9c27b0',
  bird: '#9c27b0',
  default: '#ffc107',    // Amber for unknown
};

function getEventColor(className: string): string {
  return EVENT_COLORS[className.toLowerCase()] || EVENT_COLORS.default;
}

interface TimelineProps {
  recordings: RecordingFile[];
  selectedRecording: RecordingFile | null;
  onSelectRecording: (recording: RecordingFile) => void;
  events?: TimelineEvent[];
  onEventClick?: (event: TimelineEvent) => void;
}

export function Timeline({
  recordings,
  selectedRecording,
  onSelectRecording,
  events = [],
  onEventClick,
}: TimelineProps) {
  // Calculate timeline range (full 24 hours)
  const hours = useMemo(() => {
    return Array.from({ length: 24 }, (_, i) => i);
  }, []);

  // Map recordings to timeline positions
  const recordingBlocks = useMemo(() => {
    return recordings.map((rec) => {
      const startTime = new Date(rec.start_time);
      const startHour = startTime.getHours();
      const startMinute = startTime.getMinutes();
      const durationMinutes = rec.duration_seconds ? rec.duration_seconds / 60 : 15;

      // Calculate position as percentage of day
      const startPercent = ((startHour * 60 + startMinute) / (24 * 60)) * 100;
      const widthPercent = (durationMinutes / (24 * 60)) * 100;

      return {
        recording: rec,
        left: `${startPercent}%`,
        width: `${Math.max(widthPercent, 0.5)}%`, // Minimum width for visibility
      };
    });
  }, [recordings]);

  // Map events to timeline positions
  const eventMarkers = useMemo(() => {
    return events.map((event) => {
      // timestamp_ms is milliseconds from start of day
      const totalMinutes = event.timestamp_ms / 60000;
      const positionPercent = (totalMinutes / (24 * 60)) * 100;

      // Format time for tooltip
      const hours = Math.floor(totalMinutes / 60);
      const minutes = Math.floor(totalMinutes % 60);
      const timeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;

      return {
        event,
        left: `${positionPercent}%`,
        color: getEventColor(event.class_name),
        tooltip: `${event.class_name} (${event.count}) at ${timeStr}`,
      };
    });
  }, [events]);

  const formatHour = (hour: number) => {
    if (hour === 0) return '12a';
    if (hour === 12) return '12p';
    if (hour < 12) return `${hour}a`;
    return `${hour - 12}p`;
  };

  return (
    <div className="timeline">
      <div className="timeline-header">
        {hours.map((hour) => (
          <div key={hour} className="hour-marker">
            {hour % 3 === 0 && <span className="hour-label">{formatHour(hour)}</span>}
          </div>
        ))}
      </div>

      <div className="timeline-track">
        {recordingBlocks.map(({ recording, left, width }) => (
          <button
            key={recording.id}
            className={`recording-block ${
              selectedRecording?.id === recording.id ? 'selected' : ''
            }`}
            style={{ left, width }}
            onClick={() => onSelectRecording(recording)}
            title={`${new Date(recording.start_time).toLocaleTimeString()} - ${
              recording.duration_seconds
                ? `${Math.floor(recording.duration_seconds / 60)} min`
                : 'Unknown duration'
            }`}
          />
        ))}
      </div>

      {events.length > 0 && (
        <div className="timeline-events-track">
          {eventMarkers.map(({ event, left, color, tooltip }, index) => (
            <button
              key={`${event.recording_id}-${event.timestamp_ms}-${event.class_name}-${index}`}
              className="event-marker"
              style={{ left, backgroundColor: color }}
              onClick={() => onEventClick?.(event)}
              title={tooltip}
            />
          ))}
        </div>
      )}

      <div className="timeline-hours">
        {hours.map((hour) => (
          <div key={hour} className="hour-tick" />
        ))}
      </div>
    </div>
  );
}
