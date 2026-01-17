/**
 * Timeline scrubber showing recordings throughout a day.
 * Recording blocks are colored based on detection activity.
 */

import { useMemo } from 'react';
import type { RecordingFile, TimelineEvent } from '../types/camera';
import './Timeline.css';

/**
 * Parse hours and minutes from an ISO timestamp, converting UTC to local time.
 */
function parseTimeFromISO(isoString: string): { hours: number; minutes: number } {
  const date = new Date(isoString);
  return {
    hours: date.getHours(),
    minutes: date.getMinutes(),
  };
}

/**
 * Format an ISO timestamp as a local time string.
 */
function formatTimeFromISO(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

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

interface RecordingActivity {
  eventTypes: Map<string, number>; // className -> count
  totalEvents: number;
  dominantType: string | null;
  dominantColor: string | null;
}

interface TimelineProps {
  recordings: RecordingFile[];
  selectedRecording: RecordingFile | null;
  onSelectRecording: (recording: RecordingFile) => void;
  events?: TimelineEvent[];
}

export function Timeline({
  recordings,
  selectedRecording,
  onSelectRecording,
  events = [],
}: TimelineProps) {
  // Calculate timeline range (full 24 hours)
  const hours = useMemo(() => {
    return Array.from({ length: 24 }, (_, i) => i);
  }, []);

  // Map events to recordings based on time overlap
  const recordingActivity = useMemo(() => {
    const activityMap = new Map<string, RecordingActivity>();

    // Initialize all recordings with empty activity
    recordings.forEach((rec) => {
      activityMap.set(rec.id, {
        eventTypes: new Map(),
        totalEvents: 0,
        dominantType: null,
        dominantColor: null,
      });
    });

    // Group events by recording
    events.forEach((event) => {
      // timestamp_ms is milliseconds from start of day (local time)
      const eventMinutes = event.timestamp_ms / 60000;
      const eventHours = Math.floor(eventMinutes / 60);
      const eventMins = eventMinutes % 60;

      // Find the recording that contains this event time
      const matchingRecording = recordings.find((rec) => {
        const { hours: startHour, minutes: startMinute } = parseTimeFromISO(rec.start_time);
        const recStartMinutes = startHour * 60 + startMinute;
        const durationMinutes = rec.duration_seconds ? rec.duration_seconds / 60 : 15;
        const recEndMinutes = recStartMinutes + durationMinutes;

        const eventTotalMinutes = eventHours * 60 + eventMins;
        return eventTotalMinutes >= recStartMinutes && eventTotalMinutes < recEndMinutes;
      });

      if (matchingRecording) {
        const activity = activityMap.get(matchingRecording.id)!;
        const currentCount = activity.eventTypes.get(event.class_name) || 0;
        activity.eventTypes.set(event.class_name, currentCount + event.count);
        activity.totalEvents += event.count;
      }
    });

    // Calculate dominant type for each recording
    activityMap.forEach((activity) => {
      if (activity.eventTypes.size > 0) {
        let maxCount = 0;
        let dominant: string | null = null;

        activity.eventTypes.forEach((count, className) => {
          if (count > maxCount) {
            maxCount = count;
            dominant = className;
          }
        });

        activity.dominantType = dominant;
        activity.dominantColor = dominant ? getEventColor(dominant) : null;
      }
    });

    return activityMap;
  }, [recordings, events]);

  // Map recordings to timeline positions
  const recordingBlocks = useMemo(() => {
    return recordings.map((rec) => {
      const { hours: startHour, minutes: startMinute } = parseTimeFromISO(rec.start_time);
      const durationMinutes = rec.duration_seconds ? rec.duration_seconds / 60 : 15;

      // Calculate position as percentage of day
      const startPercent = ((startHour * 60 + startMinute) / (24 * 60)) * 100;
      const widthPercent = (durationMinutes / (24 * 60)) * 100;

      const activity = recordingActivity.get(rec.id);
      const hasActivity = activity && activity.totalEvents > 0;

      // Build tooltip with activity summary
      let tooltip = `${formatTimeFromISO(rec.start_time)} - ${
        rec.is_in_progress
          ? 'Recording in progress...'
          : rec.duration_seconds
            ? `${Math.floor(rec.duration_seconds / 60)} min`
            : 'Unknown duration'
      }`;

      if (hasActivity && activity) {
        const eventSummary = Array.from(activity.eventTypes.entries())
          .sort((a, b) => b[1] - a[1])
          .map(([type, count]) => `${type}: ${count}`)
          .join(', ');
        tooltip += `\n${activity.totalEvents} events (${eventSummary})`;
      }

      return {
        recording: rec,
        left: `${startPercent}%`,
        width: `${Math.max(widthPercent, 0.5)}%`,
        hasActivity,
        activity,
        tooltip,
      };
    });
  }, [recordings, recordingActivity]);

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
        {recordingBlocks.map(({ recording, left, width, hasActivity, activity, tooltip }) => (
          <button
            key={recording.id}
            className={`recording-block ${
              selectedRecording?.id === recording.id ? 'selected' : ''
            } ${recording.is_in_progress ? 'in-progress' : ''} ${
              hasActivity ? 'has-activity' : ''
            }`}
            style={{
              left,
              width,
              ...(hasActivity && activity?.dominantColor && !recording.is_in_progress
                ? { '--activity-color': activity.dominantColor } as React.CSSProperties
                : {}),
            }}
            onClick={() => onSelectRecording(recording)}
            title={tooltip}
          >
            {/* Activity indicator dots for multiple event types */}
            {hasActivity && activity && activity.eventTypes.size > 1 && (
              <div className="activity-dots">
                {Array.from(activity.eventTypes.keys())
                  .slice(0, 4) // Max 4 dots
                  .map((className) => (
                    <span
                      key={className}
                      className="activity-dot"
                      style={{ backgroundColor: getEventColor(className) }}
                    />
                  ))}
              </div>
            )}
          </button>
        ))}
      </div>

      <div className="timeline-hours">
        {hours.map((hour) => (
          <div key={hour} className="hour-tick" />
        ))}
      </div>
    </div>
  );
}
