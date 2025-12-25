/**
 * Timeline scrubber showing recordings throughout a day.
 */

import { useMemo } from 'react';
import type { RecordingFile } from '../types/camera';
import './Timeline.css';

interface TimelineProps {
  recordings: RecordingFile[];
  selectedRecording: RecordingFile | null;
  onSelectRecording: (recording: RecordingFile) => void;
}

export function Timeline({
  recordings,
  selectedRecording,
  onSelectRecording,
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

      <div className="timeline-hours">
        {hours.map((hour) => (
          <div key={hour} className="hour-tick" />
        ))}
      </div>
    </div>
  );
}
