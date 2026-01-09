/**
 * Playback page for viewing recorded videos.
 *
 * Date handling: The backend stores all timestamps in UTC. This component
 * converts UTC timestamps to local dates/times for display using the browser's
 * timezone. Available dates are derived client-side by grouping recordings
 * by their local date.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { api } from '../services/api';
import { DatePicker } from '../components/DatePicker';
import { Timeline } from '../components/Timeline';
import { UnifiedVideoPlayer } from '../components/UnifiedVideoPlayer';
import type { RecordingFile, TimelineEvent, EventSource } from '../types/camera';
import './PlaybackPage.css';

/** Display labels for event sources */
const EVENT_SOURCE_LABELS: Record<EventSource, string> = {
  ml: 'ML Detection',
  onvif_motion: 'Camera Motion',
  onvif_analytics: 'Camera Analytics',
};

/**
 * Format an ISO timestamp string as a local time string.
 */
function formatRecordingTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString();
}

/**
 * Get the local date string (YYYY-MM-DD) for a UTC timestamp.
 */
function getLocalDateString(isoString: string): string {
  const date = new Date(isoString);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

/**
 * Get today's date string in local timezone.
 */
function getTodayString(): string {
  return getLocalDateString(new Date().toISOString());
}

export function PlaybackPage() {
  const [cameras, setCameras] = useState<string[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [allRecordings, setAllRecordings] = useState<RecordingFile[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [selectedRecording, setSelectedRecording] = useState<RecordingFile | null>(null);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
  const [eventClassCounts, setEventClassCounts] = useState<Record<string, number>>({});
  const [selectedEventTypes, setSelectedEventTypes] = useState<Set<string>>(new Set());
  const [selectedEventSources, setSelectedEventSources] = useState<Set<EventSource>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Compute event source counts from events
  const eventSourceCounts = useMemo(() => {
    const counts: Record<EventSource, number> = {
      ml: 0,
      onvif_motion: 0,
      onvif_analytics: 0,
    };
    timelineEvents.forEach((event) => {
      const source = event.event_source || 'ml';
      counts[source] = (counts[source] || 0) + 1;
    });
    // Filter out sources with no events
    return Object.fromEntries(
      Object.entries(counts).filter(([, count]) => count > 0)
    ) as Record<EventSource, number>;
  }, [timelineEvents]);

  // Derive available dates from recordings (grouped by local date)
  const availableDates = useMemo(() => {
    const dateSet = new Set<string>();
    allRecordings.forEach((rec) => {
      const localDate = getLocalDateString(rec.start_time);
      dateSet.add(localDate);
    });
    // Sort descending (most recent first)
    return Array.from(dateSet).sort((a, b) => b.localeCompare(a));
  }, [allRecordings]);

  // Filter recordings for the selected local date
  const dayRecordings = useMemo(() => {
    if (!selectedDate) return [];
    return allRecordings
      .filter((rec) => getLocalDateString(rec.start_time) === selectedDate)
      .sort((a, b) => a.start_time.localeCompare(b.start_time));
  }, [allRecordings, selectedDate]);

  // Load cameras with recordings
  useEffect(() => {
    const loadCameras = async () => {
      try {
        const cams = await api.getCamerasWithRecordings();
        setCameras(cams);
        if (cams.length > 0 && !selectedCamera) {
          setSelectedCamera(cams[0]);
        }
      } catch (err) {
        setError('Failed to load cameras');
      } finally {
        setLoading(false);
      }
    };
    loadCameras();
  }, []);

  // Load all recordings when camera changes
  useEffect(() => {
    if (!selectedCamera) {
      setAllRecordings([]);
      return;
    }

    const loadRecordings = async () => {
      try {
        // Fetch recordings for this camera, paginating if necessary
        const allRecs: RecordingFile[] = [];
        let offset = 0;
        const pageSize = 1000;
        let hasMore = true;

        while (hasMore) {
          const result = await api.listRecordings({
            camera_name: selectedCamera,
            limit: pageSize,
            offset,
          });
          allRecs.push(...result.recordings);
          offset += pageSize;
          hasMore = result.recordings.length === pageSize && offset < result.total;
        }

        setAllRecordings(allRecs);
      } catch (err) {
        setError('Failed to load recordings');
        setAllRecordings([]);
      }
    };
    loadRecordings();
  }, [selectedCamera]);

  // Auto-select the most recent date when available dates change
  useEffect(() => {
    if (availableDates.length > 0 && !availableDates.includes(selectedDate)) {
      // Prefer today if available, otherwise most recent
      const today = getTodayString();
      if (availableDates.includes(today)) {
        setSelectedDate(today);
      } else {
        setSelectedDate(availableDates[0]);
      }
    } else if (availableDates.length === 0) {
      setSelectedDate('');
    }
  }, [availableDates]);

  // Auto-select first recording when day recordings change
  useEffect(() => {
    if (dayRecordings.length > 0) {
      // Select the first recording of the day
      setSelectedRecording(dayRecordings[0]);
    } else {
      setSelectedRecording(null);
    }
  }, [dayRecordings]);

  // Load timeline events when camera/date changes
  useEffect(() => {
    if (!selectedCamera || !selectedDate) {
      setTimelineEvents([]);
      setEventClassCounts({});
      setSelectedEventTypes(new Set());
      setSelectedEventSources(new Set());
      return;
    }

    const loadEvents = async () => {
      try {
        const response = await api.getTimelineEvents({
          camera_name: selectedCamera,
          date: selectedDate,
        });
        setTimelineEvents(response.events);
        setEventClassCounts(response.class_counts);
        // Select all event types by default
        setSelectedEventTypes(new Set(Object.keys(response.class_counts)));
        // Select all event sources by default
        const sources = new Set<EventSource>();
        response.events.forEach((event) => {
          sources.add(event.event_source || 'ml');
        });
        setSelectedEventSources(sources);
      } catch (err) {
        // Events are optional, don't show error
        setTimelineEvents([]);
        setEventClassCounts({});
        setSelectedEventTypes(new Set());
        setSelectedEventSources(new Set());
      }
    };
    loadEvents();
  }, [selectedCamera, selectedDate]);

  // Filter events based on selected types and sources
  const filteredEvents = timelineEvents.filter(
    (event) =>
      selectedEventTypes.has(event.class_name) &&
      selectedEventSources.has(event.event_source || 'ml')
  );

  const handleToggleEventType = (eventType: string) => {
    setSelectedEventTypes((prev) => {
      const next = new Set(prev);
      if (next.has(eventType)) {
        next.delete(eventType);
      } else {
        next.add(eventType);
      }
      return next;
    });
  };

  const handleToggleEventSource = (source: EventSource) => {
    setSelectedEventSources((prev) => {
      const next = new Set(prev);
      if (next.has(source)) {
        next.delete(source);
      } else {
        next.add(source);
      }
      return next;
    });
  };

  const handleTimelineClick = useCallback((recording: RecordingFile) => {
    setSelectedRecording(recording);
  }, []);

  const handleEventClick = useCallback((event: TimelineEvent) => {
    // Find the recording that contains this event
    const targetRecording = dayRecordings.find(
      (rec) => rec.id === String(event.recording_id)
    );

    if (targetRecording) {
      setSelectedRecording(targetRecording);
      // Future: could also seek to the specific time within the recording
    }
  }, [dayRecordings]);

  const handleDownload = useCallback(() => {
    if (selectedRecording) {
      window.open(api.getRecordingDownloadUrl(selectedRecording.id), '_blank');
    }
  }, [selectedRecording]);

  if (loading) {
    return (
      <div className="playback-page loading">
        <div className="loading-spinner">Loading...</div>
      </div>
    );
  }

  if (cameras.length === 0) {
    return (
      <div className="playback-page empty">
        <div className="no-recordings">
          <h2>No Recordings Available</h2>
          <p>Start recording cameras to view playback.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="playback-page">
      <div className="playback-sidebar">
        <div className="sidebar-section">
          <h3>Camera</h3>
          <select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            className="camera-select"
          >
            {cameras.map((cam) => (
              <option key={cam} value={cam}>
                {cam}
              </option>
            ))}
          </select>
        </div>

        <div className="sidebar-section">
          <h3>Date</h3>
          <DatePicker
            availableDates={availableDates}
            selectedDate={selectedDate}
            onSelectDate={setSelectedDate}
          />
        </div>

        {selectedRecording && (
          <div className="sidebar-section">
            <h3>Current Clip</h3>
            {selectedRecording.is_in_progress && (
              <div className="in-progress-badge">Recording in progress...</div>
            )}
            <div className="clip-info">
              <p>
                <strong>Time:</strong>{' '}
                {formatRecordingTime(selectedRecording.start_time)}
              </p>
              <p>
                <strong>Duration:</strong>{' '}
                {selectedRecording.is_in_progress
                  ? 'Recording...'
                  : selectedRecording.duration_seconds
                    ? `${Math.floor(selectedRecording.duration_seconds / 60)}m`
                    : 'Unknown'}
              </p>
              <p>
                <strong>Size:</strong>{' '}
                {(selectedRecording.size_bytes / (1024 * 1024)).toFixed(1)} MB
              </p>
            </div>
            <button className="download-button" onClick={handleDownload}>
              Download Clip
            </button>
          </div>
        )}

        {Object.keys(eventClassCounts).length > 0 && (
          <div className="sidebar-section">
            <h3>Events Filter</h3>
            <div className="event-filters">
              {Object.entries(eventClassCounts)
                .sort((a, b) => b[1] - a[1]) // Sort by count descending
                .map(([eventType, count]) => (
                  <label key={eventType} className="event-filter-item">
                    <input
                      type="checkbox"
                      checked={selectedEventTypes.has(eventType)}
                      onChange={() => handleToggleEventType(eventType)}
                    />
                    <span className="event-type-name">{eventType}</span>
                    <span className="event-type-count">({count})</span>
                  </label>
                ))}
            </div>
          </div>
        )}

        {Object.keys(eventSourceCounts).length > 1 && (
          <div className="sidebar-section">
            <h3>Event Source</h3>
            <div className="event-filters">
              {(Object.entries(eventSourceCounts) as [EventSource, number][])
                .sort((a, b) => b[1] - a[1]) // Sort by count descending
                .map(([source, count]) => (
                  <label key={source} className="event-filter-item">
                    <input
                      type="checkbox"
                      checked={selectedEventSources.has(source)}
                      onChange={() => handleToggleEventSource(source)}
                    />
                    <span className="event-type-name">{EVENT_SOURCE_LABELS[source]}</span>
                    <span className="event-type-count">({count})</span>
                  </label>
                ))}
            </div>
          </div>
        )}
      </div>

      <div className="playback-main">
        {error && (
          <div className="error-banner">{error}</div>
        )}

        <div className="player-container">
          {selectedRecording ? (
            <UnifiedVideoPlayer
              mode="playback"
              src={api.getRecordingStreamUrl(selectedRecording.id)}
              cameraId={0}
              cameraName={`${selectedCamera} - ${formatRecordingTime(selectedRecording.start_time)}`}
              recordingIdString={selectedRecording.id}
              recordingStartTime={new Date(selectedRecording.start_time)}
              showControls={true}
              showTimeline={true}
            />
          ) : (
            <div className="no-selection">
              <p>Select a recording from the timeline</p>
            </div>
          )}
        </div>

        <div className="timeline-container">
          {dayRecordings.length > 0 && (
            <Timeline
              recordings={dayRecordings}
              selectedRecording={selectedRecording}
              onSelectRecording={handleTimelineClick}
              events={filteredEvents}
              onEventClick={handleEventClick}
            />
          )}
        </div>
      </div>
    </div>
  );
}
