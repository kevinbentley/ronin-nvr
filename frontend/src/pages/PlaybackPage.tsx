/**
 * Playback page for viewing recorded videos.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { DatePicker } from '../components/DatePicker';
import { Timeline } from '../components/Timeline';
import { RecordingPlayer } from '../components/RecordingPlayer';
import type { DayRecordings, RecordingFile, TimelineEvent } from '../types/camera';
import './PlaybackPage.css';

export function PlaybackPage() {
  const [cameras, setCameras] = useState<string[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [dayRecordings, setDayRecordings] = useState<DayRecordings | null>(null);
  const [selectedRecording, setSelectedRecording] = useState<RecordingFile | null>(null);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
  const [eventClassCounts, setEventClassCounts] = useState<Record<string, number>>({});
  const [selectedEventTypes, setSelectedEventTypes] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  // Load available dates when camera changes
  useEffect(() => {
    if (!selectedCamera) return;

    const loadDates = async () => {
      try {
        const dates = await api.getAvailableDates(selectedCamera);
        setAvailableDates(dates);
        if (dates.length > 0) {
          setSelectedDate(dates[0]); // Most recent date
        } else {
          setSelectedDate('');
          setDayRecordings(null);
        }
      } catch (err) {
        setError('Failed to load dates');
      }
    };
    loadDates();
  }, [selectedCamera]);

  // Load day recordings when date changes
  useEffect(() => {
    if (!selectedCamera || !selectedDate) {
      setDayRecordings(null);
      return;
    }

    const loadRecordings = async () => {
      try {
        const recordings = await api.getDayRecordings(selectedCamera, selectedDate);
        setDayRecordings(recordings);
        if (recordings.files.length > 0) {
          setSelectedRecording(recordings.files[0]);
        } else {
          setSelectedRecording(null);
        }
      } catch (err) {
        setDayRecordings(null);
        setSelectedRecording(null);
      }
    };
    loadRecordings();
  }, [selectedCamera, selectedDate]);

  // Load timeline events when camera/date changes
  useEffect(() => {
    if (!selectedCamera || !selectedDate) {
      setTimelineEvents([]);
      setEventClassCounts({});
      setSelectedEventTypes(new Set());
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
      } catch (err) {
        // Events are optional, don't show error
        setTimelineEvents([]);
        setEventClassCounts({});
        setSelectedEventTypes(new Set());
      }
    };
    loadEvents();
  }, [selectedCamera, selectedDate]);

  // Filter events based on selected types
  const filteredEvents = timelineEvents.filter(
    (event) => selectedEventTypes.has(event.class_name)
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

  const handleTimelineClick = useCallback((recording: RecordingFile) => {
    setSelectedRecording(recording);
  }, []);

  const handleEventClick = useCallback((event: TimelineEvent) => {
    // Find the recording that contains this event
    if (!dayRecordings) return;

    const targetRecording = dayRecordings.files.find(
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
            <div className="clip-info">
              <p>
                <strong>Time:</strong>{' '}
                {new Date(selectedRecording.start_time).toLocaleTimeString()}
              </p>
              <p>
                <strong>Duration:</strong>{' '}
                {selectedRecording.duration_seconds
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
      </div>

      <div className="playback-main">
        {error && (
          <div className="error-banner">{error}</div>
        )}

        <div className="player-container">
          {selectedRecording ? (
            <RecordingPlayer
              src={api.getRecordingStreamUrl(selectedRecording.id)}
              title={`${selectedCamera} - ${new Date(
                selectedRecording.start_time
              ).toLocaleTimeString()}`}
            />
          ) : (
            <div className="no-selection">
              <p>Select a recording from the timeline</p>
            </div>
          )}
        </div>

        <div className="timeline-container">
          {dayRecordings && (
            <Timeline
              recordings={dayRecordings.files}
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
