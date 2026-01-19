/**
 * ML Status page showing activity log and live detections.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type {
  LiveDetection,
  ObjectEvent,
} from '../types/camera';
import './MLStatusPage.css';

const DETECTIONS_PER_PAGE = 100;
const EVENTS_PER_PAGE = 100;

export function MLStatusPage() {
  const [recentDetections, setRecentDetections] = useState<LiveDetection[]>([]);
  const [detectionsTotal, setDetectionsTotal] = useState(0);
  const [detectionsPage, setDetectionsPage] = useState(0);
  const [selectedDetection, setSelectedDetection] = useState<LiveDetection | null>(null);
  const [zoomedImage, setZoomedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Object events state
  const [objectEvents, setObjectEvents] = useState<ObjectEvent[]>([]);
  const [eventsTotal, setEventsTotal] = useState(0);
  const [eventsPage, setEventsPage] = useState(0);
  const [selectedEvent, setSelectedEvent] = useState<ObjectEvent | null>(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [detections, events] = await Promise.all([
        api.getLiveDetections({
          limit: DETECTIONS_PER_PAGE,
          offset: detectionsPage * DETECTIONS_PER_PAGE,
        }),
        api.getObjectEvents({
          limit: EVENTS_PER_PAGE,
          offset: eventsPage * EVENTS_PER_PAGE,
        }),
      ]);
      setRecentDetections(detections.detections);
      setDetectionsTotal(detections.total);
      setObjectEvents(events.events);
      setEventsTotal(events.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load ML status');
    } finally {
      setLoading(false);
    }
  }, [detectionsPage, eventsPage]);

  useEffect(() => {
    loadData();
    // Refresh every 5 seconds for real-time updates
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

  const formatDetectionTime = (dateStr: string | null): string => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleString(undefined, {
      month: 'numeric',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  const formatEventType = (eventType: string): string => {
    const labels: Record<string, string> = {
      ARRIVAL: 'Arrived',
      DEPARTURE: 'Departed',
      STATE_CHANGE: 'State Change',
      LOITERING: 'Loitering',
    };
    return labels[eventType] || eventType;
  };

  const getEventTypeClass = (eventType: string): string => {
    const classes: Record<string, string> = {
      ARRIVAL: 'event-arrival',
      DEPARTURE: 'event-departure',
      STATE_CHANGE: 'event-state-change',
      LOITERING: 'event-loitering',
    };
    return classes[eventType] || '';
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  };

  // Close lightbox on Escape key
  useEffect(() => {
    if (!zoomedImage) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setZoomedImage(null);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [zoomedImage]);

  if (loading) {
    return (
      <div className="ml-status-page loading">
        <div className="loading-spinner">Loading ML status...</div>
      </div>
    );
  }

  return (
    <div className="ml-status-page">
      <div className="ml-status-header">
        <h2>ML Detection</h2>
        <div className="ml-controls">
          <button className="refresh-button" onClick={loadData}>
            Refresh
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {/* Activity Log Row */}
      <div className="ml-row">
        {/* Object Events Log */}
        <div className="ml-status-card events-card">
          <div className="events-header">
            <h3>Activity Log ({eventsTotal} total)</h3>
            {eventsTotal > EVENTS_PER_PAGE && (
              <div className="pagination-controls2">
                <button
                  className="pagination-button2"
                  onClick={() => setEventsPage((p) => Math.max(0, p - 1))}
                  disabled={eventsPage === 0}
                >
                  Previous
                </button>
                <span className="pagination-info">
                  Page {eventsPage + 1} of {Math.ceil(eventsTotal / EVENTS_PER_PAGE)}
                </span>
                <button
                  className="pagination-button2"
                  onClick={() =>
                    setEventsPage((p) =>
                      Math.min(Math.ceil(eventsTotal / EVENTS_PER_PAGE) - 1, p + 1)
                    )
                  }
                  disabled={eventsPage >= Math.ceil(eventsTotal / EVENTS_PER_PAGE) - 1}
                >
                  Next
                </button>
              </div>
            )}
          </div>
          {objectEvents.length > 0 ? (
            <div className="events-list">
              {objectEvents.map((event) => (
                <div
                  key={event.id}
                  className={`event-item ${getEventTypeClass(event.event_type)} ${selectedEvent?.id === event.id ? 'selected' : ''}`}
                  onClick={() => setSelectedEvent(event)}
                >
                  <div className="event-summary">
                    <span className={`event-type ${getEventTypeClass(event.event_type)}`}>
                      {formatEventType(event.event_type)}
                    </span>
                    <span className="event-class">{event.class_name}</span>
                    <span className="event-camera">{event.camera_name}</span>
                  </div>
                  <div className="event-details">
                    <span className="event-time">{formatDetectionTime(event.event_time)}</span>
                    {event.duration_seconds > 0 && (
                      <span className="event-duration">({formatDuration(event.duration_seconds)})</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="no-data">No events recorded</p>
          )}
        </div>

        {/* Event Snapshot Preview */}
        <div className="ml-status-card snapshot-preview-card">
          <h3>Event Snapshot</h3>
          {selectedEvent ? (
            <div className="snapshot-preview">
              {selectedEvent.snapshot_url ? (
                <img
                  src={selectedEvent.snapshot_url}
                  alt={`Event: ${selectedEvent.event_type} - ${selectedEvent.class_name}`}
                  className="snapshot-image clickable"
                  onClick={() => setZoomedImage(selectedEvent.snapshot_url)}
                  title="Click to enlarge"
                />
              ) : (
                <div className="no-snapshot">No snapshot available</div>
              )}
              <div className="snapshot-info">
                <div className="snapshot-detail">
                  <span className="snapshot-label">Camera</span>
                  <span className="snapshot-value">{selectedEvent.camera_name}</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Event</span>
                  <span className="snapshot-value">{formatEventType(selectedEvent.event_type)}</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Object</span>
                  <span className="snapshot-value">{selectedEvent.class_name}</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Confidence</span>
                  <span className="snapshot-value">{(selectedEvent.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Time</span>
                  <span className="snapshot-value">{formatDetectionTime(selectedEvent.event_time)}</span>
                </div>
              </div>
            </div>
          ) : (
            <p className="no-data">Select an event to view snapshot</p>
          )}
        </div>
      </div>

      {/* Live Detections Row */}
      <div className="ml-row">
        {/* Recent Live Detections */}
        <div className="ml-status-card detections-card">
          <div className="detections-header">
            <h3>Live Detections ({detectionsTotal} total)</h3>
            {detectionsTotal > DETECTIONS_PER_PAGE && (
              <div className="pagination-controls">
                <button
                  className="pagination-button"
                  onClick={() => setDetectionsPage((p) => Math.max(0, p - 1))}
                  disabled={detectionsPage === 0}
                >
                  Previous
                </button>
                <span className="pagination-info">
                  Page {detectionsPage + 1} of{' '}
                  {Math.ceil(detectionsTotal / DETECTIONS_PER_PAGE)}
                </span>
                <button
                  className="pagination-button"
                  onClick={() =>
                    setDetectionsPage((p) =>
                      Math.min(
                        Math.ceil(detectionsTotal / DETECTIONS_PER_PAGE) - 1,
                        p + 1
                      )
                    )
                  }
                  disabled={
                    detectionsPage >=
                    Math.ceil(detectionsTotal / DETECTIONS_PER_PAGE) - 1
                  }
                >
                  Next
                </button>
              </div>
            )}
          </div>
          {recentDetections.length > 0 ? (
            <div className="recent-detections">
              {recentDetections.map((det) => (
                <div
                  key={det.id}
                  className={`detection-item ${selectedDetection?.id === det.id ? 'selected' : ''}`}
                  onClick={() => setSelectedDetection(det)}
                >
                  <div className="detection-summary">
                    <span className="detection-camera">{det.camera_name}</span>
                    <span className="detection-timestamp">
                      {formatDetectionTime(det.detected_at)}
                    </span>
                  </div>
                  <div className="detection-details">
                    <span className="detection-class">{det.class_name}</span>
                    <span className="detection-confidence">
                      ({(det.confidence * 100).toFixed(0)}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="no-data">No recent detections</p>
          )}
        </div>

        {/* Detection Snapshot Preview */}
        <div className="ml-status-card snapshot-preview-card">
          <h3>Detection Snapshot</h3>
          {selectedDetection ? (
            <div className="snapshot-preview">
              {selectedDetection.snapshot_url ? (
                <img
                  src={selectedDetection.snapshot_url}
                  alt={`Detection: ${selectedDetection.class_name}`}
                  className="snapshot-image clickable"
                  onClick={() => setZoomedImage(selectedDetection.snapshot_url)}
                  title="Click to enlarge"
                />
              ) : (
                <div className="no-snapshot">No snapshot available</div>
              )}
              <div className="snapshot-info">
                <div className="snapshot-detail">
                  <span className="snapshot-label">Camera</span>
                  <span className="snapshot-value">{selectedDetection.camera_name}</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Detected</span>
                  <span className="snapshot-value">{selectedDetection.class_name}</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Confidence</span>
                  <span className="snapshot-value">{(selectedDetection.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Time</span>
                  <span className="snapshot-value">{formatDetectionTime(selectedDetection.detected_at)}</span>
                </div>
                {selectedDetection.concern_level && (
                  <div className="snapshot-detail">
                    <span className="snapshot-label">Concern</span>
                    <span className={`snapshot-value concern-level concern-${selectedDetection.concern_level}`}>
                      {selectedDetection.concern_level.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>
              {selectedDetection.llm_description && (
                <div className="llm-analysis">
                  <div className="llm-description">{selectedDetection.llm_description}</div>
                  {selectedDetection.activity_type && (
                    <div className="llm-activity-type">Activity: {selectedDetection.activity_type}</div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <p className="no-data">Select a detection to view snapshot</p>
          )}
        </div>
      </div>

      {/* Image Lightbox */}
      {zoomedImage && (
        <div className="image-lightbox" onClick={() => setZoomedImage(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <img src={zoomedImage} alt="Zoomed detection snapshot" />
            <button
              className="lightbox-close"
              onClick={() => setZoomedImage(null)}
              aria-label="Close"
            >
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
