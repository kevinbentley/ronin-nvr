/**
 * Hook for fetching detection events for the timeline.
 *
 * Supports two query modes for playback:
 * 1. By recording_id (historical detections with recording_id set)
 * 2. By camera_id + time range (live detections correlated by timestamp)
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../../../services/api';
import type { TimelineEvent } from '../types';

interface UseTimelineEventsOptions {
  recordingId?: number;
  cameraId?: number;
  cameraName?: string;
  recordingStartTime?: Date;
  recordingDuration?: number;
  duration: number;
  isLive: boolean;
  enabled?: boolean;
}

interface UseTimelineEventsReturn {
  events: TimelineEvent[];
  isLoading: boolean;
  error: string | null;
  refresh: () => void;
}

// Cache events to avoid refetching on every render
const eventsCache = new Map<string, { events: TimelineEvent[]; timestamp: number }>();
const CACHE_TTL_MS = 30000; // 30 seconds

export function useTimelineEvents({
  recordingId,
  cameraId,
  cameraName,
  recordingStartTime,
  recordingDuration,
  duration,
  isLive,
  enabled = true,
}: UseTimelineEventsOptions): UseTimelineEventsReturn {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchEvents = useCallback(async () => {
    if (!enabled) return;

    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    // Generate cache key
    const cacheKey = isLive
      ? `live-${cameraId}`
      : recordingId
        ? `recording-${recordingId}`
        : recordingStartTime
          ? `timerange-${cameraId}-${recordingStartTime.toISOString()}`
          : 'invalid';

    // Check cache
    const cached = eventsCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      setEvents(cached.events);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      let fetchedEvents: TimelineEvent[] = [];

      if (isLive && cameraName) {
        // For live: fetch recent events for the camera
        // Use current date and time range
        const now = new Date();
        const dateStr = now.toISOString().split('T')[0];

        const response = await api.getTimelineEvents({
          camera_name: cameraName,
          date: dateStr,
          min_confidence: 0.3,
        });

        // Filter to recent events (within DVR window)
        const dvrWindowMs = duration * 1000;
        const cutoffMs = now.getTime() - dvrWindowMs;

        fetchedEvents = (response.events || []).filter((event: TimelineEvent) => {
          // timestamp_ms is ms from midnight
          const eventDateMs = new Date(dateStr).getTime() + event.timestamp_ms;
          return eventDateMs >= cutoffMs;
        });
      } else if (cameraId && recordingStartTime) {
        // For playback: prefer time-range query since live detections use detected_at
        // (recording_id is often null for live detections)
        const startTime = recordingStartTime.toISOString();
        const durationSec = recordingDuration ?? 900; // Default 15 min
        const endTime = new Date(
          recordingStartTime.getTime() + durationSec * 1000
        ).toISOString();

        const response = await api.getDetections({
          camera_id: cameraId,
          start_time: startTime,
          end_time: endTime,
          limit: 500,
        });

        // Convert to timeline events with relative timestamps
        const recordingStartMs = recordingStartTime.getTime();
        fetchedEvents = (response.detections || []).map((d: {
          timestamp_ms: number;
          class_name: string;
          confidence: number;
          recording_id: number | null;
          detected_at?: string | null;
        }) => {
          // Calculate timestamp_ms relative to recording start
          const detectedAtMs = d.detected_at
            ? new Date(d.detected_at).getTime()
            : recordingStartMs;
          const relativeMs = Math.max(0, detectedAtMs - recordingStartMs);

          return {
            timestamp_ms: relativeMs,
            class_name: d.class_name,
            confidence: d.confidence,
            recording_id: d.recording_id ?? 0,
            count: 1,
          };
        });
      }

      // Update cache
      eventsCache.set(cacheKey, {
        events: fetchedEvents,
        timestamp: Date.now(),
      });

      // Debug logging
      console.log('[Timeline Events] Fetched', fetchedEvents.length, 'events for', cacheKey);
      if (fetchedEvents.length > 0) {
        console.log('[Timeline Events] First event:', fetchedEvents[0]);
      }

      setEvents(fetchedEvents);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        return; // Ignore abort errors
      }
      console.error('Failed to fetch timeline events:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch events');
    } finally {
      setIsLoading(false);
    }
  }, [enabled, isLive, cameraId, cameraName, recordingId, recordingStartTime, recordingDuration, duration]);

  // Fetch on mount and when dependencies change
  useEffect(() => {
    fetchEvents();

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchEvents]);

  // For live mode, refresh periodically
  useEffect(() => {
    if (!isLive || !enabled) return;

    const interval = setInterval(fetchEvents, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [isLive, enabled, fetchEvents]);

  const refresh = useCallback(() => {
    // Clear cache for this item
    const cacheKey = isLive
      ? `live-${cameraId}`
      : recordingId
        ? `recording-${recordingId}`
        : recordingStartTime
          ? `timerange-${cameraId}-${recordingStartTime.toISOString()}`
          : 'invalid';
    eventsCache.delete(cacheKey);
    fetchEvents();
  }, [isLive, cameraId, recordingId, recordingStartTime, fetchEvents]);

  return {
    events,
    isLoading,
    error,
    refresh,
  };
}
