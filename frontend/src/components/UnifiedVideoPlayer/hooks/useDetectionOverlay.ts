/**
 * Hook for fetching and managing detection data for bounding box overlay.
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { api } from '../../../services/api';
import type { Detection } from '../../../types/camera';

interface UseDetectionOverlayOptions {
  recordingId?: number;
  cameraId?: number;
  currentTime: number;
  duration: number;
  isLive: boolean;
  enabled: boolean;
  visibleObjectTypes?: Set<string>;
}

interface UseDetectionOverlayReturn {
  detections: Detection[];
  typeCounts: Map<string, number>;
  isLoading: boolean;
  error: string | null;
}

// Cache detections for the recording to avoid repeated API calls
const detectionsCache = new Map<number, Detection[]>();

// Time window around current playback position to fetch (in ms)
const FETCH_WINDOW_MS = 2000; // +/- 2 seconds

// Debounce interval for fetching (ms)
const DEBOUNCE_MS = 300;

export function useDetectionOverlay({
  recordingId,
  cameraId: _cameraId, // Reserved for future live detection support
  currentTime,
  duration: _duration, // Reserved for future use
  isLive,
  enabled,
  visibleObjectTypes,
}: UseDetectionOverlayOptions): UseDetectionOverlayReturn {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [allDetections, setAllDetections] = useState<Detection[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceTimeoutRef = useRef<number | null>(null);

  // Compute type counts from all detections
  const typeCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const d of allDetections) {
      const className = d.class_name.toLowerCase();
      counts.set(className, (counts.get(className) || 0) + 1);
    }
    return counts;
  }, [allDetections]);

  // Fetch all detections for the recording (once)
  const fetchAllDetections = useCallback(async () => {
    if (!recordingId || isLive) return;

    // Check cache first
    const cached = detectionsCache.get(recordingId);
    if (cached) {
      setAllDetections(cached);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.getDetections({
        recording_id: recordingId,
        limit: 5000, // Get all detections for the recording
      });

      const fetchedDetections = response.detections || [];
      detectionsCache.set(recordingId, fetchedDetections);
      setAllDetections(fetchedDetections);
    } catch (err) {
      console.error('Failed to fetch detections:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch detections');
    } finally {
      setIsLoading(false);
    }
  }, [recordingId, isLive]);

  // Fetch detections on recording change
  useEffect(() => {
    if (!enabled || !recordingId) {
      setAllDetections([]);
      setDetections([]);
      return;
    }

    fetchAllDetections();
  }, [enabled, recordingId, fetchAllDetections]);

  // Filter detections for current time and visible types (with debounce)
  useEffect(() => {
    if (!enabled || allDetections.length === 0) {
      setDetections([]);
      return;
    }

    // Clear previous timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    // Debounce the filtering
    debounceTimeoutRef.current = window.setTimeout(() => {
      const currentTimeMs = currentTime * 1000;

      // Find detections within the time window
      let windowDetections = allDetections.filter((d) => {
        const diff = Math.abs(d.timestamp_ms - currentTimeMs);
        return diff <= FETCH_WINDOW_MS;
      });

      // Filter by visible object types if specified
      if (visibleObjectTypes && visibleObjectTypes.size > 0) {
        windowDetections = windowDetections.filter((d) =>
          visibleObjectTypes.has(d.class_name.toLowerCase())
        );
      }

      // Limit to top 20 detections by confidence to avoid performance issues
      const sortedDetections = windowDetections
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 20);

      setDetections(sortedDetections);
    }, DEBOUNCE_MS);

    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, [enabled, allDetections, currentTime, visibleObjectTypes]);

  return {
    detections,
    typeCounts,
    isLoading,
    error,
  };
}

/**
 * Clear the detection cache for a specific recording or all recordings.
 */
export function clearDetectionCache(recordingId?: number): void {
  if (recordingId !== undefined) {
    detectionsCache.delete(recordingId);
  } else {
    detectionsCache.clear();
  }
}
