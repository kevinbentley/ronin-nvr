/**
 * Hook for fetching and parsing thumbnail sprite data.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../../../services/api';

interface ThumbnailSprite {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface ThumbnailData {
  spriteUrl: string;
  sprites: Map<number, ThumbnailSprite>; // timestamp (seconds) -> sprite coords
  intervalSeconds: number;
}

interface UseThumbnailsOptions {
  recordingId?: string;
  enabled?: boolean;
}

interface UseThumbnailsReturn {
  thumbnailData: ThumbnailData | null;
  isLoading: boolean;
  error: string | null;
  getThumbnailForTime: (time: number) => ThumbnailSprite | null;
}

/**
 * Parse a VTT file to extract thumbnail sprite coordinates.
 */
function parseVTT(vttContent: string): Map<number, ThumbnailSprite> {
  const sprites = new Map<number, ThumbnailSprite>();
  const lines = vttContent.split('\n');

  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trim();

    // Look for timestamp lines (e.g., "00:00:00.000 --> 00:00:10.000")
    if (line.includes('-->')) {
      const [startStr] = line.split('-->');
      const startSeconds = parseVTTTime(startStr.trim());

      // Next line should be the sprite URL with coordinates
      i++;
      if (i < lines.length) {
        const coordLine = lines[i].trim();
        // Parse "sprite.jpg#xywh=0,0,160,90"
        const match = coordLine.match(/#xywh=(\d+),(\d+),(\d+),(\d+)/);
        if (match) {
          sprites.set(startSeconds, {
            x: parseInt(match[1], 10),
            y: parseInt(match[2], 10),
            width: parseInt(match[3], 10),
            height: parseInt(match[4], 10),
          });
        }
      }
    }
    i++;
  }

  return sprites;
}

/**
 * Parse VTT timestamp to seconds.
 */
function parseVTTTime(timeStr: string): number {
  // Format: HH:MM:SS.mmm or MM:SS.mmm
  const parts = timeStr.split(':');
  let hours = 0;
  let minutes = 0;
  let seconds = 0;

  if (parts.length === 3) {
    hours = parseInt(parts[0], 10);
    minutes = parseInt(parts[1], 10);
    seconds = parseFloat(parts[2]);
  } else if (parts.length === 2) {
    minutes = parseInt(parts[0], 10);
    seconds = parseFloat(parts[1]);
  }

  return hours * 3600 + minutes * 60 + seconds;
}

export function useThumbnails({
  recordingId,
  enabled = true,
}: UseThumbnailsOptions): UseThumbnailsReturn {
  const [thumbnailData, setThumbnailData] = useState<ThumbnailData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const retryTimeoutRef = useRef<number | null>(null);

  const fetchThumbnails = useCallback(async () => {
    if (!enabled || !recordingId) {
      setThumbnailData(null);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // First, get thumbnail info
      const info = await api.getThumbnailInfo(recordingId);

      if (!info.available) {
        // Thumbnails not yet generated, retry in a few seconds
        if (retryTimeoutRef.current) {
          clearTimeout(retryTimeoutRef.current);
        }
        retryTimeoutRef.current = window.setTimeout(() => {
          fetchThumbnails();
        }, 5000);
        setIsLoading(false);
        return;
      }

      if (!info.vtt_url || !info.sprite_url) {
        setError('Thumbnail URLs not available');
        setIsLoading(false);
        return;
      }

      // Fetch the VTT file to parse sprite coordinates
      const vttResponse = await fetch(info.vtt_url);
      if (!vttResponse.ok) {
        throw new Error('Failed to fetch VTT file');
      }
      const vttContent = await vttResponse.text();
      const sprites = parseVTT(vttContent);

      setThumbnailData({
        spriteUrl: info.sprite_url,
        sprites,
        intervalSeconds: info.interval_seconds,
      });
    } catch (err) {
      console.error('Failed to fetch thumbnails:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch thumbnails');
    } finally {
      setIsLoading(false);
    }
  }, [enabled, recordingId]);

  useEffect(() => {
    fetchThumbnails();

    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, [fetchThumbnails]);

  const getThumbnailForTime = useCallback(
    (time: number): ThumbnailSprite | null => {
      if (!thumbnailData) return null;

      // Find the thumbnail for this time
      // Thumbnails are at fixed intervals, so round down to nearest interval
      const interval = thumbnailData.intervalSeconds;
      const snappedTime = Math.floor(time / interval) * interval;

      return thumbnailData.sprites.get(snappedTime) || null;
    },
    [thumbnailData]
  );

  return {
    thumbnailData,
    isLoading,
    error,
    getThumbnailForTime,
  };
}
