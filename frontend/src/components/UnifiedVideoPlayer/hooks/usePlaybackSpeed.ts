/**
 * Hook for managing video playback speed with persistence.
 */

import { useState, useCallback, useEffect } from 'react';
import { PLAYBACK_SPEEDS } from '../types';

const STORAGE_KEY = 'unified-player-speed';

interface UsePlaybackSpeedOptions {
  initialSpeed?: number;
}

interface UsePlaybackSpeedReturn {
  speed: number;
  setSpeed: (speed: number) => void;
  cycleSpeed: () => void;
  availableSpeeds: number[];
}

export function usePlaybackSpeed({
  initialSpeed,
}: UsePlaybackSpeedOptions = {}): UsePlaybackSpeedReturn {

  // Load from localStorage or use initial/default
  const getInitialSpeed = (): number => {
    if (initialSpeed !== undefined) return initialSpeed;

    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = parseFloat(stored);
        if (PLAYBACK_SPEEDS.includes(parsed)) {
          return parsed;
        }
      }
    } catch {
      // Ignore localStorage errors
    }
    return 1;
  };

  const [speed, setSpeedState] = useState<number>(getInitialSpeed);

  // Persist to localStorage when speed changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, speed.toString());
    } catch {
      // Ignore localStorage errors
    }
  }, [speed]);

  const setSpeed = useCallback((newSpeed: number) => {
    // Validate speed is in allowed list
    if (!PLAYBACK_SPEEDS.includes(newSpeed)) {
      console.warn(`Invalid playback speed: ${newSpeed}`);
      return;
    }
    setSpeedState(newSpeed);
  }, []);

  const cycleSpeed = useCallback(() => {
    const currentIndex = PLAYBACK_SPEEDS.indexOf(speed);
    const nextIndex = (currentIndex + 1) % PLAYBACK_SPEEDS.length;
    setSpeedState(PLAYBACK_SPEEDS[nextIndex]);
  }, [speed]);

  return {
    speed,
    setSpeed,
    cycleSpeed,
    availableSpeeds: PLAYBACK_SPEEDS,
  };
}

/**
 * Apply playback speed to a video element.
 * Call this in a useEffect when speed or videoRef changes.
 */
export function applyPlaybackSpeed(
  video: HTMLVideoElement | null,
  speed: number
): void {
  if (video) {
    video.playbackRate = speed;

    // Mute audio at very high speeds (browser behavior is inconsistent)
    if (speed >= 4) {
      video.muted = true;
    }
  }
}
