/**
 * Hook for managing HLS.js video player with DVR support.
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import Hls from 'hls.js';
import type { ConnectionState, PlayerMode } from '../types';

interface UseHlsPlayerOptions {
  mode: PlayerMode;
  src: string;
  cameraId: number;
  status?: 'online' | 'offline' | 'unknown';
  onTimeUpdate?: (time: number) => void;
}

interface UseHlsPlayerReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  hlsRef: React.RefObject<Hls | null>;
  connectionState: ConnectionState;
  errorMessage: string | null;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  bufferedEnd: number;
  isAtLiveEdge: boolean;
  timeBehindLive: number;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  seek: (time: number) => void;
  seekToLive: () => void;
  reconnect: () => void;
}

const MAX_AUTO_RETRIES = 5;
const RETRY_DELAY_MS = 3000;
const LIVE_EDGE_THRESHOLD_SECONDS = 10;

// DVR Configuration for 15-minute rewind
const DVR_HLS_CONFIG: Partial<Hls['config']> = {
  enableWorker: true,
  lowLatencyMode: false, // Disable for DVR stability
  backBufferLength: 900, // 15 minutes
  maxBufferLength: 60,
  maxMaxBufferLength: 120,
  liveSyncDurationCount: 3,
  liveMaxLatencyDurationCount: 10,
  manifestLoadPolicy: {
    default: {
      maxTimeToFirstByteMs: 15000,
      maxLoadTimeMs: 20000,
      timeoutRetry: { maxNumRetry: 3, retryDelayMs: 1000, maxRetryDelayMs: 4000 },
      errorRetry: { maxNumRetry: 3, retryDelayMs: 1000, maxRetryDelayMs: 4000 },
    },
  },
  fragLoadPolicy: {
    default: {
      maxTimeToFirstByteMs: 10000,
      maxLoadTimeMs: 20000,
      timeoutRetry: { maxNumRetry: 3, retryDelayMs: 500, maxRetryDelayMs: 2000 },
      errorRetry: { maxNumRetry: 3, retryDelayMs: 500, maxRetryDelayMs: 2000 },
    },
  },
};

export function useHlsPlayer({
  mode,
  src,
  cameraId,
  status = 'online',
  onTimeUpdate,
}: UseHlsPlayerOptions): UseHlsPlayerReturn {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hlsRef = useRef<Hls | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const retryCountRef = useRef(0);
  const stallCheckerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const directPlayerCleanupRef = useRef<(() => void) | null>(null);
  const hlsVideoCleanupRef = useRef<(() => void) | null>(null);
  const connectionStateRef = useRef<ConnectionState>('connecting');

  const [connectionState, setConnectionStateInternal] = useState<ConnectionState>('connecting');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Keep ref in sync with state for use in callbacks
  const setConnectionState = useCallback((state: ConnectionState) => {
    connectionStateRef.current = state;
    setConnectionStateInternal(state);
  }, []);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [bufferedEnd, setBufferedEnd] = useState(0);

  // Calculate live edge tracking
  const isAtLiveEdge = mode === 'live' && duration > 0 && (duration - currentTime) < LIVE_EDGE_THRESHOLD_SECONDS;
  const timeBehindLive = mode === 'live' && duration > 0 ? Math.max(0, duration - currentTime) : 0;

  const cleanup = useCallback(() => {
    if (stallCheckerRef.current) {
      clearInterval(stallCheckerRef.current);
      stallCheckerRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }
    if (hlsVideoCleanupRef.current) {
      hlsVideoCleanupRef.current();
      hlsVideoCleanupRef.current = null;
    }
    if (directPlayerCleanupRef.current) {
      directPlayerCleanupRef.current();
      directPlayerCleanupRef.current = null;
    }
  }, []);

  const initializeHlsPlayer = useCallback(async (_isRetry = false) => {
    const video = videoRef.current;
    if (!video || !mountedRef.current) return;

    cleanup();

    if (!mountedRef.current) return;
    setConnectionState('connecting');
    setErrorMessage(null);

    if (!Hls.isSupported()) {
      // Native HLS fallback (Safari)
      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = src;
        video.addEventListener('loadedmetadata', () => {
          if (mountedRef.current) {
            setConnectionState('connected');
            setDuration(video.duration);
            retryCountRef.current = 0;
          }
          video.play().catch((e) => console.warn('Autoplay blocked:', e));
        });
        video.addEventListener('error', () => {
          if (mountedRef.current) {
            setConnectionState('error');
            setErrorMessage('Playback error');
          }
        });
      }
      return;
    }

    const hls = new Hls(DVR_HLS_CONFIG);
    hlsRef.current = hls;
    hls.loadSource(src);
    hls.attachMedia(video);

    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      if (!mountedRef.current) return;
      setConnectionState('connected');
      setErrorMessage(null);
      retryCountRef.current = 0;
      video.play().catch((e) => console.warn('Autoplay blocked:', e));
    });

    hls.on(Hls.Events.LEVEL_LOADED, (_, data) => {
      if (!mountedRef.current) return;
      // Update duration from live playlist
      if (data.details.live) {
        const liveDuration = data.details.totalduration;
        setDuration(liveDuration);
      }
    });

    hls.on(Hls.Events.ERROR, (_, data) => {
      if (!mountedRef.current) return;
      console.warn('HLS error:', data.type, data.details, data.fatal);

      // Handle non-fatal errors gracefully (e.g., decode errors from corrupted segments)
      // These are transient and the stream usually recovers on its own
      if (!data.fatal) {
        // For decode errors, try to recover by seeking slightly
        if (data.details === 'fragParsingError' || data.type === Hls.ErrorTypes.MEDIA_ERROR) {
          console.log('Non-fatal decode error, attempting recovery...');
          // Seek forward slightly to skip corrupted segment
          if (video.currentTime > 0 && video.duration > 0) {
            const newTime = Math.min(video.currentTime + 2, video.duration - 1);
            video.currentTime = newTime;
          }
        }
        return;
      }

      // Fatal error handling
      if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
        console.log('Attempting media error recovery...');
        hls.recoverMediaError();
        return;
      }

      if (retryCountRef.current < MAX_AUTO_RETRIES) {
        retryCountRef.current++;
        setConnectionState('reconnecting');
        setErrorMessage(`Reconnecting... (${retryCountRef.current}/${MAX_AUTO_RETRIES})`);

        reconnectTimeoutRef.current = window.setTimeout(() => {
          if (mountedRef.current) {
            console.log(`Auto-reconnect attempt ${retryCountRef.current}`);
            initializeHlsPlayer(true);
          }
        }, RETRY_DELAY_MS);
      } else {
        setConnectionState('error');
        setErrorMessage('Connection lost. Click to reconnect.');
      }
    });

    // Video element event listeners
    const handleTimeUpdate = () => {
      if (!mountedRef.current) return;
      const time = video.currentTime;
      setCurrentTime(time);
      onTimeUpdate?.(time);
    };

    const handleDurationChange = () => {
      if (!mountedRef.current) return;
      setDuration(video.duration);
    };

    const handlePlay = () => {
      if (mountedRef.current) setIsPlaying(true);
    };

    const handlePause = () => {
      if (mountedRef.current) setIsPlaying(false);
    };

    const handleProgress = () => {
      if (!mountedRef.current) return;
      if (video.buffered.length > 0) {
        setBufferedEnd(video.buffered.end(video.buffered.length - 1));
      }
    };

    // Handle video element errors (decode errors from corrupted segments)
    const handleVideoError = () => {
      if (!mountedRef.current || !hlsRef.current) return;
      const error = video.error;

      // MEDIA_ERR_DECODE (3) often happens with corrupted segments
      // Try to recover by seeking forward and reloading
      if (error?.code === MediaError.MEDIA_ERR_DECODE) {
        console.warn('Video decode error, attempting recovery...');

        // Try HLS.js recovery first
        hlsRef.current.recoverMediaError();

        // Also seek forward slightly to skip corrupted segment
        if (video.currentTime > 0 && video.duration > 0) {
          const newTime = Math.min(video.currentTime + 2, video.duration - 1);
          video.currentTime = newTime;
        }

        // Don't show error state for recoverable errors
        return;
      }

      // For other errors, fall through to normal error handling
      console.error('Video error:', error?.code, error?.message);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('durationchange', handleDurationChange);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('progress', handleProgress);
    video.addEventListener('error', handleVideoError);

    // Store cleanup function for video element listeners
    hlsVideoCleanupRef.current = () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('durationchange', handleDurationChange);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('progress', handleProgress);
      video.removeEventListener('error', handleVideoError);
    };

    // Monitor for video stalls
    let lastTime = 0;
    let stallCount = 0;
    stallCheckerRef.current = window.setInterval(() => {
      if (!mountedRef.current || !hlsRef.current) return;

      // Use ref to avoid dependency on connectionState
      if (video.currentTime === lastTime && !video.paused && connectionStateRef.current === 'connected') {
        stallCount++;
        if (stallCount >= 5) {
          console.warn('Video stalled, attempting recovery');
          stallCount = 0;
          hlsRef.current?.startLoad();
        }
      } else {
        stallCount = 0;
      }
      lastTime = video.currentTime;
    }, 1000);
  }, [src, cleanup, onTimeUpdate]);

  const initializeDirectPlayer = useCallback(() => {
    const video = videoRef.current;
    if (!video || !mountedRef.current) {
      return;
    }

    cleanup();
    setConnectionState('connecting');
    setErrorMessage(null);
    setIsPlaying(false);  // Reset playing state when switching videos
    setCurrentTime(0);
    setDuration(0);

    const handleTimeUpdate = () => {
      if (!mountedRef.current) return;
      const time = video.currentTime;
      setCurrentTime(time);
      onTimeUpdate?.(time);
    };

    const handleDurationChange = () => {
      if (!mountedRef.current) return;
      setDuration(video.duration);
    };

    const handlePlay = () => {
      if (mountedRef.current) setIsPlaying(true);
    };

    const handlePause = () => {
      if (mountedRef.current) setIsPlaying(false);
    };

    const handleCanPlay = () => {
      if (mountedRef.current) {
        setConnectionState('connected');
        retryCountRef.current = 0;
      }
    };

    const handleError = (e: Event) => {
      if (mountedRef.current) {
        const videoEl = e.target as HTMLVideoElement;
        const errorMsg = videoEl.error?.message || 'Failed to load video';
        setConnectionState('error');
        setErrorMessage(errorMsg);
      }
    };

    const handleLoadedMetadata = () => {
      if (mountedRef.current) {
        setDuration(video.duration);
      }
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('durationchange', handleDurationChange);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('canplay', handleCanPlay);
    video.addEventListener('error', handleError);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);

    // Store cleanup function
    directPlayerCleanupRef.current = () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('durationchange', handleDurationChange);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('canplay', handleCanPlay);
      video.removeEventListener('error', handleError);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };

    video.src = src;
    video.load();
  }, [src, cleanup, onTimeUpdate]);

  // Initialize player based on mode
  useEffect(() => {
    mountedRef.current = true;

    if (mode === 'live') {
      if (status !== 'online') {
        cleanup();
        setConnectionState('connecting');
        setErrorMessage(null);
        return;
      }
      retryCountRef.current = 0;
      initializeHlsPlayer(false);
    } else {
      // For playback mode, we might need to wait for the video element to be mounted
      const initWithRetry = () => {
        if (!videoRef.current && mountedRef.current) {
          setTimeout(initWithRetry, 50);
          return;
        }
        initializeDirectPlayer();
      };
      initWithRetry();
    }

    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, [mode, src, status, cameraId, cleanup, initializeHlsPlayer, initializeDirectPlayer]);

  // Playback controls
  const play = useCallback(() => {
    videoRef.current?.play().catch((e) => console.warn('Play blocked:', e));
  }, []);

  const pause = useCallback(() => {
    videoRef.current?.pause();
  }, []);

  const togglePlay = useCallback(() => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }, [isPlaying, play, pause]);

  const seek = useCallback((time: number) => {
    const video = videoRef.current;
    if (video) {
      video.currentTime = time;
      setCurrentTime(time);
    }
  }, []);

  const seekToLive = useCallback(() => {
    const video = videoRef.current;
    if (video && mode === 'live') {
      // Seek to near the live edge
      const livePoint = duration - 3;
      if (livePoint > 0) {
        video.currentTime = livePoint;
        setCurrentTime(livePoint);
      }
    }
  }, [mode, duration]);

  const reconnect = useCallback(() => {
    retryCountRef.current = 0;
    setConnectionState('reconnecting');
    setErrorMessage('Reconnecting...');
    if (mode === 'live') {
      initializeHlsPlayer(true);
    } else {
      initializeDirectPlayer();
    }
  }, [mode, initializeHlsPlayer, initializeDirectPlayer]);

  return {
    videoRef,
    hlsRef,
    connectionState,
    errorMessage,
    isPlaying,
    currentTime,
    duration,
    bufferedEnd,
    isAtLiveEdge,
    timeBehindLive,
    play,
    pause,
    togglePlay,
    seek,
    seekToLive,
    reconnect,
  };
}
