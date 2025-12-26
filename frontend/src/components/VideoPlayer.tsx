/**
 * HLS video player component using hls.js with auto-reconnection.
 */

import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import { api } from '../services/api';
import './VideoPlayer.css';

interface VideoPlayerProps {
  src: string;
  cameraId: number;
  cameraName: string;
  status: 'online' | 'offline' | 'unknown';
  isRecording?: boolean;
}

type ConnectionState = 'connecting' | 'connected' | 'error' | 'reconnecting';

const MAX_AUTO_RETRIES = 5;
const RETRY_DELAY_MS = 3000;

export function VideoPlayer({
  src,
  cameraId,
  cameraName,
  status,
  isRecording = false,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const retryCountRef = useRef(0);
  const stallCheckerRef = useRef<number | null>(null);
  const connectionStateRef = useRef<ConnectionState>('connecting');
  const mountedRef = useRef(true);

  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Keep ref in sync with state
  useEffect(() => {
    connectionStateRef.current = connectionState;
  }, [connectionState]);

  // Track mounted state
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const cleanup = () => {
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
  };

  const restartBackendStream = async (forceRestart: boolean = false) => {
    try {
      // Only stop/restart the backend stream if explicitly requested (manual reconnect)
      // For auto-retry, just let HLS.js reload - the backend stream is likely still running
      if (forceRestart) {
        await api.stopStream(cameraId).catch(() => {});
        await new Promise((resolve) => setTimeout(resolve, 500));
        await api.startStream(cameraId);
      }
      return true;
    } catch (err) {
      console.error('Failed to restart stream:', err);
      return false;
    }
  };

  const initializePlayer = async (isRetry: boolean = false, forceRestart: boolean = false) => {
    const video = videoRef.current;
    if (!video || !mountedRef.current) return;

    cleanup();

    if (!mountedRef.current) return;
    setConnectionState('connecting');
    setErrorMessage(null);

    if (isRetry) {
      // Only force restart backend stream on manual reconnect, not auto-retry
      await restartBackendStream(forceRestart);
      if (!mountedRef.current) return;
    }

    if (!Hls.isSupported()) {
      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = src;
        video.addEventListener('loadedmetadata', () => {
          if (mountedRef.current) {
            setConnectionState('connected');
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

    const hls = new Hls({
      enableWorker: true,
      lowLatencyMode: true,
      backBufferLength: 30,
      maxBufferLength: 10,
      maxMaxBufferLength: 30,
      liveSyncDurationCount: 3,
      liveMaxLatencyDurationCount: 6,
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
    });

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

    hls.on(Hls.Events.ERROR, (_, data) => {
      if (!mountedRef.current) return;
      console.warn('HLS error:', data.type, data.details, data.fatal);

      if (data.fatal) {
        // Try built-in recovery first for media errors
        if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
          console.log('Attempting media error recovery...');
          hls.recoverMediaError();
          return;
        }

        // For network errors, attempt auto-reconnect
        if (retryCountRef.current < MAX_AUTO_RETRIES) {
          retryCountRef.current++;
          setConnectionState('reconnecting');
          setErrorMessage(`Reconnecting... (${retryCountRef.current}/${MAX_AUTO_RETRIES})`);

          reconnectTimeoutRef.current = window.setTimeout(() => {
            if (mountedRef.current) {
              console.log(`Auto-reconnect attempt ${retryCountRef.current}`);
              initializePlayer(true);
            }
          }, RETRY_DELAY_MS);
        } else {
          setConnectionState('error');
          setErrorMessage('Connection lost. Click to reconnect.');
        }
      }
    });

    // Monitor for video stalls
    let lastTime = 0;
    let stallCount = 0;
    stallCheckerRef.current = window.setInterval(() => {
      if (!mountedRef.current || !hlsRef.current) return;

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
  };

  // Initialize on mount and when src/status changes
  useEffect(() => {
    if (status !== 'online') {
      cleanup();
      setConnectionState('connecting');
      setErrorMessage(null);
      return;
    }

    retryCountRef.current = 0;
    initializePlayer(false);

    return () => {
      cleanup();
    };
    // Only reinitialize when src or status actually changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, status, cameraId]);

  const handleManualReconnect = () => {
    retryCountRef.current = 0;
    setConnectionState('reconnecting');
    setErrorMessage('Reconnecting...');
    // Force restart the backend stream on manual reconnect
    initializePlayer(true, true);
  };

  const getStatusClass = () => {
    if (status === 'offline') return 'status-offline';
    if (connectionState === 'connected') return 'status-online';
    if (connectionState === 'error') return 'status-error';
    return 'status-connecting';
  };

  const showReconnectButton = connectionState === 'error';

  return (
    <div className="video-player">
      <div className="video-header">
        <span className="camera-name">{cameraName}</span>
        <div className="status-indicators">
          {isRecording && <span className="recording-indicator">REC</span>}
          <span className={`status-dot ${getStatusClass()}`} />
        </div>
      </div>

      <div className="video-container">
        {status === 'online' ? (
          <>
            <video
              ref={videoRef}
              muted
              playsInline
              className="video-element"
            />
            {connectionState === 'connecting' && (
              <div className="video-overlay">
                <span className="loading-text">Connecting...</span>
              </div>
            )}
            {connectionState === 'reconnecting' && (
              <div className="video-overlay">
                <span className="loading-text">{errorMessage || 'Reconnecting...'}</span>
              </div>
            )}
            {showReconnectButton && (
              <div className="video-overlay error-overlay" onClick={handleManualReconnect}>
                <span className="error-text">{errorMessage}</span>
                <button className="reconnect-button">Reconnect</button>
              </div>
            )}
          </>
        ) : (
          <div className="video-placeholder">
            <span className="placeholder-text">
              {status === 'offline' ? 'Camera Offline' : 'Connecting...'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
