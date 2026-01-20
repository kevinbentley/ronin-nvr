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
  const [isMuted, setIsMuted] = useState(true);

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

  const handleToggleMute = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsMuted(!isMuted);
  };

  return (
    <div className="video-player">
      <div className="video-container">
        {status === 'online' ? (
          <>
            <video
              ref={videoRef}
              muted={isMuted}
              playsInline
              className="video-element"
            />
            <button
              className="audio-toggle-button"
              onClick={handleToggleMute}
              title={isMuted ? 'Unmute' : 'Mute'}
            >
              {isMuted ? (
                <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                  <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                  <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                </svg>
              )}
            </button>
            <div className="video-info-overlay">
              {isRecording && <span className="recording-dot" />}
              <span className="camera-name-overlay">{cameraName}</span>
              <span className={`status-dot ${getStatusClass()}`} />
            </div>
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
          <>
            <div className="video-placeholder">
              <span className="placeholder-text">
                {status === 'offline' ? 'Camera Offline' : 'Connecting...'}
              </span>
            </div>
            <div className="video-info-overlay">
              {isRecording && <span className="recording-dot" />}
              <span className="camera-name-overlay">{cameraName}</span>
              <span className={`status-dot ${getStatusClass()}`} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
