/**
 * HLS video player component using hls.js with auto-reconnection.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import Hls from 'hls.js';
import { api } from '../services/api';
import './VideoPlayer.css';

interface VideoPlayerProps {
  src: string;
  cameraId: number;
  cameraName: string;
  status: 'online' | 'offline' | 'unknown';
  isRecording?: boolean;
  onError?: (error: string) => void;
}

type ConnectionState = 'connecting' | 'connected' | 'error' | 'reconnecting';

export function VideoPlayer({
  src,
  cameraId,
  cameraName,
  status,
  isRecording = false,
  onError,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const retryCountRef = useRef(0);

  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');
  const [error, setError] = useState<string | null>(null);

  const MAX_AUTO_RETRIES = 5;
  const RETRY_DELAY_MS = 3000;

  const destroyHls = useCallback(() => {
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const restartStream = useCallback(async () => {
    try {
      // Stop and restart the stream on the backend
      await api.stopStream(cameraId).catch(() => {});
      await new Promise((resolve) => setTimeout(resolve, 500));
      await api.startStream(cameraId);
      return true;
    } catch (err) {
      console.error('Failed to restart stream:', err);
      return false;
    }
  }, [cameraId]);

  const initializeHls = useCallback(() => {
    const video = videoRef.current;
    if (!video || status !== 'online') return;

    destroyHls();
    setConnectionState('connecting');
    setError(null);

    if (!Hls.isSupported()) {
      // Native HLS support (Safari)
      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = src;
        video.addEventListener('loadedmetadata', () => {
          setConnectionState('connected');
          video.play().catch((e) => console.warn('Autoplay blocked:', e));
        });
        video.addEventListener('error', () => {
          setConnectionState('error');
          setError('Playback error');
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
      setConnectionState('connected');
      setError(null);
      retryCountRef.current = 0;
      video.play().catch((e) => console.warn('Autoplay blocked:', e));
    });

    hls.on(Hls.Events.ERROR, (_, data) => {
      console.warn('HLS error:', data.type, data.details, data.fatal);

      if (data.fatal) {
        setError(`Stream error: ${data.details}`);
        onError?.(`Stream error: ${data.details}`);

        // Try built-in recovery first
        if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
          console.log('Attempting media error recovery...');
          hls.recoverMediaError();
          return;
        }

        // For network errors, attempt auto-reconnect
        if (retryCountRef.current < MAX_AUTO_RETRIES) {
          retryCountRef.current++;
          setConnectionState('reconnecting');
          setError(`Reconnecting... (${retryCountRef.current}/${MAX_AUTO_RETRIES})`);

          reconnectTimeoutRef.current = window.setTimeout(async () => {
            console.log(`Auto-reconnect attempt ${retryCountRef.current}`);
            await restartStream();
            initializeHls();
          }, RETRY_DELAY_MS);
        } else {
          setConnectionState('error');
          setError('Connection lost. Click to reconnect.');
        }
      }
    });

    // Monitor for stalls
    let lastTime = 0;
    let stallCount = 0;
    const stallChecker = setInterval(() => {
      if (video.currentTime === lastTime && !video.paused && connectionState === 'connected') {
        stallCount++;
        if (stallCount >= 5) {
          console.warn('Video stalled, attempting recovery');
          stallCount = 0;
          if (hlsRef.current) {
            hlsRef.current.startLoad();
          }
        }
      } else {
        stallCount = 0;
      }
      lastTime = video.currentTime;
    }, 1000);

    return () => {
      clearInterval(stallChecker);
    };
  }, [src, status, onError, destroyHls, restartStream]);

  useEffect(() => {
    if (status !== 'online') {
      destroyHls();
      setConnectionState('connecting');
      return;
    }

    const cleanup = initializeHls();
    return () => {
      cleanup?.();
      destroyHls();
    };
  }, [src, status, initializeHls, destroyHls]);

  const handleManualReconnect = async () => {
    retryCountRef.current = 0;
    setConnectionState('reconnecting');
    setError('Reconnecting...');
    await restartStream();
    initializeHls();
  };

  const getStatusClass = () => {
    if (status === 'offline') return 'status-offline';
    if (connectionState === 'connected') return 'status-online';
    if (connectionState === 'error') return 'status-error';
    return 'status-connecting';
  };

  const showReconnectButton = connectionState === 'error' ||
    (connectionState === 'reconnecting' && retryCountRef.current >= MAX_AUTO_RETRIES);

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
                <span className="loading-text">{error || 'Reconnecting...'}</span>
              </div>
            )}
            {showReconnectButton && (
              <div className="video-overlay error-overlay" onClick={handleManualReconnect}>
                <span className="error-text">{error}</span>
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
