/**
 * HLS video player component using hls.js.
 */

import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import './VideoPlayer.css';

interface VideoPlayerProps {
  src: string;
  cameraName: string;
  status: 'online' | 'offline' | 'unknown';
  isRecording?: boolean;
  onError?: (error: string) => void;
}

export function VideoPlayer({
  src,
  cameraName,
  status,
  isRecording = false,
  onError,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || status !== 'online') return;

    // Check if HLS is supported
    if (Hls.isSupported()) {
      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
        backBufferLength: 30,
        maxBufferLength: 10,
        maxMaxBufferLength: 30,
        liveSyncDurationCount: 3,
        liveMaxLatencyDurationCount: 6,
      });

      hlsRef.current = hls;
      hls.loadSource(src);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch((e) => {
          console.warn('Autoplay blocked:', e);
        });
      });

      hls.on(Hls.Events.ERROR, (_, data) => {
        if (data.fatal) {
          const errorMsg = `Stream error: ${data.type}`;
          setError(errorMsg);
          onError?.(errorMsg);

          // Try to recover
          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            hls.startLoad();
          } else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
            hls.recoverMediaError();
          }
        }
      });

      return () => {
        hls.destroy();
        hlsRef.current = null;
      };
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Native HLS support (Safari)
      video.src = src;
      video.addEventListener('loadedmetadata', () => {
        video.play().catch((e) => {
          console.warn('Autoplay blocked:', e);
        });
      });
    }
  }, [src, status, onError]);

  const getStatusClass = () => {
    if (status === 'offline') return 'status-offline';
    if (status === 'online') return 'status-online';
    return 'status-unknown';
  };

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
          <video
            ref={videoRef}
            muted
            playsInline
            className="video-element"
          />
        ) : (
          <div className="video-placeholder">
            <span className="placeholder-text">
              {status === 'offline' ? 'Camera Offline' : 'Connecting...'}
            </span>
          </div>
        )}

        {error && status === 'online' && (
          <div className="video-error">
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
