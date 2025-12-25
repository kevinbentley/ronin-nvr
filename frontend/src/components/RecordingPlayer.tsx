/**
 * Video player for recorded files with seek functionality.
 */

import { useRef, useEffect, useState } from 'react';
import './RecordingPlayer.css';

interface RecordingPlayerProps {
  src: string;
  title?: string;
}

export function RecordingPlayer({ src, title }: RecordingPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(true);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handleDurationChange = () => setDuration(video.duration);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('durationchange', handleDurationChange);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('durationchange', handleDurationChange);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, []);

  // Reset when source changes
  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.load();
      setCurrentTime(0);
      setIsPlaying(false);
    }
  }, [src]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const time = parseFloat(e.target.value);
    video.currentTime = time;
    setCurrentTime(time);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const vol = parseFloat(e.target.value);
    video.volume = vol;
    setVolume(vol);
    setIsMuted(vol === 0);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isMuted) {
      video.muted = false;
      setIsMuted(false);
    } else {
      video.muted = true;
      setIsMuted(true);
    }
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const toggleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;

    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      video.requestFullscreen();
    }
  };

  return (
    <div className="recording-player">
      {title && <div className="player-title">{title}</div>}

      <video
        ref={videoRef}
        src={src}
        className="player-video"
        muted={isMuted}
        playsInline
        onClick={togglePlay}
      />

      <div className="player-controls">
        <button className="control-button play-button" onClick={togglePlay}>
          {isPlaying ? '⏸' : '▶'}
        </button>

        <span className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        <input
          type="range"
          className="seek-bar"
          min={0}
          max={duration || 0}
          value={currentTime}
          onChange={handleSeek}
          step={0.1}
        />

        <button className="control-button mute-button" onClick={toggleMute}>
          {isMuted ? '🔇' : '🔊'}
        </button>

        <input
          type="range"
          className="volume-bar"
          min={0}
          max={1}
          value={isMuted ? 0 : volume}
          onChange={handleVolumeChange}
          step={0.1}
        />

        <button className="control-button fullscreen-button" onClick={toggleFullscreen}>
          ⛶
        </button>
      </div>
    </div>
  );
}
