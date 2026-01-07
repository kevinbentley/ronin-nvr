/**
 * Thumbnail preview tooltip that shows when hovering over the timeline.
 */

import { useRef, useEffect } from 'react';
import type { ThumbnailPreviewProps } from './types';
import './ThumbnailPreview.css';

export function ThumbnailPreview({
  visible,
  spriteUrl,
  spriteX,
  spriteY,
  spriteWidth,
  spriteHeight,
  time,
  positionX,
}: ThumbnailPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Format time as HH:MM:SS or MM:SS
  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Adjust position to keep preview within viewport
  useEffect(() => {
    if (!containerRef.current || !visible) return;

    const container = containerRef.current;
    const rect = container.getBoundingClientRect();

    // Keep within viewport horizontally
    const viewportWidth = window.innerWidth;
    let adjustedX = positionX;

    if (rect.left < 0) {
      adjustedX = rect.width / 2;
    } else if (rect.right > viewportWidth) {
      adjustedX = viewportWidth - rect.width / 2;
    }

    container.style.left = `${adjustedX}px`;
  }, [visible, positionX]);

  if (!visible || !spriteUrl) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      className="thumbnail-preview"
      style={{
        left: `${positionX}px`,
      }}
    >
      <div
        className="thumbnail-preview-image"
        style={{
          width: `${spriteWidth}px`,
          height: `${spriteHeight}px`,
          backgroundImage: `url(${spriteUrl})`,
          backgroundPosition: `-${spriteX}px -${spriteY}px`,
          backgroundRepeat: 'no-repeat',
        }}
      />
      <div className="thumbnail-preview-time">
        {formatTime(time)}
      </div>
    </div>
  );
}
