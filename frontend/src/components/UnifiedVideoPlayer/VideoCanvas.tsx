/**
 * Canvas overlay component for drawing bounding boxes on video.
 */

import { useRef, useEffect, useCallback } from 'react';
import type { VideoCanvasProps } from './types';
import { getEventColor } from './types';
import type { Detection } from '../../types/camera';
import './VideoCanvas.css';

// Use Detection type from camera.ts which has the correct fields
type DetectionWithBbox = Detection;

export function VideoCanvas({ videoRef, detections, visible }: VideoCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Draw a single bounding box with label
  const drawBoundingBox = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      detection: DetectionWithBbox,
      videoWidth: number,
      videoHeight: number
    ) => {
      const { bbox_x, bbox_y, bbox_width, bbox_height, class_name, confidence } = detection;

      // Denormalize coordinates (bbox values are normalized 0-1)
      const x = bbox_x * videoWidth;
      const y = bbox_y * videoHeight;
      const width = bbox_width * videoWidth;
      const height = bbox_height * videoHeight;

      const color = getEventColor(class_name);

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Draw semi-transparent fill
      ctx.fillStyle = `${color}20`; // 12.5% opacity
      ctx.fillRect(x, y, width, height);

      // Draw label background
      const label = `${class_name} ${Math.round(confidence * 100)}%`;
      ctx.font = 'bold 12px sans-serif';
      const textMetrics = ctx.measureText(label);
      const textHeight = 16;
      const padding = 4;

      const labelX = x;
      const labelY = y - textHeight - padding;
      const labelWidth = textMetrics.width + padding * 2;
      const labelHeight = textHeight + padding;

      // Position label inside box if it would go off screen
      const finalLabelY = labelY < 0 ? y : labelY;

      ctx.fillStyle = color;
      ctx.fillRect(labelX, finalLabelY, labelWidth, labelHeight);

      // Draw label text
      ctx.fillStyle = '#fff';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, labelX + padding, finalLabelY + labelHeight / 2);
    },
    []
  );

  // Main render function
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video || !visible) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get video dimensions
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    // Skip if video hasn't loaded yet
    if (videoWidth === 0 || videoHeight === 0) return;

    // Update canvas size to match video
    if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
      canvas.width = videoWidth;
      canvas.height = videoHeight;
    }

    // Clear previous frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all detection bounding boxes
    for (const detection of detections) {
      // Cast to our type that has bbox fields
      const det = detection as unknown as DetectionWithBbox;
      if (det.bbox_x !== undefined && det.bbox_y !== undefined) {
        drawBoundingBox(ctx, det, videoWidth, videoHeight);
      }
    }
  }, [videoRef, detections, visible, drawBoundingBox]);

  // Render on each animation frame when visible
  useEffect(() => {
    if (!visible) {
      // Clear canvas when hidden
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
      return;
    }

    const animate = () => {
      render();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    // Start animation loop
    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [visible, render]);

  // Also re-render when detections change
  useEffect(() => {
    if (visible) {
      render();
    }
  }, [detections, visible, render]);

  if (!visible) return null;

  return (
    <canvas
      ref={canvasRef}
      className="video-canvas-overlay"
      aria-hidden="true"
    />
  );
}
