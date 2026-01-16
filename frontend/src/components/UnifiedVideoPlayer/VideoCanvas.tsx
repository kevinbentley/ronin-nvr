/**
 * Canvas overlay component for drawing bounding boxes on video.
 * Handles letterboxing alignment when video uses object-fit: contain.
 */

import { useRef, useEffect, useCallback } from 'react';
import type { VideoCanvasProps } from './types';
import { getEventColor } from './types';
import type { Detection } from '../../types/camera';
import './VideoCanvas.css';

type DetectionWithBbox = Detection;

interface VideoDisplayInfo {
  // The actual rendered video area within the container
  offsetX: number;
  offsetY: number;
  displayWidth: number;
  displayHeight: number;
}

/**
 * Calculate the actual display area of a video with object-fit: contain.
 * This accounts for letterboxing (black bars) around the video.
 */
function getVideoDisplayInfo(video: HTMLVideoElement): VideoDisplayInfo {
  const containerWidth = video.clientWidth;
  const containerHeight = video.clientHeight;
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  if (videoWidth === 0 || videoHeight === 0) {
    return { offsetX: 0, offsetY: 0, displayWidth: 0, displayHeight: 0 };
  }

  const containerAspect = containerWidth / containerHeight;
  const videoAspect = videoWidth / videoHeight;

  let displayWidth: number;
  let displayHeight: number;
  let offsetX: number;
  let offsetY: number;

  if (videoAspect > containerAspect) {
    // Video is wider than container - letterbox top/bottom
    displayWidth = containerWidth;
    displayHeight = containerWidth / videoAspect;
    offsetX = 0;
    offsetY = (containerHeight - displayHeight) / 2;
  } else {
    // Video is taller than container - letterbox left/right
    displayHeight = containerHeight;
    displayWidth = containerHeight * videoAspect;
    offsetX = (containerWidth - displayWidth) / 2;
    offsetY = 0;
  }

  return { offsetX, offsetY, displayWidth, displayHeight };
}

export function VideoCanvas({ videoRef, detections, visible }: VideoCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Draw a single bounding box with label
  const drawBoundingBox = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      detection: DetectionWithBbox,
      displayInfo: VideoDisplayInfo
    ) => {
      const { bbox_x, bbox_y, bbox_width, bbox_height, class_name, confidence } = detection;
      const { offsetX, offsetY, displayWidth, displayHeight } = displayInfo;

      // Denormalize coordinates relative to the actual video display area
      const x = offsetX + bbox_x * displayWidth;
      const y = offsetY + bbox_y * displayHeight;
      const width = bbox_width * displayWidth;
      const height = bbox_height * displayHeight;

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
      const finalLabelY = labelY < offsetY ? y : labelY;

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

    // Skip if video hasn't loaded yet
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    // Get the container dimensions (where video is displayed)
    const containerWidth = video.clientWidth;
    const containerHeight = video.clientHeight;

    if (containerWidth === 0 || containerHeight === 0) return;

    // Update canvas size to match container
    if (canvas.width !== containerWidth || canvas.height !== containerHeight) {
      canvas.width = containerWidth;
      canvas.height = containerHeight;
    }

    // Calculate video display area (accounting for object-fit: contain)
    const displayInfo = getVideoDisplayInfo(video);

    // Clear previous frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all detection bounding boxes
    for (const detection of detections) {
      const det = detection as unknown as DetectionWithBbox;
      if (det.bbox_x !== undefined && det.bbox_y !== undefined) {
        drawBoundingBox(ctx, det, displayInfo);
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
