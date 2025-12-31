/**
 * Camera grid component with configurable layout and zoom functionality.
 */

import { useState, useEffect, useCallback } from 'react';
import { VideoPlayer } from './VideoPlayer';
import { api } from '../services/api';
import type { Camera, RecordingStatus, GridLayout } from '../types/camera';
import './CameraGrid.css';

interface CameraGridProps {
  cameras: Camera[];
  recordingStatus: Map<number, RecordingStatus>;
  layout: GridLayout;
}

const layoutConfig: Record<GridLayout, { columns: number; rows: number }> = {
  '1x1': { columns: 1, rows: 1 },
  '2x2': { columns: 2, rows: 2 },
  '3x3': { columns: 3, rows: 3 },
  '4x4': { columns: 4, rows: 4 },
};

export function CameraGrid({ cameras, recordingStatus, layout }: CameraGridProps) {
  const [zoomedCameraId, setZoomedCameraId] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(0);

  const { columns, rows } = layoutConfig[layout];
  const camerasPerPage = columns * rows;
  const totalPages = Math.max(1, Math.ceil(cameras.length / camerasPerPage));

  // Reset to first page when layout changes or camera count reduces
  useEffect(() => {
    if (currentPage >= totalPages) {
      setCurrentPage(Math.max(0, totalPages - 1));
    }
  }, [currentPage, totalPages]);

  const startIndex = currentPage * camerasPerPage;
  const displayCameras = cameras.slice(startIndex, startIndex + camerasPerPage);

  // Fill empty slots on the current page
  const emptySlots = camerasPerPage - displayCameras.length;

  const handlePrevPage = useCallback(() => {
    setCurrentPage((p) => Math.max(0, p - 1));
  }, []);

  const handleNextPage = useCallback(() => {
    setCurrentPage((p) => Math.min(totalPages - 1, p + 1));
  }, [totalPages]);

  // Handle Escape key to exit zoom
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape' && zoomedCameraId !== null) {
      setZoomedCameraId(null);
    }
  }, [zoomedCameraId]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const handleCameraClick = (cameraId: number) => {
    if (zoomedCameraId === cameraId) {
      // Already zoomed on this camera, exit zoom
      setZoomedCameraId(null);
    } else {
      // Zoom to this camera
      setZoomedCameraId(cameraId);
    }
  };

  // If a camera is zoomed, show only that camera full-screen
  if (zoomedCameraId !== null) {
    const zoomedCamera = cameras.find((c) => c.id === zoomedCameraId);
    if (zoomedCamera) {
      const status = recordingStatus.get(zoomedCamera.id);
      return (
        <div className="camera-grid-container">
          <div className="camera-grid zoomed">
            <div
              className="grid-cell zoomed-cell"
              onClick={() => setZoomedCameraId(null)}
            >
              <VideoPlayer
                src={api.getStreamUrl(zoomedCamera.id)}
                cameraId={zoomedCamera.id}
                cameraName={zoomedCamera.name}
                status={zoomedCamera.status}
                isRecording={status?.is_recording}
              />
              <button
                className="zoom-exit-button"
                onClick={(e) => {
                  e.stopPropagation();
                  setZoomedCameraId(null);
                }}
                title="Exit zoom (Esc)"
              >
                &times;
              </button>
            </div>
          </div>
        </div>
      );
    }
  }

  return (
    <div className="camera-grid-container">
      <div
        className="camera-grid"
        style={{
          gridTemplateColumns: `repeat(${columns}, 1fr)`,
          gridTemplateRows: `repeat(${rows}, 1fr)`,
        }}
      >
        {displayCameras.map((camera) => {
          const status = recordingStatus.get(camera.id);
          return (
            <div
              key={camera.id}
              className="grid-cell clickable"
              onClick={() => handleCameraClick(camera.id)}
              title="Click to zoom"
            >
              <VideoPlayer
                src={api.getStreamUrl(camera.id)}
                cameraId={camera.id}
                cameraName={camera.name}
                status={camera.status}
                isRecording={status?.is_recording}
              />
            </div>
          );
        })}

        {Array.from({ length: emptySlots }).map((_, index) => (
          <div key={`empty-${index}`} className="grid-cell empty">
            <div className="empty-slot">
              <span>No Camera</span>
            </div>
          </div>
        ))}
      </div>

      {totalPages > 1 && (
        <div className="pagination-controls">
          <button
            className="pagination-button"
            onClick={handlePrevPage}
            disabled={currentPage === 0}
            title="Previous page"
          >
            ‹
          </button>
          <span className="pagination-info">
            {currentPage + 1} / {totalPages}
          </span>
          <button
            className="pagination-button"
            onClick={handleNextPage}
            disabled={currentPage === totalPages - 1}
            title="Next page"
          >
            ›
          </button>
        </div>
      )}
    </div>
  );
}
