/**
 * Camera grid component with configurable layout.
 */

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
  const { columns, rows } = layoutConfig[layout];
  const maxCameras = columns * rows;
  const displayCameras = cameras.slice(0, maxCameras);

  // Fill empty slots
  const emptySlots = maxCameras - displayCameras.length;

  return (
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
          <div key={camera.id} className="grid-cell">
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
  );
}
