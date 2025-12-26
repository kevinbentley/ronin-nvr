/**
 * Sidebar showing camera list with show/hide controls.
 */

import type { Camera, RecordingStatus } from '../types/camera';
import './CameraSidebar.css';

interface CameraSidebarProps {
  cameras: Camera[];
  recordingStatus: Map<number, RecordingStatus>;
  hiddenCameraIds: Set<number>;
  onToggleVisibility: (cameraId: number) => void;
  onShowAll: () => void;
  onHideAll: () => void;
}

export function CameraSidebar({
  cameras,
  recordingStatus,
  hiddenCameraIds,
  onToggleVisibility,
  onShowAll,
  onHideAll,
}: CameraSidebarProps) {
  if (cameras.length === 0) {
    return (
      <aside className="camera-sidebar">
        <div className="sidebar-header">
          <h3>Cameras</h3>
        </div>
        <div className="no-cameras">
          <p>No cameras configured</p>
          <p className="hint">Go to Setup to add cameras</p>
        </div>
      </aside>
    );
  }

  const visibleCount = cameras.length - hiddenCameraIds.size;

  return (
    <aside className="camera-sidebar">
      <div className="sidebar-header">
        <h3>Cameras</h3>
        <span className="camera-count">{visibleCount}/{cameras.length}</span>
      </div>

      <div className="sidebar-actions">
        <button className="sidebar-action-btn" onClick={onShowAll}>
          Show All
        </button>
        <button className="sidebar-action-btn" onClick={onHideAll}>
          Hide All
        </button>
      </div>

      <ul className="camera-list">
        {cameras.map((camera) => {
          const isHidden = hiddenCameraIds.has(camera.id);
          const status = recordingStatus.get(camera.id);
          return (
            <li key={camera.id} className={`camera-item ${isHidden ? 'hidden-camera' : ''}`}>
              <div className="camera-info">
                <span className={`status-indicator ${camera.status}`} />
                <span className="camera-name">{camera.name}</span>
                {status?.is_recording && (
                  <span className="recording-badge" title="Recording">REC</span>
                )}
              </div>
              <div className="camera-actions">
                <button
                  className={`visibility-button ${isHidden ? 'hidden' : 'visible'}`}
                  onClick={() => onToggleVisibility(camera.id)}
                  title={isHidden ? 'Show in grid' : 'Hide from grid'}
                >
                  {isHidden ? 'Show' : 'Hide'}
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
