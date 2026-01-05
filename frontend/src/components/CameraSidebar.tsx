/**
 * Sidebar showing camera list with show/hide controls.
 * Can be collapsed to save screen space.
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
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export function CameraSidebar({
  cameras,
  recordingStatus,
  hiddenCameraIds,
  onToggleVisibility,
  onShowAll,
  onHideAll,
  isCollapsed,
  onToggleCollapse,
}: CameraSidebarProps) {
  const visibleCount = cameras.length - hiddenCameraIds.size;

  if (isCollapsed) {
    return (
      <aside className="camera-sidebar collapsed">
        <button
          className="collapse-toggle"
          onClick={onToggleCollapse}
          title="Expand sidebar"
        >
          <span className="collapse-icon">&#9654;</span>
        </button>
        <div className="collapsed-cameras">
          {cameras.map((camera) => {
            const isHidden = hiddenCameraIds.has(camera.id);
            const status = recordingStatus.get(camera.id);
            return (
              <div
                key={camera.id}
                className={`collapsed-camera-dot ${camera.status} ${isHidden ? 'hidden-camera' : ''}`}
                title={`${camera.name}${status?.is_recording ? ' (Recording)' : ''}`}
              >
                {status?.is_recording && <span className="mini-rec" />}
              </div>
            );
          })}
        </div>
      </aside>
    );
  }

  if (cameras.length === 0) {
    return (
      <aside className="camera-sidebar">
        <div className="sidebar-header">
          <button
            className="collapse-toggle"
            onClick={onToggleCollapse}
            title="Collapse sidebar"
          >
            <span className="collapse-icon">&#9664;</span>
          </button>
          <h3>Cameras</h3>
        </div>
        <div className="no-cameras">
          <p>No cameras configured</p>
          <p className="hint">Go to Setup to add cameras</p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="camera-sidebar">
      <div className="sidebar-header">
        <button
          className="collapse-toggle"
          onClick={onToggleCollapse}
          title="Collapse sidebar"
        >
          <span className="collapse-icon">&#9664;</span>
        </button>
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
