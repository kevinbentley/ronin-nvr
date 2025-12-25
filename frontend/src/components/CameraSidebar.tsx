/**
 * Sidebar showing camera list with controls.
 */

import { api } from '../services/api';
import type { Camera, RecordingStatus } from '../types/camera';
import './CameraSidebar.css';

interface CameraSidebarProps {
  cameras: Camera[];
  recordingStatus: Map<number, RecordingStatus>;
  onEditCamera: (camera: Camera) => void;
  onRefresh: () => void;
}

export function CameraSidebar({
  cameras,
  recordingStatus,
  onEditCamera,
  onRefresh,
}: CameraSidebarProps) {
  const handleToggleRecording = async (camera: Camera) => {
    const status = recordingStatus.get(camera.id);
    try {
      if (status?.is_recording) {
        await api.stopRecording(camera.id);
      } else {
        await api.startRecording(camera.id);
      }
      onRefresh();
    } catch (err) {
      console.error('Failed to toggle recording:', err);
    }
  };

  if (cameras.length === 0) {
    return (
      <aside className="camera-sidebar">
        <div className="sidebar-header">
          <h3>Cameras</h3>
        </div>
        <div className="no-cameras">
          <p>No cameras configured</p>
          <p className="hint">Click "Add Camera" to get started</p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="camera-sidebar">
      <div className="sidebar-header">
        <h3>Cameras</h3>
        <span className="camera-count">{cameras.length}</span>
      </div>

      <ul className="camera-list">
        {cameras.map((camera) => {
          const status = recordingStatus.get(camera.id);
          return (
            <li key={camera.id} className="camera-item">
              <div className="camera-info" onClick={() => onEditCamera(camera)}>
                <span className={`status-indicator ${camera.status}`} />
                <span className="camera-name">{camera.name}</span>
              </div>
              <div className="camera-actions">
                {status?.is_recording ? (
                  <button
                    className="rec-button recording"
                    onClick={() => handleToggleRecording(camera)}
                    title="Stop Recording"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    className="rec-button"
                    onClick={() => handleToggleRecording(camera)}
                    title="Start Recording"
                    disabled={camera.status !== 'online'}
                  >
                    Rec
                  </button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
