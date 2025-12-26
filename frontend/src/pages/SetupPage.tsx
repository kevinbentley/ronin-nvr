/**
 * Setup page for camera configuration and system settings.
 */

import { useState } from 'react';
import { api } from '../services/api';
import { CameraModal } from '../components/CameraModal';
import type { Camera, RecordingStatus } from '../types/camera';
import './SetupPage.css';

interface SetupPageProps {
  cameras: Camera[];
  recordingStatus: Map<number, RecordingStatus>;
  onRefresh: () => void;
}

export function SetupPage({ cameras, recordingStatus, onRefresh }: SetupPageProps) {
  const [showModal, setShowModal] = useState(false);
  const [editingCamera, setEditingCamera] = useState<Camera | undefined>();
  const [actionError, setActionError] = useState<string | null>(null);

  const handleAddCamera = () => {
    setEditingCamera(undefined);
    setShowModal(true);
  };

  const handleEditCamera = (camera: Camera) => {
    setEditingCamera(camera);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setEditingCamera(undefined);
  };

  const handleSaveCamera = () => {
    onRefresh();
  };

  const handleToggleRecording = async (camera: Camera) => {
    const status = recordingStatus.get(camera.id);
    setActionError(null);
    try {
      if (status?.is_recording) {
        await api.stopRecording(camera.id);
      } else {
        await api.startRecording(camera.id);
      }
      onRefresh();
    } catch (err) {
      setActionError(
        `Failed to ${status?.is_recording ? 'stop' : 'start'} recording: ${
          err instanceof Error ? err.message : 'Unknown error'
        }`
      );
    }
  };

  const handleStartAllRecording = async () => {
    setActionError(null);
    try {
      const onlineCameras = cameras.filter((c) => c.status === 'online');
      await Promise.all(
        onlineCameras.map((camera) => api.startRecording(camera.id))
      );
      onRefresh();
    } catch (err) {
      setActionError(
        `Failed to start all recordings: ${
          err instanceof Error ? err.message : 'Unknown error'
        }`
      );
    }
  };

  const handleStopAllRecording = async () => {
    setActionError(null);
    try {
      await Promise.all(
        cameras.map((camera) => api.stopRecording(camera.id))
      );
      onRefresh();
    } catch (err) {
      setActionError(
        `Failed to stop all recordings: ${
          err instanceof Error ? err.message : 'Unknown error'
        }`
      );
    }
  };

  return (
    <div className="setup-page">
      <div className="setup-header">
        <h2>Setup</h2>
      </div>

      {actionError && (
        <div className="error-banner">
          {actionError}
          <button onClick={() => setActionError(null)}>Dismiss</button>
        </div>
      )}

      {/* Camera Management Section */}
      <section className="setup-section">
        <div className="section-header">
          <h3>Cameras</h3>
          <div className="section-actions">
            <button className="action-button" onClick={handleStartAllRecording}>
              Start All Recording
            </button>
            <button className="action-button" onClick={handleStopAllRecording}>
              Stop All Recording
            </button>
            <button className="add-button" onClick={handleAddCamera}>
              + Add Camera
            </button>
          </div>
        </div>

        {cameras.length === 0 ? (
          <div className="empty-state">
            <p>No cameras configured yet.</p>
            <button className="add-button large" onClick={handleAddCamera}>
              + Add Your First Camera
            </button>
          </div>
        ) : (
          <div className="camera-table-container">
            <table className="camera-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Host</th>
                  <th>Status</th>
                  <th>Recording</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {cameras.map((camera) => {
                  const status = recordingStatus.get(camera.id);
                  return (
                    <tr key={camera.id}>
                      <td className="camera-name-cell">
                        <span className={`status-dot ${camera.status}`} />
                        {camera.name}
                      </td>
                      <td className="host-cell">
                        {camera.host}:{camera.port}
                      </td>
                      <td>
                        <span className={`status-badge ${camera.status}`}>
                          {camera.status}
                        </span>
                      </td>
                      <td>
                        {status?.is_recording ? (
                          <span className="status-badge recording">Recording</span>
                        ) : (
                          <span className="status-badge stopped">Stopped</span>
                        )}
                      </td>
                      <td className="actions-cell">
                        <button
                          className="table-action-button"
                          onClick={() => handleEditCamera(camera)}
                        >
                          Edit
                        </button>
                        {status?.is_recording ? (
                          <button
                            className="table-action-button stop"
                            onClick={() => handleToggleRecording(camera)}
                          >
                            Stop Rec
                          </button>
                        ) : (
                          <button
                            className="table-action-button start"
                            onClick={() => handleToggleRecording(camera)}
                            disabled={camera.status !== 'online'}
                          >
                            Start Rec
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Future Settings Section Placeholder */}
      <section className="setup-section">
        <div className="section-header">
          <h3>System Settings</h3>
        </div>
        <div className="settings-placeholder">
          <p>Additional settings will be available in future updates.</p>
          <ul>
            <li>Retention policy configuration</li>
            <li>User authentication</li>
            <li>Email/notification settings</li>
            <li>Storage path configuration</li>
          </ul>
        </div>
      </section>

      {showModal && (
        <CameraModal
          camera={editingCamera}
          onClose={handleCloseModal}
          onSave={handleSaveCamera}
        />
      )}
    </div>
  );
}
