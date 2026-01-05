/**
 * Setup page for camera configuration and system settings.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { CameraModal } from '../components/CameraModal';
import type { Camera, RecordingStatus, RetentionSettings } from '../types/camera';
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

  // Retention settings state
  const [retentionSettings, setRetentionSettings] = useState<RetentionSettings | null>(null);
  const [retentionLoading, setRetentionLoading] = useState(true);
  const [retentionSaving, setRetentionSaving] = useState(false);
  const [retentionDays, setRetentionDays] = useState<string>('');
  const [retentionMaxGb, setRetentionMaxGb] = useState<string>('');
  const [retentionUnlimitedDays, setRetentionUnlimitedDays] = useState(false);
  const [retentionUnlimitedSize, setRetentionUnlimitedSize] = useState(false);
  const [retentionSuccess, setRetentionSuccess] = useState<string | null>(null);

  const loadRetentionSettings = useCallback(async () => {
    try {
      setRetentionLoading(true);
      const settings = await api.getRetentionSettings();
      setRetentionSettings(settings);

      // Set form values
      if (settings.retention_days === null) {
        setRetentionUnlimitedDays(true);
        setRetentionDays('');
      } else {
        setRetentionUnlimitedDays(false);
        setRetentionDays(settings.retention_days.toString());
      }

      if (settings.retention_max_gb === null) {
        setRetentionUnlimitedSize(true);
        setRetentionMaxGb('');
      } else {
        setRetentionUnlimitedSize(false);
        setRetentionMaxGb(settings.retention_max_gb.toString());
      }
    } catch (err) {
      setActionError(
        `Failed to load retention settings: ${
          err instanceof Error ? err.message : 'Unknown error'
        }`
      );
    } finally {
      setRetentionLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRetentionSettings();
  }, [loadRetentionSettings]);

  const handleSaveRetentionSettings = async () => {
    setActionError(null);
    setRetentionSuccess(null);
    setRetentionSaving(true);

    try {
      const update: { retention_days?: number | null; retention_max_gb?: number | null } = {};

      if (retentionUnlimitedDays) {
        update.retention_days = null;
      } else if (retentionDays) {
        const days = parseInt(retentionDays, 10);
        if (isNaN(days) || days < 1 || days > 3650) {
          throw new Error('Retention days must be between 1 and 3650');
        }
        update.retention_days = days;
      }

      if (retentionUnlimitedSize) {
        update.retention_max_gb = null;
      } else if (retentionMaxGb) {
        const gb = parseFloat(retentionMaxGb);
        if (isNaN(gb) || gb < 1 || gb > 100000) {
          throw new Error('Max storage must be between 1 and 100,000 GB');
        }
        update.retention_max_gb = gb;
      }

      const newSettings = await api.updateRetentionSettings(update);
      setRetentionSettings(newSettings);
      setRetentionSuccess('Retention settings saved successfully');
      setTimeout(() => setRetentionSuccess(null), 3000);
    } catch (err) {
      setActionError(
        `Failed to save retention settings: ${
          err instanceof Error ? err.message : 'Unknown error'
        }`
      );
    } finally {
      setRetentionSaving(false);
    }
  };

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

      {/* Retention Policy Section */}
      <section className="setup-section">
        <div className="section-header">
          <h3>Retention Policy</h3>
        </div>

        {retentionSuccess && (
          <div className="success-banner">
            {retentionSuccess}
          </div>
        )}

        {retentionLoading ? (
          <div className="settings-loading">Loading settings...</div>
        ) : (
          <div className="retention-settings">
            <p className="settings-description">
              Configure how long recordings are kept before automatic deletion.
              Retention checks run every {retentionSettings?.retention_check_interval_minutes ?? 60} minutes.
            </p>

            <div className="settings-form">
              <div className="form-group">
                <label htmlFor="retention-days">Keep recordings for</label>
                <div className="input-with-checkbox">
                  <input
                    type="number"
                    id="retention-days"
                    value={retentionDays}
                    onChange={(e) => setRetentionDays(e.target.value)}
                    disabled={retentionUnlimitedDays || retentionSaving}
                    min="1"
                    max="3650"
                    placeholder="30"
                  />
                  <span className="input-suffix">days</span>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={retentionUnlimitedDays}
                      onChange={(e) => setRetentionUnlimitedDays(e.target.checked)}
                      disabled={retentionSaving}
                    />
                    Unlimited
                  </label>
                </div>
                <span className="form-hint">
                  Recordings older than this will be automatically deleted
                </span>
              </div>

              <div className="form-group">
                <label htmlFor="retention-max-gb">Maximum storage</label>
                <div className="input-with-checkbox">
                  <input
                    type="number"
                    id="retention-max-gb"
                    value={retentionMaxGb}
                    onChange={(e) => setRetentionMaxGb(e.target.value)}
                    disabled={retentionUnlimitedSize || retentionSaving}
                    min="1"
                    max="100000"
                    placeholder="500"
                  />
                  <span className="input-suffix">GB</span>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={retentionUnlimitedSize}
                      onChange={(e) => setRetentionUnlimitedSize(e.target.checked)}
                      disabled={retentionSaving}
                    />
                    Unlimited
                  </label>
                </div>
                <span className="form-hint">
                  When storage exceeds this limit, oldest recordings are deleted first
                </span>
              </div>

              <div className="form-actions">
                <button
                  className="save-button"
                  onClick={handleSaveRetentionSettings}
                  disabled={retentionSaving}
                >
                  {retentionSaving ? 'Saving...' : 'Save Retention Settings'}
                </button>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Future Settings Placeholder */}
      <section className="setup-section">
        <div className="section-header">
          <h3>Additional Settings</h3>
        </div>
        <div className="settings-placeholder">
          <p>More settings coming in future updates:</p>
          <ul>
            <li>User management</li>
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
