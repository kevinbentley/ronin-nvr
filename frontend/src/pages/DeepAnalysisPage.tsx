/**
 * Deep Analysis page for VLLM activity characterization configuration.
 * Allows setting scene descriptions for cameras to provide context to the Vision LLM.
 */

import { useState, useCallback } from 'react';
import type { Camera, RecordingStatus } from '../types/camera';
import { api } from '../services/api';
import './DeepAnalysisPage.css';

interface DeepAnalysisPageProps {
  cameras: Camera[];
  recordingStatus: Map<number, RecordingStatus>;
  onCameraUpdated?: () => void;
}

export function DeepAnalysisPage({
  cameras,
  recordingStatus,
  onCameraUpdated,
}: DeepAnalysisPageProps) {
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [sceneDescription, setSceneDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSelectCamera = useCallback((camera: Camera) => {
    setSelectedCamera(camera);
    setSceneDescription(camera.scene_description || '');
    setError(null);
    setSuccess(null);
  }, []);

  const handleSave = useCallback(async () => {
    if (!selectedCamera) return;

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      await api.updateCamera(selectedCamera.id, {
        scene_description: sceneDescription || undefined,
      });
      setSuccess('Scene description saved successfully');
      // Update local state
      setSelectedCamera({
        ...selectedCamera,
        scene_description: sceneDescription,
      });
      // Notify parent to refresh camera list
      onCameraUpdated?.();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  }, [selectedCamera, sceneDescription, onCameraUpdated]);

  const getStatusIndicator = (camera: Camera) => {
    const status = recordingStatus.get(camera.id);
    if (status?.is_recording) {
      return <span className="status-dot recording" title="Recording" />;
    }
    if (camera.status === 'online') {
      return <span className="status-dot online" title="Online" />;
    }
    if (camera.status === 'offline') {
      return <span className="status-dot offline" title="Offline" />;
    }
    return <span className="status-dot unknown" title="Unknown" />;
  };

  const hasDescription = (camera: Camera) => {
    return camera.scene_description && camera.scene_description.trim().length > 0;
  };

  return (
    <div className="deep-analysis-page">
      <div className="page-header">
        <h2>Deep Analysis Configuration</h2>
        <p className="page-description">
          Configure scene descriptions for each camera. These descriptions help the Vision LLM
          understand the camera's view and provide better activity characterization.
        </p>
      </div>

      <div className="analysis-content">
        <div className="camera-list-section">
          <div className="section-header">
            <h3>Cameras</h3>
            <span className="camera-count">
              {cameras.filter(hasDescription).length}/{cameras.length} configured
            </span>
          </div>

          <div className="camera-list">
            {cameras.map((camera) => (
              <div
                key={camera.id}
                className={`camera-item ${selectedCamera?.id === camera.id ? 'selected' : ''} ${hasDescription(camera) ? 'configured' : ''}`}
                onClick={() => handleSelectCamera(camera)}
              >
                <div className="camera-item-left">
                  {getStatusIndicator(camera)}
                  <span className="camera-name">{camera.name}</span>
                </div>
                <div className="camera-item-right">
                  {hasDescription(camera) ? (
                    <span className="config-badge">Configured</span>
                  ) : (
                    <span className="config-badge not-configured">Not Configured</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="editor-section">
          {selectedCamera ? (
            <>
              <div className="section-header">
                <h3>Scene Description for {selectedCamera.name}</h3>
              </div>

              {error && <div className="error-banner">{error}</div>}
              {success && <div className="success-banner">{success}</div>}

              <div className="editor-content">
                <div className="form-group">
                  <label htmlFor="scene_description">
                    Describe what the camera sees (landmarks, typical objects, areas)
                  </label>
                  <textarea
                    id="scene_description"
                    value={sceneDescription}
                    onChange={(e) => setSceneDescription(e.target.value)}
                    placeholder="e.g., Front yard view: white house with red door, concrete driveway on right, black garbage bin near garage, flower planter on porch steps, mailbox at street"
                    maxLength={2000}
                    rows={6}
                  />
                  <div className="char-count">
                    {sceneDescription.length}/2000 characters
                  </div>
                </div>

                <div className="description-tips">
                  <h4>Tips for effective scene descriptions:</h4>
                  <ul>
                    <li>Describe fixed landmarks (house color, furniture, bins)</li>
                    <li>Note typical activity zones (driveway, walkway, porch)</li>
                    <li>Mention where deliveries normally go</li>
                    <li>Include any permanent objects that help orient the view</li>
                  </ul>
                </div>

                <div className="form-actions">
                  <button
                    className="save-button"
                    onClick={handleSave}
                    disabled={saving}
                  >
                    {saving ? 'Saving...' : 'Save Description'}
                  </button>
                  {selectedCamera.scene_description !== sceneDescription && (
                    <button
                      className="reset-button"
                      onClick={() => setSceneDescription(selectedCamera.scene_description || '')}
                      disabled={saving}
                    >
                      Reset
                    </button>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="no-selection">
              <div className="no-selection-icon">&#128247;</div>
              <p>Select a camera from the list to configure its scene description</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
