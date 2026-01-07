/**
 * Modal for adding/editing cameras.
 */

import { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';
import type {
  Camera,
  CameraCreate,
  CameraUpdate,
  CameraTestResult,
  ONVIFProbeResponse,
  ONVIFProfile,
} from '../types/camera';
import './CameraModal.css';

interface CameraModalProps {
  camera?: Camera;
  onClose: () => void;
  onSave: () => void;
}

// Placeholder shown when a password exists but hasn't been modified
const PASSWORD_PLACEHOLDER = '••••••••';

export function CameraModal({ camera, onClose, onSave }: CameraModalProps) {
  const [formData, setFormData] = useState<CameraCreate>({
    name: '',
    host: '',
    port: 554,
    path: '/cam/realmonitor',
    username: '',
    password: '',
    transport: 'tcp',
    recording_enabled: true,
    onvif_port: 80,
    onvif_enabled: false,
    onvif_events_enabled: false,
  });
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<CameraTestResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track if password was modified during editing
  const [passwordModified, setPasswordModified] = useState(false);
  const hasExistingPassword = useRef(false);

  // Track the saved camera ID after initial creation (for "Save & Test" flow)
  const [savedCameraId, setSavedCameraId] = useState<number | null>(null);

  // ONVIF probe state
  const [probing, setProbing] = useState(false);
  const [probeResult, setProbeResult] = useState<ONVIFProbeResponse | null>(null);
  const [selectedProfile, setSelectedProfile] = useState<ONVIFProfile | null>(null);

  const isEditing = !!camera || savedCameraId !== null;

  useEffect(() => {
    if (camera) {
      // Check if camera likely has a password (username exists)
      // Backend doesn't return password for security, so we infer from username
      hasExistingPassword.current = !!camera.username;

      setFormData({
        name: camera.name,
        host: camera.host,
        port: camera.port,
        path: camera.path,
        username: camera.username || '',
        password: '', // Don't populate - backend doesn't return it
        transport: camera.transport,
        recording_enabled: camera.recording_enabled,
        onvif_port: camera.onvif_port || 80,
        onvif_enabled: camera.onvif_enabled || false,
        onvif_events_enabled: camera.onvif_events_enabled || false,
      });
      setPasswordModified(false);
    }
  }, [camera]);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;

    // Track if password field was modified
    if (name === 'password') {
      setPasswordModified(true);
    }

    setFormData((prev) => ({
      ...prev,
      [name]:
        type === 'checkbox'
          ? (e.target as HTMLInputElement).checked
          : type === 'number'
          ? parseInt(value, 10)
          : value,
    }));
  };

  const handleTest = async () => {
    if (!camera) return;

    setTesting(true);
    setTestResult(null);
    try {
      const result = await api.testCamera(camera.id);
      setTestResult(result);
    } catch (err) {
      setTestResult({
        success: false,
        message: err instanceof Error ? err.message : 'Test failed',
      });
    } finally {
      setTesting(false);
    }
  };

  const handleProbeONVIF = async () => {
    if (!formData.host) return;

    setProbing(true);
    setProbeResult(null);
    setSelectedProfile(null);
    setError(null);

    try {
      const result = await api.probeONVIF({
        host: formData.host,
        onvif_port: formData.onvif_port || 80,
        username: formData.username || undefined,
        password: formData.password || undefined,
        timeout: 10,
      });
      setProbeResult(result);

      if (result.success && result.profiles.length > 0) {
        // Auto-select the first (usually main stream) profile
        setSelectedProfile(result.profiles[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'ONVIF probe failed');
    } finally {
      setProbing(false);
    }
  };

  const handleApplyProfile = (profile: ONVIFProfile) => {
    setSelectedProfile(profile);

    // Parse RTSP URL to extract path and port
    try {
      const url = new URL(profile.rtsp_url);
      let newPath = url.pathname;
      if (url.search) {
        newPath += url.search;
      }
      const newPort = parseInt(url.port, 10) || 554;

      setFormData((prev) => ({
        ...prev,
        port: newPort,
        path: newPath || '/stream',
        onvif_enabled: true,
      }));
    } catch {
      // If URL parsing fails, just use the path as-is
      setFormData((prev) => ({
        ...prev,
        onvif_enabled: true,
      }));
    }
  };

  const handleSubmit = async (e: React.FormEvent, andTest: boolean = false) => {
    e.preventDefault();
    setError(null);
    setSaving(true);
    setTestResult(null);

    try {
      let savedCamera: Camera;
      const cameraId = camera?.id ?? savedCameraId;

      if (isEditing && cameraId !== null) {
        // Build update payload - only include password if it was modified
        const updateData: CameraUpdate = {
          name: formData.name,
          host: formData.host,
          port: formData.port,
          path: formData.path,
          username: formData.username,
          transport: formData.transport,
          recording_enabled: formData.recording_enabled,
        };
        // Only include password if user actually changed it
        if (passwordModified && formData.password) {
          updateData.password = formData.password;
        }
        savedCamera = await api.updateCamera(cameraId, updateData);
      } else {
        savedCamera = await api.createCamera(formData);
        // Track the ID so subsequent saves don't try to create again
        setSavedCameraId(savedCamera.id);
      }

      if (andTest) {
        // Test the newly saved camera
        setTesting(true);
        try {
          const result = await api.testCamera(savedCamera.id);
          setTestResult(result);
          onSave(); // Refresh camera list to show updated status
        } catch (err) {
          setTestResult({
            success: false,
            message: err instanceof Error ? err.message : 'Test failed',
          });
        } finally {
          setTesting(false);
          setSaving(false);
        }
      } else {
        onSave();
        onClose();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save camera');
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!camera || !confirm(`Delete camera "${camera.name}"?`)) return;

    try {
      await api.deleteCamera(camera.id);
      onSave();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete camera');
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">
          <h2>{isEditing ? 'Edit Camera' : 'Add Camera'}</h2>
          <button className="close-button" onClick={onClose}>
            &times;
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Name</label>
            <input
              id="name"
              name="name"
              type="text"
              value={formData.name}
              onChange={handleChange}
              required
              placeholder="Front Door"
            />
          </div>

          <div className="form-row">
            <div className="form-group flex-2">
              <label htmlFor="host">Host/IP</label>
              <input
                id="host"
                name="host"
                type="text"
                value={formData.host}
                onChange={handleChange}
                required
                placeholder="192.168.1.100"
              />
            </div>
            <div className="form-group flex-1">
              <label htmlFor="port">Port</label>
              <input
                id="port"
                name="port"
                type="number"
                value={formData.port}
                onChange={handleChange}
                min="1"
                max="65535"
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="path">RTSP Path</label>
            <input
              id="path"
              name="path"
              type="text"
              value={formData.path}
              onChange={handleChange}
              placeholder="/cam/realmonitor"
            />
          </div>

          <div className="form-row">
            <div className="form-group flex-1">
              <label htmlFor="username">Username</label>
              <input
                id="username"
                name="username"
                type="text"
                value={formData.username}
                onChange={handleChange}
                placeholder="admin"
              />
            </div>
            <div className="form-group flex-1">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                placeholder={
                  isEditing && hasExistingPassword.current && !passwordModified
                    ? PASSWORD_PLACEHOLDER
                    : ''
                }
              />
              {isEditing && hasExistingPassword.current && !passwordModified && (
                <small className="field-hint">Leave empty to keep current password</small>
              )}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group flex-1">
              <label htmlFor="transport">Transport</label>
              <select
                id="transport"
                name="transport"
                value={formData.transport}
                onChange={handleChange}
              >
                <option value="tcp">TCP</option>
                <option value="udp">UDP</option>
              </select>
            </div>
            <div className="form-group flex-1 checkbox-group">
              <label>
                <input
                  type="checkbox"
                  name="recording_enabled"
                  checked={formData.recording_enabled}
                  onChange={handleChange}
                />
                Enable Recording
              </label>
            </div>
          </div>

          {/* ONVIF Section */}
          <div className="form-section">
            <h3>ONVIF Auto-Detection</h3>
            <div className="form-row">
              <div className="form-group flex-1">
                <label htmlFor="onvif_port">ONVIF Port</label>
                <input
                  id="onvif_port"
                  name="onvif_port"
                  type="number"
                  value={formData.onvif_port || 80}
                  onChange={handleChange}
                  min="1"
                  max="65535"
                />
              </div>
              <div className="form-group flex-2 onvif-probe-group">
                <label>&nbsp;</label>
                <button
                  type="button"
                  className="btn-onvif-probe"
                  onClick={handleProbeONVIF}
                  disabled={probing || !formData.host}
                >
                  {probing ? 'Detecting...' : 'Detect Streams'}
                </button>
              </div>
            </div>

            {probeResult && probeResult.success && (
              <div className="onvif-result">
                {probeResult.device_info.manufacturer && (
                  <div className="device-info">
                    {probeResult.device_info.manufacturer} {probeResult.device_info.model}
                    {probeResult.device_info.firmware && (
                      <span className="firmware"> (FW: {probeResult.device_info.firmware})</span>
                    )}
                  </div>
                )}
                <div className="profiles-list">
                  <label>Available Streams:</label>
                  {probeResult.profiles.map((profile) => (
                    <div
                      key={profile.token}
                      className={`profile-item ${selectedProfile?.token === profile.token ? 'selected' : ''}`}
                      onClick={() => handleApplyProfile(profile)}
                    >
                      <span className="profile-name">{profile.name}</span>
                      <span className="profile-details">
                        {profile.resolution} {profile.encoding} {profile.fps && `@ ${profile.fps}fps`}
                      </span>
                    </div>
                  ))}
                </div>
                {probeResult.has_events && (
                  <div className="form-group checkbox-group events-option">
                    <label>
                      <input
                        type="checkbox"
                        name="onvif_events_enabled"
                        checked={formData.onvif_events_enabled}
                        onChange={handleChange}
                      />
                      Enable Camera Motion Events
                    </label>
                    <small>Receive instant alerts when camera detects motion</small>
                  </div>
                )}
              </div>
            )}

            {probeResult && !probeResult.success && (
              <div className="onvif-error">
                ONVIF detection failed: {probeResult.error}
              </div>
            )}
          </div>

          {testResult && (
            <div className={`test-result ${testResult.success ? 'success' : 'error'}`}>
              <strong>{testResult.success ? 'Success' : 'Failed'}:</strong>{' '}
              {testResult.message}
              {testResult.success && testResult.codec && (
                <div className="test-details">
                  {testResult.codec} | {testResult.resolution} | {testResult.fps} fps
                </div>
              )}
            </div>
          )}

          {error && <div className="form-error">{error}</div>}

          <div className="modal-actions">
            {isEditing && (
              <>
                <button
                  type="button"
                  className="btn-test"
                  onClick={handleTest}
                  disabled={testing || saving}
                >
                  {testing ? 'Testing...' : 'Test Connection'}
                </button>
                <button
                  type="button"
                  className="btn-delete"
                  onClick={handleDelete}
                >
                  Delete
                </button>
              </>
            )}
            <div className="spacer" />
            <button type="button" className="btn-cancel" onClick={onClose}>
              Cancel
            </button>
            {!isEditing && (
              <button
                type="button"
                className="btn-test"
                onClick={(e) => handleSubmit(e, true)}
                disabled={saving || testing}
              >
                {saving || testing ? 'Testing...' : 'Save & Test'}
              </button>
            )}
            <button type="submit" className="btn-save" disabled={saving || testing}>
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
