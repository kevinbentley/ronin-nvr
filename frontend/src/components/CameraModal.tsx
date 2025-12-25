/**
 * Modal for adding/editing cameras.
 */

import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { Camera, CameraCreate, CameraTestResult } from '../types/camera';
import './CameraModal.css';

interface CameraModalProps {
  camera?: Camera;
  onClose: () => void;
  onSave: () => void;
}

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
  });
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<CameraTestResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isEditing = !!camera;

  useEffect(() => {
    if (camera) {
      setFormData({
        name: camera.name,
        host: camera.host,
        port: camera.port,
        path: camera.path,
        username: camera.username || '',
        password: camera.password || '',
        transport: camera.transport,
        recording_enabled: camera.recording_enabled,
      });
    }
  }, [camera]);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSaving(true);

    try {
      if (isEditing && camera) {
        await api.updateCamera(camera.id, formData);
      } else {
        await api.createCamera(formData);
      }
      onSave();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save camera');
    } finally {
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
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
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
              />
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
                  disabled={testing}
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
            <button type="submit" className="btn-save" disabled={saving}>
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
