/**
 * ML Status page showing live detection status, detection statistics, and models.
 *
 * The system uses real-time detection from live camera streams (not worker-based
 * queue processing). Historical recording processing is done via ml_worker containers.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type {
  MLStatus,
  MLJob,
  MLDetectionSummary,
  MLModel,
  LiveDetectionStatus,
  LiveDetection,
  MLSettings,
} from '../types/camera';
import './MLStatusPage.css';

// Available detection classes from YOLO COCO dataset
const AVAILABLE_CLASSES = [
  'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
  'dog', 'cat', 'bird', 'backpack', 'handbag', 'suitcase',
];

export function MLStatusPage() {
  const [mlStatus, setMlStatus] = useState<MLStatus | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveDetectionStatus | null>(null);
  const [recentDetections, setRecentDetections] = useState<LiveDetection[]>([]);
  const [jobs, setJobs] = useState<MLJob[]>([]);
  const [jobsTotal, setJobsTotal] = useState(0);
  const [detectionSummary, setDetectionSummary] = useState<MLDetectionSummary | null>(null);
  const [models, setModels] = useState<MLModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionInProgress, setActionInProgress] = useState(false);

  // Settings state
  const [settings, setSettings] = useState<MLSettings | null>(null);
  const [editedSettings, setEditedSettings] = useState<MLSettings | null>(null);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [status, live, detections, jobsResponse, summary, modelsResponse, settingsResponse] = await Promise.all([
        api.getMLStatus(),
        api.getLiveDetectionStatus(),
        api.getLiveDetections({ limit: 10 }),
        api.getMLJobs({ limit: 20 }),
        api.getMLDetectionSummary(),
        api.getMLModels(),
        api.getMLSettings(),
      ]);
      setMlStatus(status);
      setLiveStatus(live);
      setRecentDetections(detections.detections);
      setJobs(jobsResponse.jobs);
      setJobsTotal(jobsResponse.total);
      setDetectionSummary(summary);
      setModels(modelsResponse.models);
      setSettings(settingsResponse);
      // Only set edited settings if not currently editing
      if (!editedSettings) {
        setEditedSettings(settingsResponse);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load ML status');
    } finally {
      setLoading(false);
    }
  }, [editedSettings]);

  useEffect(() => {
    loadData();
    // Refresh every 5 seconds for real-time updates
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

  const formatDate = (dateStr: string | undefined | null): string => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleString();
  };

  const formatDuration = (seconds: number | undefined): string => {
    if (seconds === undefined || seconds === null) return '-';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const formatDetectionTime = (dateStr: string | null): string => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleString(undefined, {
      month: 'numeric',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  const getJobStatusClass = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'completed';
      case 'processing':
        return 'processing';
      case 'failed':
        return 'failed';
      case 'cancelled':
        return 'cancelled';
      case 'pending':
      case 'queued':
        return 'pending';
      default:
        return '';
    }
  };

  const handleStartML = async () => {
    try {
      setActionInProgress(true);
      setError(null);
      await api.startMLSystem();
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start ML system');
    } finally {
      setActionInProgress(false);
    }
  };

  const handleStopML = async () => {
    try {
      setActionInProgress(true);
      setError(null);
      await api.stopMLSystem();
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop ML system');
    } finally {
      setActionInProgress(false);
    }
  };

  const handleProcessAll = async () => {
    try {
      setActionInProgress(true);
      setError(null);
      const result = await api.processAllRecordings({ limit: 500 });
      await loadData();
      alert(`Queued ${result.queued} recordings for processing`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to queue recordings');
    } finally {
      setActionInProgress(false);
    }
  };

  const handleRetryFailed = async () => {
    try {
      setActionInProgress(true);
      setError(null);
      const result = await api.retryFailedJobs();
      await loadData();
      alert(`Reset ${result.reset_count} failed jobs to pending`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retry jobs');
    } finally {
      setActionInProgress(false);
    }
  };

  const handleResetStuck = async () => {
    try {
      setActionInProgress(true);
      setError(null);
      const result = await api.resetStuckJobs();
      await loadData();
      alert(`Reset ${result.reset_count} stuck processing jobs to pending`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset stuck jobs');
    } finally {
      setActionInProgress(false);
    }
  };

  const handleSaveSettings = async () => {
    if (!editedSettings) return;

    try {
      setSettingsSaving(true);
      setSettingsError(null);
      const updated = await api.updateMLSettings({
        live_detection_enabled: editedSettings.live_detection_enabled,
        live_detection_fps: editedSettings.live_detection_fps,
        live_detection_cooldown: editedSettings.live_detection_cooldown,
        live_detection_confidence: editedSettings.live_detection_confidence,
        live_detection_classes: editedSettings.live_detection_classes,
        historical_confidence: editedSettings.historical_confidence,
        historical_classes: editedSettings.historical_classes,
      });
      setSettings(updated);
      setEditedSettings(updated);
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSettingsSaving(false);
    }
  };

  const handleResetSettings = () => {
    if (settings) {
      setEditedSettings(settings);
      setSettingsError(null);
    }
  };

  const hasSettingsChanges = (): boolean => {
    if (!settings || !editedSettings) return false;
    return (
      settings.live_detection_enabled !== editedSettings.live_detection_enabled ||
      settings.live_detection_fps !== editedSettings.live_detection_fps ||
      settings.live_detection_cooldown !== editedSettings.live_detection_cooldown ||
      settings.live_detection_confidence !== editedSettings.live_detection_confidence ||
      JSON.stringify(settings.live_detection_classes) !== JSON.stringify(editedSettings.live_detection_classes) ||
      settings.historical_confidence !== editedSettings.historical_confidence ||
      JSON.stringify(settings.historical_classes) !== JSON.stringify(editedSettings.historical_classes)
    );
  };

  const toggleClass = (classList: string[], className: string): string[] => {
    if (classList.includes(className)) {
      return classList.filter(c => c !== className);
    }
    return [...classList, className];
  };

  if (loading) {
    return (
      <div className="ml-status-page loading">
        <div className="loading-spinner">Loading ML status...</div>
      </div>
    );
  }

  return (
    <div className="ml-status-page">
      <div className="ml-status-header">
        <h2>ML Processing</h2>
        <div className="ml-controls">
          {mlStatus?.running ? (
            <button
              className="control-button stop"
              onClick={handleStopML}
              disabled={actionInProgress}
            >
              Stop ML
            </button>
          ) : (
            <button
              className="control-button start"
              onClick={handleStartML}
              disabled={actionInProgress}
            >
              Start ML
            </button>
          )}
          <button
            className="control-button process"
            onClick={handleProcessAll}
            disabled={actionInProgress || !mlStatus?.running}
            title={!mlStatus?.running ? 'Start ML system first' : 'Queue all unprocessed recordings'}
          >
            Process All
          </button>
          <button
            className="control-button retry"
            onClick={handleRetryFailed}
            disabled={actionInProgress}
            title="Reset failed jobs to pending for retry"
          >
            Retry Failed
          </button>
          <button
            className="control-button reset"
            onClick={handleResetStuck}
            disabled={actionInProgress}
            title="Reset stuck processing jobs to pending"
          >
            Reset Stuck
          </button>
          <button className="refresh-button" onClick={loadData}>
            Refresh
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="ml-status-grid">
        {/* Live Detection Status */}
        <div className="ml-status-card">
          <h3>Live Detection</h3>
          <div className="status-items">
            <div className="status-item">
              <span className="label">Status</span>
              <span className={`value ${liveStatus?.enabled ? 'good' : 'bad'}`}>
                {liveStatus?.enabled ? 'Active' : 'Disabled'}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Detections (1h)</span>
              <span className={`value ${(liveStatus?.detections_last_hour ?? 0) > 0 ? 'good' : ''}`}>
                {liveStatus?.detections_last_hour ?? 0}
              </span>
            </div>
          </div>
          {liveStatus?.config && (
            <div className="config-section">
              <h4>Current Configuration</h4>
              <div className="config-items">
                <span className="config-item">
                  <strong>Model:</strong> {liveStatus.config.model}
                </span>
                <span className="config-item">
                  <strong>FPS:</strong> {liveStatus.config.fps}
                </span>
                <span className="config-item">
                  <strong>Cooldown:</strong> {liveStatus.config.cooldown}s
                </span>
                <span className="config-item">
                  <strong>Confidence:</strong> {(liveStatus.config.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="config-classes">
                <strong>Classes:</strong>{' '}
                {liveStatus.config.classes.map((cls: string, i: number) => (
                  <span key={cls} className="class-tag">
                    {cls}{i < liveStatus.config.classes.length - 1 ? ', ' : ''}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ML Settings Editor */}
        <div className="ml-status-card settings-card">
          <div className="settings-header">
            <h3>ML Settings</h3>
            <div className="settings-actions">
              {hasSettingsChanges() && (
                <button
                  className="reset-button"
                  onClick={handleResetSettings}
                  disabled={settingsSaving}
                >
                  Reset
                </button>
              )}
              <button
                className="save-button"
                onClick={handleSaveSettings}
                disabled={settingsSaving || !hasSettingsChanges()}
              >
                {settingsSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>

          {settingsError && <div className="settings-error">{settingsError}</div>}

          {editedSettings && (
            <div className="settings-content">
              {/* Live Detection Settings */}
              <div className="settings-section">
                <h4>Live Detection</h4>

                <div className="setting-row">
                  <label className="setting-label">
                    <input
                      type="checkbox"
                      checked={editedSettings.live_detection_enabled}
                      onChange={(e) =>
                        setEditedSettings({
                          ...editedSettings,
                          live_detection_enabled: e.target.checked,
                        })
                      }
                    />
                    Enabled
                  </label>
                </div>

                <div className="setting-row">
                  <label className="setting-label">
                    Confidence Threshold
                    <span className="setting-value">
                      {(editedSettings.live_detection_confidence * 100).toFixed(0)}%
                    </span>
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={editedSettings.live_detection_confidence * 100}
                    onChange={(e) =>
                      setEditedSettings({
                        ...editedSettings,
                        live_detection_confidence: parseInt(e.target.value) / 100,
                      })
                    }
                    className="setting-slider"
                  />
                </div>

                <div className="setting-row">
                  <label className="setting-label">
                    FPS
                    <span className="setting-value">{editedSettings.live_detection_fps.toFixed(1)}</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={editedSettings.live_detection_fps * 10}
                    onChange={(e) =>
                      setEditedSettings({
                        ...editedSettings,
                        live_detection_fps: parseInt(e.target.value) / 10,
                      })
                    }
                    className="setting-slider"
                  />
                </div>

                <div className="setting-row">
                  <label className="setting-label">
                    Cooldown
                    <span className="setting-value">{editedSettings.live_detection_cooldown.toFixed(0)}s</span>
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="120"
                    value={editedSettings.live_detection_cooldown}
                    onChange={(e) =>
                      setEditedSettings({
                        ...editedSettings,
                        live_detection_cooldown: parseInt(e.target.value),
                      })
                    }
                    className="setting-slider"
                  />
                </div>

                <div className="setting-row classes-row">
                  <label className="setting-label">Detection Classes</label>
                  <div className="class-checkboxes">
                    {AVAILABLE_CLASSES.map((cls) => (
                      <label key={cls} className="class-checkbox">
                        <input
                          type="checkbox"
                          checked={editedSettings.live_detection_classes.includes(cls)}
                          onChange={() =>
                            setEditedSettings({
                              ...editedSettings,
                              live_detection_classes: toggleClass(
                                editedSettings.live_detection_classes,
                                cls
                              ),
                            })
                          }
                        />
                        {cls}
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              {/* Historical Processing Settings */}
              <div className="settings-section">
                <h4>Historical Processing</h4>

                <div className="setting-row">
                  <label className="setting-label">
                    Confidence Threshold
                    <span className="setting-value">
                      {(editedSettings.historical_confidence * 100).toFixed(0)}%
                    </span>
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={editedSettings.historical_confidence * 100}
                    onChange={(e) =>
                      setEditedSettings({
                        ...editedSettings,
                        historical_confidence: parseInt(e.target.value) / 100,
                      })
                    }
                    className="setting-slider"
                  />
                </div>

                <div className="setting-row classes-row">
                  <label className="setting-label">Detection Classes</label>
                  <div className="class-checkboxes">
                    {AVAILABLE_CLASSES.map((cls) => (
                      <label key={cls} className="class-checkbox">
                        <input
                          type="checkbox"
                          checked={editedSettings.historical_classes.includes(cls)}
                          onChange={() =>
                            setEditedSettings({
                              ...editedSettings,
                              historical_classes: toggleClass(
                                editedSettings.historical_classes,
                                cls
                              ),
                            })
                          }
                        />
                        {cls}
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <div className="settings-note">
                Changes take effect within 60 seconds
              </div>
            </div>
          )}
        </div>

        {/* Historical Processing Status */}
        <div className="ml-status-card">
          <h3>Historical Processing</h3>
          <div className="status-items">
            <div className="status-item">
              <span className="label">ML Engine</span>
              <span className={`value ${mlStatus?.running ? 'good' : 'bad'}`}>
                {mlStatus?.running ? 'Running' : 'Stopped'}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Pending Jobs</span>
              <span className={`value ${(mlStatus?.queue.pending ?? 0) > 10 ? 'warning' : ''}`}>
                {mlStatus?.queue.pending ?? 0}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Active Jobs</span>
              <span className={`value ${(mlStatus?.queue.active ?? 0) > 0 ? 'processing' : ''}`}>
                {mlStatus?.queue.active ?? 0}
              </span>
            </div>
          </div>
        </div>

        {/* Detection Summary */}
        <div className="ml-status-card">
          <h3>Detection Summary</h3>
          <div className="status-items">
            <div className="status-item">
              <span className="label">Total Detections</span>
              <span className="value">
                {detectionSummary?.total_detections.toLocaleString() ?? 0}
              </span>
            </div>
          </div>
          {detectionSummary && detectionSummary.items.length > 0 && (
            <div className="detection-breakdown">
              <h4>By Class</h4>
              <div className="detection-classes">
                {detectionSummary.items.slice(0, 8).map((item) => (
                  <div key={item.label} className="detection-class-item">
                    <span className="class-name">{item.label}</span>
                    <span className="class-count">{item.count.toLocaleString()}</span>
                    <span className="class-confidence">
                      {(item.avg_confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Recent Live Detections */}
        <div className="ml-status-card">
          <h3>Recent Live Detections</h3>
          {recentDetections.length > 0 ? (
            <div className="recent-detections">
              {recentDetections.map((det) => (
                <div key={det.id} className="detection-item">
                  <div className="detection-summary">
                    <span className="detection-camera">{det.camera_name}</span>
                    <span className="detection-timestamp">
                      {formatDetectionTime(det.detected_at)}
                    </span>
                  </div>
                  <div className="detection-details">
                    <span className="detection-class">{det.class_name}</span>
                    <span className="detection-confidence">
                      ({(det.confidence * 100).toFixed(0)}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="no-data">No recent detections</p>
          )}
        </div>

        {/* Recent Jobs */}
        <div className="ml-status-card wide">
          <h3>Recent Historical Jobs ({jobsTotal} total)</h3>
          {jobs.length > 0 ? (
            <div className="jobs-table-container">
            <table className="jobs-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Recording</th>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Detections</th>
                  <th>Duration</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id}>
                    <td>#{job.id}</td>
                    <td>{job.recording_id}</td>
                    <td>{job.model_name}</td>
                    <td>
                      <span className={`status-badge ${getJobStatusClass(job.status)}`}>
                        {job.status}
                      </span>
                    </td>
                    <td>
                      {job.status === 'processing' ? (
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{ width: `${job.progress_percent}%` }}
                          />
                          <span className="progress-text">
                            {job.progress_percent.toFixed(0)}%
                          </span>
                        </div>
                      ) : job.status === 'completed' ? (
                        '100%'
                      ) : (
                        '-'
                      )}
                    </td>
                    <td>{job.detections_count.toLocaleString()}</td>
                    <td>{formatDuration(job.processing_time_seconds)}</td>
                    <td>{formatDate(job.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          ) : (
            <p className="no-data">No jobs yet</p>
          )}
        </div>

        {/* Models */}
        <div className="ml-status-card wide">
          <h3>Available Models</h3>
          {models.length > 0 ? (
            <table className="models-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Version</th>
                  <th>Type</th>
                  <th>Input Size</th>
                  <th>Classes</th>
                  <th>Confidence</th>
                  <th>Default</th>
                  <th>Enabled</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.id}>
                    <td>
                      <strong>{model.display_name}</strong>
                      <span className="model-name-sub">{model.name}</span>
                    </td>
                    <td>{model.version}</td>
                    <td>{model.model_type.toUpperCase()}</td>
                    <td>{model.input_size.join('x')}</td>
                    <td>{model.class_names.length}</td>
                    <td>{(model.default_confidence_threshold * 100).toFixed(0)}%</td>
                    <td>
                      {model.is_default && (
                        <span className="default-badge">Default</span>
                      )}
                    </td>
                    <td>
                      <span className={`status-badge ${model.is_enabled ? 'enabled' : 'disabled'}`}>
                        {model.is_enabled ? 'Yes' : 'No'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="no-data">No models registered</p>
          )}
        </div>
      </div>
    </div>
  );
}
