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
} from '../types/camera';
import './MLStatusPage.css';

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

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [status, live, detections, jobsResponse, summary, modelsResponse] = await Promise.all([
        api.getMLStatus(),
        api.getLiveDetectionStatus(),
        api.getLiveDetections({ limit: 10 }),
        api.getMLJobs({ limit: 20 }),
        api.getMLDetectionSummary(),
        api.getMLModels(),
      ]);
      setMlStatus(status);
      setLiveStatus(live);
      setRecentDetections(detections.detections);
      setJobs(jobsResponse.jobs);
      setJobsTotal(jobsResponse.total);
      setDetectionSummary(summary);
      setModels(modelsResponse.models);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load ML status');
    } finally {
      setLoading(false);
    }
  }, []);

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

  const formatTimeAgo = (dateStr: string | null): string => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);

    if (diffSecs < 60) return `${diffSecs}s ago`;
    const diffMins = Math.floor(diffSecs / 60);
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    return `${diffHours}h ago`;
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
              <h4>Configuration</h4>
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
                  <div className="detection-main">
                    <span className="detection-class">{det.class_name}</span>
                    <span className="detection-confidence">
                      {(det.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="detection-meta">
                    <span className="detection-camera">Camera {det.camera_id}</span>
                    <span className="detection-time">{formatTimeAgo(det.detected_at)}</span>
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
