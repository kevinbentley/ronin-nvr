/**
 * ML Status page showing ML system health, jobs, and detection statistics.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type {
  MLStatus,
  MLJob,
  MLDetectionSummary,
  MLModel,
} from '../types/camera';
import './MLStatusPage.css';

export function MLStatusPage() {
  const [mlStatus, setMlStatus] = useState<MLStatus | null>(null);
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
      const [status, jobsResponse, summary, modelsResponse] = await Promise.all([
        api.getMLStatus(),
        api.getMLJobs({ limit: 20 }),
        api.getMLDetectionSummary(),
        api.getMLModels(),
      ]);
      setMlStatus(status);
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
    // Refresh every 5 seconds for real-time job progress
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

  const formatDate = (dateStr: string | undefined): string => {
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

  const activeWorkers = mlStatus?.worker_status.filter(w => w.running).length ?? 0;
  const busyWorkers = mlStatus?.worker_status.filter(w => w.current_job !== null).length ?? 0;

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
        {/* System Status */}
        <div className="ml-status-card">
          <h3>System Status</h3>
          <div className="status-items">
            <div className="status-item">
              <span className="label">ML Engine</span>
              <span className={`value ${mlStatus?.running ? 'good' : 'bad'}`}>
                {mlStatus?.running ? 'Running' : 'Stopped'}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Workers</span>
              <span className="value">
                {busyWorkers} / {activeWorkers} busy
              </span>
            </div>
            <div className="status-item">
              <span className="label">Models Loaded</span>
              <span className="value">{mlStatus?.models_loaded.length ?? 0}</span>
            </div>
          </div>
        </div>

        {/* Queue Status */}
        <div className="ml-status-card">
          <h3>Job Queue</h3>
          <div className="status-items">
            <div className="status-item">
              <span className="label">Pending</span>
              <span className={`value ${(mlStatus?.queue.pending ?? 0) > 10 ? 'warning' : ''}`}>
                {mlStatus?.queue.pending ?? 0}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Active</span>
              <span className={`value ${(mlStatus?.queue.active ?? 0) > 0 ? 'processing' : ''}`}>
                {mlStatus?.queue.active ?? 0}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Queue Capacity</span>
              <span className="value">
                {mlStatus?.queue.max_size ?? 100}
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

        {/* Workers */}
        <div className="ml-status-card">
          <h3>Workers</h3>
          {mlStatus && mlStatus.worker_status.length > 0 ? (
            <div className="workers-grid">
              {mlStatus.worker_status.map((worker) => (
                <div
                  key={worker.id}
                  className={`worker-item ${worker.current_job ? 'busy' : 'idle'}`}
                >
                  <span className="worker-id">Worker {worker.id}</span>
                  <span className={`worker-status ${worker.running ? 'running' : 'stopped'}`}>
                    {worker.current_job
                      ? `Job #${worker.current_job}`
                      : worker.running
                      ? 'Idle'
                      : 'Stopped'}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="no-data">No workers configured</p>
          )}
        </div>

        {/* Recent Jobs */}
        <div className="ml-status-card wide">
          <h3>Recent Jobs ({jobsTotal} total)</h3>
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
