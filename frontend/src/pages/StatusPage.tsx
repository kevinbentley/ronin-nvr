/**
 * Status page showing system health, storage usage, and camera status.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type { StorageStats, RecordingStatus, Camera } from '../types/camera';
import './StatusPage.css';

interface StatusPageProps {
  cameras: Camera[];
  recordingStatus: Map<number, RecordingStatus>;
}

interface HealthStatus {
  status: string;
  database: string;
}

export function StatusPage({ cameras, recordingStatus }: StatusPageProps) {
  const [storageStats, setStorageStats] = useState<StorageStats | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cleanupResult, setCleanupResult] = useState<string | null>(null);
  const [cleaning, setCleaning] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [storage, health] = await Promise.all([
        api.getStorageStats(),
        api.getHealth(),
      ]);
      setStorageStats(storage);
      setHealthStatus(health);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    // Refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleCleanup = async () => {
    setCleaning(true);
    setCleanupResult(null);
    try {
      const result = await api.runStorageCleanup();
      setCleanupResult(
        `Cleaned ${result.files_deleted} files, freed ${result.gb_freed.toFixed(2)} GB`
      );
      // Refresh storage stats
      const storage = await api.getStorageStats();
      setStorageStats(storage);
    } catch (err) {
      setCleanupResult(
        `Cleanup failed: ${err instanceof Error ? err.message : 'Unknown error'}`
      );
    } finally {
      setCleaning(false);
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes >= 1024 * 1024 * 1024) {
      return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    }
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return `${(bytes / 1024).toFixed(0)} KB`;
  };

  const formatDate = (dateStr: string | undefined): string => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleString();
  };

  if (loading) {
    return (
      <div className="status-page loading">
        <div className="loading-spinner">Loading status...</div>
      </div>
    );
  }

  const onlineCameras = cameras.filter((c) => c.status === 'online').length;
  const recordingCameras = Array.from(recordingStatus.values()).filter(
    (s) => s.is_recording
  ).length;

  return (
    <div className="status-page">
      <div className="status-header">
        <h2>System Status</h2>
        <button className="refresh-button" onClick={loadData}>
          Refresh
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="status-grid">
        {/* System Health */}
        <div className="status-card">
          <h3>System Health</h3>
          <div className="status-items">
            <div className="status-item">
              <span className="label">API Status</span>
              <span className={`value ${healthStatus?.status === 'ok' ? 'good' : 'bad'}`}>
                {healthStatus?.status === 'ok' ? 'Online' : 'Error'}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Database</span>
              <span className={`value ${healthStatus?.database === 'connected' ? 'good' : 'bad'}`}>
                {healthStatus?.database === 'connected' ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>

        {/* Camera Status */}
        <div className="status-card">
          <h3>Cameras</h3>
          <div className="status-items">
            <div className="status-item">
              <span className="label">Total Cameras</span>
              <span className="value">{cameras.length}</span>
            </div>
            <div className="status-item">
              <span className="label">Online</span>
              <span className={`value ${onlineCameras === cameras.length ? 'good' : 'warning'}`}>
                {onlineCameras} / {cameras.length}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Recording</span>
              <span className={`value ${recordingCameras > 0 ? 'recording' : ''}`}>
                {recordingCameras}
              </span>
            </div>
          </div>
        </div>

        {/* Storage Overview */}
        <div className="status-card wide">
          <h3>Storage</h3>
          <div className="status-items">
            <div className="status-item">
              <span className="label">Total Used</span>
              <span className="value">{storageStats?.total_size_gb.toFixed(2)} GB</span>
            </div>
            <div className="status-item">
              <span className="label">Total Files</span>
              <span className="value">{storageStats?.total_files.toLocaleString()}</span>
            </div>
            <div className="status-item">
              <span className="label">Oldest Recording</span>
              <span className="value">{formatDate(storageStats?.oldest_file)}</span>
            </div>
            <div className="status-item">
              <span className="label">Newest Recording</span>
              <span className="value">{formatDate(storageStats?.newest_file)}</span>
            </div>
          </div>
          <div className="storage-actions">
            <button
              className="cleanup-button"
              onClick={handleCleanup}
              disabled={cleaning}
            >
              {cleaning ? 'Cleaning...' : 'Run Cleanup'}
            </button>
            {cleanupResult && (
              <span className="cleanup-result">{cleanupResult}</span>
            )}
          </div>
        </div>

        {/* Per-Camera Storage */}
        <div className="status-card wide">
          <h3>Storage by Camera</h3>
          {storageStats?.cameras && storageStats.cameras.length > 0 ? (
            <table className="storage-table">
              <thead>
                <tr>
                  <th>Camera</th>
                  <th>Size</th>
                  <th>Files</th>
                  <th>% of Total</th>
                </tr>
              </thead>
              <tbody>
                {storageStats.cameras
                  .sort((a, b) => b.size_bytes - a.size_bytes)
                  .map((cam) => (
                    <tr key={cam.name}>
                      <td>{cam.name}</td>
                      <td>{formatBytes(cam.size_bytes)}</td>
                      <td>{cam.file_count}</td>
                      <td>
                        {storageStats.total_size_bytes > 0
                          ? ((cam.size_bytes / storageStats.total_size_bytes) * 100).toFixed(1)
                          : 0}
                        %
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          ) : (
            <p className="no-data">No recordings yet</p>
          )}
        </div>

        {/* Active Streams */}
        <div className="status-card wide">
          <h3>Active Streams</h3>
          {cameras.length > 0 ? (
            <table className="streams-table">
              <thead>
                <tr>
                  <th>Camera</th>
                  <th>Status</th>
                  <th>Recording</th>
                  <th>Started</th>
                </tr>
              </thead>
              <tbody>
                {cameras.map((camera) => {
                  const status = recordingStatus.get(camera.id);
                  return (
                    <tr key={camera.id}>
                      <td>{camera.name}</td>
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
                      <td>
                        {status?.started_at
                          ? new Date(status.started_at).toLocaleTimeString()
                          : '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <p className="no-data">No cameras configured</p>
          )}
        </div>
      </div>
    </div>
  );
}
