/**
 * Storage page showing tiered storage management and offline export.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type {
  Camera,
  TierConfigResponse,
  TierStatsResponse,
  TierStats,
  MigrationResult,
  OfflineExportResponse,
} from '../types/camera';
import './StoragePage.css';

interface StoragePageProps {
  cameras: Camera[];
}

export function StoragePage({ cameras }: StoragePageProps) {
  const [tierConfig, setTierConfig] = useState<TierConfigResponse | null>(null);
  const [tierStats, setTierStats] = useState<TierStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Migration state
  const [migrating, setMigrating] = useState(false);
  const [migrationResult, setMigrationResult] = useState<MigrationResult | null>(null);

  // Export state
  const [exportCameraIds, setExportCameraIds] = useState<number[]>([]);
  const [exportStartDate, setExportStartDate] = useState('');
  const [exportEndDate, setExportEndDate] = useState('');
  const [exportOutputPath, setExportOutputPath] = useState('');
  const [exportIncludeDetections, setExportIncludeDetections] = useState(true);
  const [exportIncludeSnapshots, setExportIncludeSnapshots] = useState(true);
  const [exportDeleteAfterCopy, setExportDeleteAfterCopy] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<OfflineExportResponse | null>(null);

  // Config editing state
  const [editingConfig, setEditingConfig] = useState(false);
  const [hotMaxGb, setHotMaxGb] = useState<string>('');
  const [hotRetentionDays, setHotRetentionDays] = useState<string>('');
  const [warmMaxGb, setWarmMaxGb] = useState<string>('');
  const [warmRetentionDays, setWarmRetentionDays] = useState<string>('');
  const [savingConfig, setSavingConfig] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [config, stats] = await Promise.all([
        api.getTierConfig(),
        api.getTierStats(),
      ]);
      setTierConfig(config);
      setTierStats(stats);

      // Initialize edit form values
      setHotMaxGb(config.hot_max_gb?.toString() || '');
      setHotRetentionDays(config.hot_retention_days?.toString() || '');
      setWarmMaxGb(config.warm_max_gb?.toString() || '');
      setWarmRetentionDays(config.warm_retention_days?.toString() || '');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load storage data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleSaveConfig = async () => {
    setSavingConfig(true);
    try {
      const config = await api.updateTierConfig({
        hot_max_gb: hotMaxGb ? parseFloat(hotMaxGb) : null,
        hot_retention_days: hotRetentionDays ? parseInt(hotRetentionDays, 10) : null,
        warm_max_gb: warmMaxGb ? parseFloat(warmMaxGb) : null,
        warm_retention_days: warmRetentionDays ? parseInt(warmRetentionDays, 10) : null,
      });
      setTierConfig(config);
      setEditingConfig(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
    } finally {
      setSavingConfig(false);
    }
  };

  const handleMigrate = async (fromTier: 'hot' | 'warm') => {
    setMigrating(true);
    setMigrationResult(null);
    setError(null);
    try {
      const result = await api.triggerMigration({
        from_tier: fromTier,
        max_files: 100,
      });
      setMigrationResult(result);
      // If migration started in background, we'll poll for status
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Migration failed');
      setMigrating(false);
    }
  };

  // Poll for migration status while migration is in progress
  useEffect(() => {
    if (!migrating || migrationResult?.status === 'completed') {
      return;
    }

    const pollStatus = async () => {
      try {
        const status = await api.getMigrationStatus();
        setMigrationResult(status);
        if (status.status !== 'in_progress') {
          setMigrating(false);
          loadData(); // Refresh tier stats when done
        }
      } catch (err) {
        console.error('Failed to poll migration status:', err);
      }
    };

    const interval = setInterval(pollStatus, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, [migrating, migrationResult?.status, loadData]);

  const handleExport = async () => {
    if (exportCameraIds.length === 0) {
      setError('Please select at least one camera');
      return;
    }
    if (!exportStartDate || !exportEndDate) {
      setError('Please select start and end dates');
      return;
    }
    if (!exportOutputPath) {
      setError('Please enter an output path');
      return;
    }

    setExporting(true);
    setExportResult(null);
    setError(null);

    try {
      const result = await api.createOfflineExport({
        camera_ids: exportCameraIds,
        start_time: new Date(exportStartDate).toISOString(),
        end_time: new Date(exportEndDate).toISOString(),
        output_path: exportOutputPath,
        include_detections: exportIncludeDetections,
        include_snapshots: exportIncludeSnapshots,
        delete_after_copy: exportDeleteAfterCopy,
      });
      setExportResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setExporting(false);
    }
  };

  const handleCameraToggle = (cameraId: number) => {
    setExportCameraIds((prev) =>
      prev.includes(cameraId)
        ? prev.filter((id) => id !== cameraId)
        : [...prev, cameraId]
    );
  };

  const handleSelectAllCameras = () => {
    if (exportCameraIds.length === cameras.length) {
      setExportCameraIds([]);
    } else {
      setExportCameraIds(cameras.map((c) => c.id));
    }
  };

  const formatDate = (dateStr: string | null): string => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleString();
  };

  const renderTierCard = (
    title: string,
    tier: TierStats | null,
    tierName: string,
    canMigrate: boolean
  ) => {
    if (!tier) {
      return (
        <div className="tier-card disabled">
          <h3>{title}</h3>
          <div className="tier-status">Not Enabled</div>
          <p className="tier-hint">
            {tierName === 'warm'
              ? 'Set WARM_STORAGE_ENABLED=true in environment'
              : 'Set COLD_STORAGE_ENABLED=true and configure S3 in environment'}
          </p>
        </div>
      );
    }

    return (
      <div className={`tier-card ${tierName}`}>
        <h3>{title}</h3>
        <div className="tier-stats">
          <div className="stat-row">
            <span className="label">Total Size</span>
            <span className="value">{tier.total_size_gb.toFixed(2)} GB</span>
          </div>
          <div className="stat-row">
            <span className="label">Files</span>
            <span className="value">{tier.file_count.toLocaleString()}</span>
          </div>
          {tier.max_size_gb && (
            <div className="stat-row">
              <span className="label">Capacity</span>
              <span className={`value ${(tier.percent_full || 0) > 80 ? 'warning' : ''}`}>
                {tier.percent_full?.toFixed(1)}% of {tier.max_size_gb} GB
              </span>
            </div>
          )}
          {tier.retention_days && (
            <div className="stat-row">
              <span className="label">Retention</span>
              <span className="value">{tier.retention_days} days</span>
            </div>
          )}
          <div className="stat-row">
            <span className="label">Oldest File</span>
            <span className="value small">{formatDate(tier.oldest_file)}</span>
          </div>
          <div className="stat-row">
            <span className="label">Newest File</span>
            <span className="value small">{formatDate(tier.newest_file)}</span>
          </div>
        </div>
        {canMigrate && tier.file_count > 0 && (
          <button
            className="migrate-button"
            onClick={() => handleMigrate(tierName as 'hot' | 'warm')}
            disabled={migrating}
          >
            {migrating ? 'Migrating...' : `Migrate Files`}
          </button>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="storage-page loading">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <div className="loading-text">Loading storage data...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="storage-page">
      <div className="storage-header">
        <h2>Tiered Storage Management</h2>
        <button className="refresh-button" onClick={loadData}>
          Refresh
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {migrationResult && (
        <div className={`migration-result ${migrationResult.status === 'in_progress' ? 'in-progress' : ''}`}>
          {migrationResult.status === 'in_progress' ? (
            <>
              <strong>Migration in progress...</strong>{' '}
              {migrationResult.files_migrated} files ({migrationResult.bytes_migrated_gb.toFixed(2)} GB) migrated so far
              {migrationResult.message && <div className="message">{migrationResult.message}</div>}
            </>
          ) : (
            <>
              Migrated {migrationResult.files_migrated} files (
              {migrationResult.bytes_migrated_gb.toFixed(2)} GB)
              {migrationResult.files_skipped > 0 && (
                <span className="skipped">
                  , {migrationResult.files_skipped} skipped (missing files)
                </span>
              )}
              {migrationResult.orphans_cleaned > 0 && (
                <span className="cleaned">
                  , {migrationResult.orphans_cleaned} orphan records cleaned
                </span>
              )}
              {migrationResult.files_failed > 0 && (
                <span className="failed">, {migrationResult.files_failed} failed</span>
              )}
              {migrationResult.message && <div className="message">{migrationResult.message}</div>}
            </>
          )}
        </div>
      )}

      {/* Filesystem Overview */}
      {tierStats && (
        <div className="filesystem-overview">
          <div className="fs-stat">
            <span className="fs-label">Total Disk Usage</span>
            <span className="fs-value">{tierStats.filesystem_total_gb.toFixed(2)} GB</span>
          </div>
          <div className="fs-stat">
            <span className="fs-label">Total Files</span>
            <span className="fs-value">{tierStats.filesystem_total_files.toLocaleString()}</span>
          </div>
          <div className="fs-stat">
            <span className="fs-label">DB-Tracked</span>
            <span className="fs-value">{tierStats.hot.file_count.toLocaleString()} files ({tierStats.hot.total_size_gb.toFixed(2)} GB)</span>
          </div>
        </div>
      )}

      {/* Storage Tiers */}
      <div className="tiers-section">
        <h3>Storage Tiers (Database-Tracked)</h3>
        <div className="tiers-grid">
          {renderTierCard(
            'Hot Storage (Primary)',
            tierStats?.hot || null,
            'hot',
            (tierConfig?.warm_storage_enabled || tierConfig?.cold_storage_enabled) || false
          )}
          {renderTierCard(
            'Warm Storage (Secondary)',
            tierStats?.warm || null,
            'warm',
            tierConfig?.cold_storage_enabled || false
          )}
          {renderTierCard('Cold Storage (S3)', tierStats?.cold || null, 'cold', false)}
        </div>
      </div>

      {/* Tier Configuration */}
      <div className="config-section">
        <div className="section-header">
          <h3>Migration Thresholds</h3>
          {!editingConfig ? (
            <button className="edit-button" onClick={() => setEditingConfig(true)}>
              Edit
            </button>
          ) : (
            <div className="edit-actions">
              <button
                className="save-button"
                onClick={handleSaveConfig}
                disabled={savingConfig}
              >
                {savingConfig ? 'Saving...' : 'Save'}
              </button>
              <button
                className="cancel-button"
                onClick={() => setEditingConfig(false)}
                disabled={savingConfig}
              >
                Cancel
              </button>
            </div>
          )}
        </div>

        <div className="config-grid">
          <div className="config-card">
            <h4>Hot Storage Thresholds</h4>
            <p className="hint">Files migrate from hot when these thresholds are exceeded</p>
            <div className="config-field">
              <label>Max Size (GB)</label>
              {editingConfig ? (
                <input
                  type="number"
                  value={hotMaxGb}
                  onChange={(e) => setHotMaxGb(e.target.value)}
                  placeholder="Unlimited"
                  min="1"
                />
              ) : (
                <span>{tierConfig?.hot_max_gb || 'Unlimited'}</span>
              )}
            </div>
            <div className="config-field">
              <label>Max Age (days)</label>
              {editingConfig ? (
                <input
                  type="number"
                  value={hotRetentionDays}
                  onChange={(e) => setHotRetentionDays(e.target.value)}
                  placeholder="Unlimited"
                  min="1"
                />
              ) : (
                <span>{tierConfig?.hot_retention_days || 'Unlimited'}</span>
              )}
            </div>
          </div>

          {tierConfig?.warm_storage_enabled && (
            <div className="config-card">
              <h4>Warm Storage Thresholds</h4>
              <p className="hint">Files migrate from warm to cold when exceeded</p>
              <div className="config-field">
                <label>Max Size (GB)</label>
                {editingConfig ? (
                  <input
                    type="number"
                    value={warmMaxGb}
                    onChange={(e) => setWarmMaxGb(e.target.value)}
                    placeholder="Unlimited"
                    min="1"
                  />
                ) : (
                  <span>{tierConfig?.warm_max_gb || 'Unlimited'}</span>
                )}
              </div>
              <div className="config-field">
                <label>Max Age (days)</label>
                {editingConfig ? (
                  <input
                    type="number"
                    value={warmRetentionDays}
                    onChange={(e) => setWarmRetentionDays(e.target.value)}
                    placeholder="Unlimited"
                    min="1"
                  />
                ) : (
                  <span>{tierConfig?.warm_retention_days || 'Unlimited'}</span>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="config-info">
          <p>
            <strong>Migration Interval:</strong>{' '}
            {tierConfig?.tier_migration_check_interval_minutes} minutes
          </p>
          {tierConfig?.warm_storage_enabled && tierConfig.warm_storage_path && (
            <p>
              <strong>Warm Storage Path:</strong> {tierConfig.warm_storage_path}
            </p>
          )}
          {tierConfig?.cold_storage_enabled && (
            <p>
              <strong>S3 Bucket:</strong>{' '}
              {tierConfig.s3_configured
                ? `${tierConfig.s3_bucket_name} (${tierConfig.s3_endpoint_url || 'AWS S3'})`
                : 'Not configured'}
            </p>
          )}
        </div>
      </div>

      {/* Offline Export */}
      <div className="export-section">
        <h3>Offline Export</h3>
        <p className="section-hint">
          Export recordings and detection data to removable media
        </p>

        <div className="export-form">
          <div className="form-group">
            <label>Cameras</label>
            <div className="camera-select">
              <button
                className="select-all-button"
                onClick={handleSelectAllCameras}
              >
                {exportCameraIds.length === cameras.length ? 'Deselect All' : 'Select All'}
              </button>
              <div className="camera-checkboxes">
                {cameras.map((camera) => (
                  <label key={camera.id} className="camera-checkbox">
                    <input
                      type="checkbox"
                      checked={exportCameraIds.includes(camera.id)}
                      onChange={() => handleCameraToggle(camera.id)}
                    />
                    {camera.name}
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Start Date/Time</label>
              <input
                type="datetime-local"
                value={exportStartDate}
                onChange={(e) => setExportStartDate(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label>End Date/Time</label>
              <input
                type="datetime-local"
                value={exportEndDate}
                onChange={(e) => setExportEndDate(e.target.value)}
              />
            </div>
          </div>

          <div className="form-group">
            <label>Output Path</label>
            <input
              type="text"
              value={exportOutputPath}
              onChange={(e) => setExportOutputPath(e.target.value)}
              placeholder="/mnt/usb"
            />
            <span className="hint">Path to mounted external drive</span>
          </div>

          <div className="form-group checkboxes">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={exportIncludeDetections}
                onChange={(e) => setExportIncludeDetections(e.target.checked)}
              />
              Include detection events
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={exportIncludeSnapshots}
                onChange={(e) => setExportIncludeSnapshots(e.target.checked)}
              />
              Include detection snapshots
            </label>
            <label className="checkbox warning">
              <input
                type="checkbox"
                checked={exportDeleteAfterCopy}
                onChange={(e) => setExportDeleteAfterCopy(e.target.checked)}
              />
              Delete source files after copy
            </label>
          </div>

          <button
            className="export-button"
            onClick={handleExport}
            disabled={exporting}
          >
            {exporting ? 'Exporting...' : 'Start Export'}
          </button>
        </div>

        {exportResult && (
          <div className={`export-result ${exportResult.success ? 'success' : 'error'}`}>
            {exportResult.success ? (
              <>
                <h4>Export Complete</h4>
                <p>
                  Exported {exportResult.files_exported} files (
                  {exportResult.bytes_exported_gb.toFixed(2)} GB)
                </p>
                <p>
                  {exportResult.events_exported} detection events,{' '}
                  {exportResult.snapshots_exported} snapshots
                </p>
                <p>
                  <strong>Output:</strong> {exportResult.output_path}
                </p>
              </>
            ) : (
              <>
                <h4>Export Failed</h4>
                <p>{exportResult.error_message}</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
