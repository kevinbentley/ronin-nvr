/**
 * ML Status page showing live detection settings and recent detections.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type {
  LiveDetection,
  MLSettings,
} from '../types/camera';
import './MLStatusPage.css';

// Available detection classes from YOLO COCO dataset
const AVAILABLE_CLASSES = [
  'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
  'dog', 'cat', 'bird', 'backpack', 'handbag', 'suitcase',
];

// Classes that support per-class confidence thresholds
const THRESHOLD_CLASSES = ['person', 'car', 'truck', 'dog', 'cat'];

const DETECTIONS_PER_PAGE = 100;

export function MLStatusPage() {
  const [recentDetections, setRecentDetections] = useState<LiveDetection[]>([]);
  const [detectionsTotal, setDetectionsTotal] = useState(0);
  const [detectionsPage, setDetectionsPage] = useState(0);
  const [selectedDetection, setSelectedDetection] = useState<LiveDetection | null>(null);
  const [zoomedImage, setZoomedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Settings state
  const [settings, setSettings] = useState<MLSettings | null>(null);
  const [editedSettings, setEditedSettings] = useState<MLSettings | null>(null);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [detections, settingsResponse] = await Promise.all([
        api.getLiveDetections({
          limit: DETECTIONS_PER_PAGE,
          offset: detectionsPage * DETECTIONS_PER_PAGE,
        }),
        api.getMLSettings(),
      ]);
      setRecentDetections(detections.detections);
      setDetectionsTotal(detections.total);
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
  }, [editedSettings, detectionsPage]);

  useEffect(() => {
    loadData();
    // Refresh every 5 seconds for real-time updates
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

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
        class_thresholds: editedSettings.class_thresholds,
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
      JSON.stringify(settings.class_thresholds) !== JSON.stringify(editedSettings.class_thresholds)
    );
  };

  const toggleClass = (classList: string[], className: string): string[] => {
    if (classList.includes(className)) {
      return classList.filter(c => c !== className);
    }
    return [...classList, className];
  };

  const updateClassThreshold = (className: string, value: number) => {
    if (!editedSettings) return;
    setEditedSettings({
      ...editedSettings,
      class_thresholds: {
        ...editedSettings.class_thresholds,
        [className]: value,
      },
    });
  };

  // Close lightbox on Escape key
  useEffect(() => {
    if (!zoomedImage) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setZoomedImage(null);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [zoomedImage]);

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
        <h2>ML Detection</h2>
        <div className="ml-controls">
          <button className="refresh-button" onClick={loadData}>
            Refresh
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="ml-status-grid">
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
                <h4>Detection</h4>

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
                    Default Confidence
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

              {/* Per-Class Confidence Thresholds */}
              <div className="settings-section">
                <h4>Per-Class Confidence</h4>
                <p className="settings-section-note">
                  Override confidence threshold for specific classes (lower = more sensitive)
                </p>

                {THRESHOLD_CLASSES.map((cls) => (
                  <div key={cls} className="setting-row">
                    <label className="setting-label">
                      {cls}
                      <span className="setting-value">
                        {((editedSettings.class_thresholds?.[cls] ?? editedSettings.live_detection_confidence) * 100).toFixed(0)}%
                      </span>
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      value={(editedSettings.class_thresholds?.[cls] ?? editedSettings.live_detection_confidence) * 100}
                      onChange={(e) => updateClassThreshold(cls, parseInt(e.target.value) / 100)}
                      className="setting-slider"
                    />
                  </div>
                ))}
              </div>

              <div className="settings-note">
                Changes take effect within 60 seconds
              </div>
            </div>
          )}
        </div>

        {/* Recent Live Detections */}
        <div className="ml-status-card detections-card">
          <div className="detections-header">
            <h3>Live Detections ({detectionsTotal} total)</h3>
            {detectionsTotal > DETECTIONS_PER_PAGE && (
              <div className="pagination-controls">
                <button
                  className="pagination-button"
                  onClick={() => setDetectionsPage((p) => Math.max(0, p - 1))}
                  disabled={detectionsPage === 0}
                >
                  Previous
                </button>
                <span className="pagination-info">
                  Page {detectionsPage + 1} of{' '}
                  {Math.ceil(detectionsTotal / DETECTIONS_PER_PAGE)}
                </span>
                <button
                  className="pagination-button"
                  onClick={() =>
                    setDetectionsPage((p) =>
                      Math.min(
                        Math.ceil(detectionsTotal / DETECTIONS_PER_PAGE) - 1,
                        p + 1
                      )
                    )
                  }
                  disabled={
                    detectionsPage >=
                    Math.ceil(detectionsTotal / DETECTIONS_PER_PAGE) - 1
                  }
                >
                  Next
                </button>
              </div>
            )}
          </div>
          {recentDetections.length > 0 ? (
            <div className="recent-detections">
              {recentDetections.map((det) => (
                <div
                  key={det.id}
                  className={`detection-item ${selectedDetection?.id === det.id ? 'selected' : ''}`}
                  onClick={() => setSelectedDetection(det)}
                >
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

        {/* Detection Snapshot Preview */}
        <div className="ml-status-card snapshot-preview-card">
          <h3>Detection Snapshot</h3>
          {selectedDetection ? (
            <div className="snapshot-preview">
              {selectedDetection.snapshot_url ? (
                <img
                  src={selectedDetection.snapshot_url}
                  alt={`Detection: ${selectedDetection.class_name}`}
                  className="snapshot-image clickable"
                  onClick={() => setZoomedImage(selectedDetection.snapshot_url)}
                  title="Click to enlarge"
                />
              ) : (
                <div className="no-snapshot">No snapshot available</div>
              )}
              <div className="snapshot-info">
                <div className="snapshot-detail">
                  <span className="snapshot-label">Camera</span>
                  <span className="snapshot-value">{selectedDetection.camera_name}</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Detected</span>
                  <span className="snapshot-value">{selectedDetection.class_name}</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Confidence</span>
                  <span className="snapshot-value">{(selectedDetection.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="snapshot-detail">
                  <span className="snapshot-label">Time</span>
                  <span className="snapshot-value">{formatDetectionTime(selectedDetection.detected_at)}</span>
                </div>
              </div>
            </div>
          ) : (
            <p className="no-data">Select a detection to view snapshot</p>
          )}
        </div>
      </div>

      {/* Image Lightbox */}
      {zoomedImage && (
        <div className="image-lightbox" onClick={() => setZoomedImage(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <img src={zoomedImage} alt="Zoomed detection snapshot" />
            <button
              className="lightbox-close"
              onClick={() => setZoomedImage(null)}
              aria-label="Close"
            >
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
