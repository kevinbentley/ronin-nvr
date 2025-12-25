/**
 * Hook for managing camera data.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type { Camera, RecordingStatus } from '../types/camera';

interface UseCamerasResult {
  cameras: Camera[];
  recordingStatus: Map<number, RecordingStatus>;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useCameras(pollInterval: number = 5000): UseCamerasResult {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [recordingStatus, setRecordingStatus] = useState<Map<number, RecordingStatus>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [cameraList, statusList] = await Promise.all([
        api.listCameras(),
        api.getAllRecordingStatus(),
      ]);

      setCameras(cameraList);

      const statusMap = new Map<number, RecordingStatus>();
      statusList.forEach((status) => {
        statusMap.set(status.camera_id, status);
      });
      setRecordingStatus(statusMap);

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch cameras');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();

    const interval = setInterval(fetchData, pollInterval);
    return () => clearInterval(interval);
  }, [fetchData, pollInterval]);

  return {
    cameras,
    recordingStatus,
    loading,
    error,
    refresh: fetchData,
  };
}
