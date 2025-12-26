/**
 * RoninNVR main application component.
 */

import { useState, useEffect, useCallback } from 'react';
import { Header, type Page } from './components/Header';
import { CameraGrid } from './components/CameraGrid';
import { CameraSidebar } from './components/CameraSidebar';
import { PlaybackPage } from './pages/PlaybackPage';
import { StatusPage } from './pages/StatusPage';
import { SetupPage } from './pages/SetupPage';
import { useCameras } from './hooks/useCameras';
import type { GridLayout } from './types/camera';
import './App.css';

function App() {
  const { cameras, recordingStatus, loading, error, refresh } = useCameras();
  const [currentPage, setCurrentPage] = useState<Page>('live');
  const [layout, setLayout] = useState<GridLayout>(() => {
    const saved = localStorage.getItem('gridLayout');
    return (saved as GridLayout) || '2x2';
  });
  const [hiddenCameraIds, setHiddenCameraIds] = useState<Set<number>>(() => {
    const saved = localStorage.getItem('hiddenCameras');
    if (saved) {
      try {
        return new Set(JSON.parse(saved));
      } catch {
        return new Set();
      }
    }
    return new Set();
  });

  useEffect(() => {
    localStorage.setItem('gridLayout', layout);
  }, [layout]);

  useEffect(() => {
    localStorage.setItem('hiddenCameras', JSON.stringify([...hiddenCameraIds]));
  }, [hiddenCameraIds]);

  const handleToggleVisibility = useCallback((cameraId: number) => {
    setHiddenCameraIds((prev) => {
      const next = new Set(prev);
      if (next.has(cameraId)) {
        next.delete(cameraId);
      } else {
        next.add(cameraId);
      }
      return next;
    });
  }, []);

  const handleShowAll = useCallback(() => {
    setHiddenCameraIds(new Set());
  }, []);

  const handleHideAll = useCallback(() => {
    setHiddenCameraIds(new Set(cameras.map((c) => c.id)));
  }, [cameras]);

  // Filter visible cameras for the grid
  const visibleCameras = cameras.filter((c) => !hiddenCameraIds.has(c.id));

  if (loading && currentPage === 'live') {
    return (
      <div className="app loading">
        <div className="loading-spinner">Loading...</div>
      </div>
    );
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'live':
        return (
          <div className="app-content">
            <CameraSidebar
              cameras={cameras}
              recordingStatus={recordingStatus}
              hiddenCameraIds={hiddenCameraIds}
              onToggleVisibility={handleToggleVisibility}
              onShowAll={handleShowAll}
              onHideAll={handleHideAll}
            />
            <main className="main-content">
              {error && (
                <div className="error-banner">
                  {error}
                  <button onClick={refresh}>Retry</button>
                </div>
              )}
              <CameraGrid
                cameras={visibleCameras}
                recordingStatus={recordingStatus}
                layout={layout}
              />
            </main>
          </div>
        );
      case 'playback':
        return <PlaybackPage />;
      case 'status':
        return (
          <StatusPage
            cameras={cameras}
            recordingStatus={recordingStatus}
          />
        );
      case 'setup':
        return (
          <SetupPage
            cameras={cameras}
            recordingStatus={recordingStatus}
            onRefresh={refresh}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="app">
      <Header
        layout={layout}
        onLayoutChange={setLayout}
        currentPage={currentPage}
        onPageChange={setCurrentPage}
      />
      {renderPage()}
    </div>
  );
}

export default App;
