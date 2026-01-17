/**
 * RoninNVR main application component.
 */

import { useState, useEffect, useCallback } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { Header, type Page } from './components/Header';
import { CameraGrid } from './components/CameraGrid';
import { CameraSidebar } from './components/CameraSidebar';
import { PlaybackPage } from './pages/PlaybackPage';
import { StatusPage } from './pages/StatusPage';
import { MLStatusPage } from './pages/MLStatusPage';
import { SetupPage } from './pages/SetupPage';
import { LoginPage } from './pages/LoginPage';
import { useCameras } from './hooks/useCameras';
import type { GridLayout } from './types/camera';
import './App.css';

function AppContent() {
  const { isAuthenticated, isLoading: authLoading, user, logout } = useAuth();
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    const saved = localStorage.getItem('sidebarCollapsed');
    return saved === 'true';
  });

  useEffect(() => {
    localStorage.setItem('gridLayout', layout);
  }, [layout]);

  useEffect(() => {
    localStorage.setItem('hiddenCameras', JSON.stringify([...hiddenCameraIds]));
  }, [hiddenCameraIds]);

  useEffect(() => {
    localStorage.setItem('sidebarCollapsed', String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  const handleToggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => !prev);
  }, []);

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

  // Show loading spinner while checking auth
  if (authLoading) {
    return (
      <div className="app loading">
        <div className="loading-spinner" />
        <div className="loading-text">Loading...</div>
      </div>
    );
  }

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return <LoginPage />;
  }

  if (loading && currentPage === 'live') {
    return (
      <div className="app loading">
        <div className="loading-spinner" />
        <div className="loading-text">Loading...</div>
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
              isCollapsed={sidebarCollapsed}
              onToggleCollapse={handleToggleSidebar}
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
      case 'ml':
        return <MLStatusPage />;
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
        user={user}
        onLogout={logout}
      />
      {renderPage()}
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
