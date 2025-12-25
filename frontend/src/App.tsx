/**
 * RoninNVR main application component.
 */

import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { CameraGrid } from './components/CameraGrid';
import { CameraSidebar } from './components/CameraSidebar';
import { CameraModal } from './components/CameraModal';
import { PlaybackPage } from './pages/PlaybackPage';
import { useCameras } from './hooks/useCameras';
import type { Camera, GridLayout } from './types/camera';
import './App.css';

type Page = 'live' | 'playback';

function App() {
  const { cameras, recordingStatus, loading, error, refresh } = useCameras();
  const [currentPage, setCurrentPage] = useState<Page>('live');
  const [layout, setLayout] = useState<GridLayout>(() => {
    const saved = localStorage.getItem('gridLayout');
    return (saved as GridLayout) || '2x2';
  });
  const [showModal, setShowModal] = useState(false);
  const [editingCamera, setEditingCamera] = useState<Camera | undefined>();

  useEffect(() => {
    localStorage.setItem('gridLayout', layout);
  }, [layout]);

  const handleAddCamera = () => {
    setEditingCamera(undefined);
    setShowModal(true);
  };

  const handleEditCamera = (camera: Camera) => {
    setEditingCamera(camera);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setEditingCamera(undefined);
  };

  const handleSaveCamera = () => {
    refresh();
  };

  if (loading && currentPage === 'live') {
    return (
      <div className="app loading">
        <div className="loading-spinner">Loading...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <Header
        layout={layout}
        onLayoutChange={setLayout}
        onAddCamera={handleAddCamera}
        currentPage={currentPage}
        onPageChange={setCurrentPage}
      />

      {currentPage === 'live' ? (
        <div className="app-content">
          <CameraSidebar
            cameras={cameras}
            recordingStatus={recordingStatus}
            onEditCamera={handleEditCamera}
            onRefresh={refresh}
          />

          <main className="main-content">
            {error && (
              <div className="error-banner">
                {error}
                <button onClick={refresh}>Retry</button>
              </div>
            )}

            <CameraGrid
              cameras={cameras}
              recordingStatus={recordingStatus}
              layout={layout}
            />
          </main>
        </div>
      ) : (
        <PlaybackPage />
      )}

      {showModal && (
        <CameraModal
          camera={editingCamera}
          onClose={handleCloseModal}
          onSave={handleSaveCamera}
        />
      )}
    </div>
  );
}

export default App;
