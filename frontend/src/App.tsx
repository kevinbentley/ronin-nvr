/**
 * RoninNVR main application component.
 */

import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { CameraGrid } from './components/CameraGrid';
import { CameraSidebar } from './components/CameraSidebar';
import { CameraModal } from './components/CameraModal';
import { useCameras } from './hooks/useCameras';
import type { Camera, GridLayout } from './types/camera';
import './App.css';

function App() {
  const { cameras, recordingStatus, loading, error, refresh } = useCameras();
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

  if (loading) {
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
      />

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
