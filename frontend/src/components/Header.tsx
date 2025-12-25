/**
 * Application header component.
 */

import { LayoutSelector } from './LayoutSelector';
import type { GridLayout } from '../types/camera';
import './Header.css';

type Page = 'live' | 'playback';

interface HeaderProps {
  layout: GridLayout;
  onLayoutChange: (layout: GridLayout) => void;
  onAddCamera: () => void;
  currentPage: Page;
  onPageChange: (page: Page) => void;
}

export function Header({
  layout,
  onLayoutChange,
  onAddCamera,
  currentPage,
  onPageChange,
}: HeaderProps) {
  return (
    <header className="app-header">
      <div className="header-left">
        <h1 className="app-title">RoninNVR</h1>
        <nav className="header-nav">
          <button
            className={`nav-button ${currentPage === 'live' ? 'active' : ''}`}
            onClick={() => onPageChange('live')}
          >
            Live View
          </button>
          <button
            className={`nav-button ${currentPage === 'playback' ? 'active' : ''}`}
            onClick={() => onPageChange('playback')}
          >
            Playback
          </button>
        </nav>
      </div>

      <div className="header-center">
        {currentPage === 'live' && (
          <LayoutSelector layout={layout} onChange={onLayoutChange} />
        )}
      </div>

      <div className="header-right">
        {currentPage === 'live' && (
          <button className="add-camera-button" onClick={onAddCamera}>
            + Add Camera
          </button>
        )}
      </div>
    </header>
  );
}
