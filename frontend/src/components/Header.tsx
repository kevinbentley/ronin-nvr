/**
 * Application header component.
 */

import { LayoutSelector } from './LayoutSelector';
import type { GridLayout } from '../types/camera';
import './Header.css';

interface HeaderProps {
  layout: GridLayout;
  onLayoutChange: (layout: GridLayout) => void;
  onAddCamera: () => void;
  onShowSettings?: () => void;
}

export function Header({
  layout,
  onLayoutChange,
  onAddCamera,
}: HeaderProps) {
  return (
    <header className="app-header">
      <div className="header-left">
        <h1 className="app-title">RoninNVR</h1>
      </div>

      <div className="header-center">
        <LayoutSelector layout={layout} onChange={onLayoutChange} />
      </div>

      <div className="header-right">
        <button className="add-camera-button" onClick={onAddCamera}>
          + Add Camera
        </button>
      </div>
    </header>
  );
}
