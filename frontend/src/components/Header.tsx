/**
 * Application header component.
 */

import { LayoutSelector } from './LayoutSelector';
import type { GridLayout } from '../types/camera';
import type { User } from '../services/api';
import './Header.css';

export type Page = 'live' | 'playback' | 'status' | 'ml' | 'analysis' | 'setup';

interface HeaderProps {
  layout: GridLayout;
  onLayoutChange: (layout: GridLayout) => void;
  currentPage: Page;
  onPageChange: (page: Page) => void;
  user: User | null;
  onLogout: () => void;
}

export function Header({
  layout,
  onLayoutChange,
  currentPage,
  onPageChange,
  user,
  onLogout,
}: HeaderProps) {
  return (
    <header className="app-header">
      <div className="header-left">
        <h1 className="app-title" onClick={() => window.location.reload()}>
          RoninNVR
        </h1>
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
          <button
            className={`nav-button ${currentPage === 'status' ? 'active' : ''}`}
            onClick={() => onPageChange('status')}
          >
            Status
          </button>
          <button
            className={`nav-button ${currentPage === 'ml' ? 'active' : ''}`}
            onClick={() => onPageChange('ml')}
          >
            ML
          </button>
          <button
            className={`nav-button ${currentPage === 'analysis' ? 'active' : ''}`}
            onClick={() => onPageChange('analysis')}
          >
            Deep Analysis
          </button>
          <button
            className={`nav-button ${currentPage === 'setup' ? 'active' : ''}`}
            onClick={() => onPageChange('setup')}
          >
            Setup
          </button>
        </nav>
      </div>

      <div className="header-center">
        {currentPage === 'live' && (
          <LayoutSelector layout={layout} onChange={onLayoutChange} />
        )}
      </div>

      <div className="header-right">
        {user && (
          <div className="user-info">
            <span className="username">
              {user.username}
              {user.is_admin && <span className="admin-badge">Admin</span>}
            </span>
            <button className="logout-button" onClick={onLogout}>
              Logout
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
