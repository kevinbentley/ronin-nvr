/**
 * Grid layout selector component.
 */

import type { GridLayout } from '../types/camera';
import './LayoutSelector.css';

interface LayoutSelectorProps {
  layout: GridLayout;
  onChange: (layout: GridLayout) => void;
}

const layouts: GridLayout[] = ['1x1', '2x2', '3x3', '4x4'];

export function LayoutSelector({ layout, onChange }: LayoutSelectorProps) {
  return (
    <div className="layout-selector">
      {layouts.map((l) => (
        <button
          key={l}
          className={`layout-button ${layout === l ? 'active' : ''}`}
          onClick={() => onChange(l)}
        >
          {l}
        </button>
      ))}
    </div>
  );
}
