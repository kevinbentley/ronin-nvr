/**
 * Dropdown filter for selecting which object types to display in detection overlay.
 */

import { useState, useRef, useEffect } from 'react';
import type { ObjectTypeFilterProps } from './types';
import { getEventColor } from './types';
import './ObjectTypeFilter.css';

export function ObjectTypeFilter({
  visible,
  visibleTypes,
  typeCounts,
  onToggleType,
  onToggleAll,
}: ObjectTypeFilterProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  // Don't render if overlay is not visible or no detections
  if (!visible || typeCounts.size === 0) {
    return null;
  }

  // Sort types by count (descending)
  const sortedTypes = Array.from(typeCounts.entries())
    .sort((a, b) => b[1] - a[1]);

  const totalTypes = sortedTypes.length;
  const hiddenCount = totalTypes - visibleTypes.size;
  const allVisible = hiddenCount === 0;

  return (
    <div className="object-type-filter" ref={dropdownRef}>
      <button
        className={`filter-btn ${isOpen ? 'active' : ''} ${hiddenCount > 0 ? 'has-hidden' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title="Filter object types"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z" />
        </svg>
        {hiddenCount > 0 && (
          <span className="filter-badge">{hiddenCount}</span>
        )}
      </button>

      {isOpen && (
        <div className="filter-dropdown">
          <div className="filter-header">
            <span className="filter-title">Object Types</span>
            <button
              className="toggle-all-btn"
              onClick={() => onToggleAll(!allVisible)}
            >
              {allVisible ? 'Hide All' : 'Show All'}
            </button>
          </div>

          <div className="filter-list">
            {sortedTypes.map(([className, count]) => {
              const isVisible = visibleTypes.has(className);
              const color = getEventColor(className);

              return (
                <label key={className} className="filter-item">
                  <input
                    type="checkbox"
                    checked={isVisible}
                    onChange={() => onToggleType(className)}
                  />
                  <span
                    className="type-indicator"
                    style={{ backgroundColor: color }}
                  />
                  <span className="type-name">{className}</span>
                  <span className="type-count">({count})</span>
                </label>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
