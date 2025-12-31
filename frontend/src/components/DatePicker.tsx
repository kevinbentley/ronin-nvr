/**
 * Simple date picker showing available recording dates.
 */

import { useMemo } from 'react';
import './DatePicker.css';

interface DatePickerProps {
  availableDates: string[];
  selectedDate: string;
  onSelectDate: (date: string) => void;
}

export function DatePicker({
  availableDates,
  selectedDate,
  onSelectDate,
}: DatePickerProps) {
  // Parse date string as local date (not UTC)
  // "2025-12-31" should show as Dec 31, not Dec 30 in western timezones
  const parseLocalDate = (dateStr: string) => {
    const [year, month, day] = dateStr.split('-').map(Number);
    return new Date(year, month - 1, day);
  };

  // Group dates by month for display
  const datesByMonth = useMemo(() => {
    const groups: Map<string, string[]> = new Map();

    availableDates.forEach((date) => {
      const d = parseLocalDate(date);
      const monthKey = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;

      if (!groups.has(monthKey)) {
        groups.set(monthKey, []);
      }
      groups.get(monthKey)!.push(date);
    });

    return groups;
  }, [availableDates]);

  if (availableDates.length === 0) {
    return (
      <div className="date-picker empty">
        <p>No recordings available</p>
      </div>
    );
  }

  const formatDay = (dateStr: string) => {
    const d = parseLocalDate(dateStr);
    return d.getDate();
  };

  const formatDayName = (dateStr: string) => {
    const d = parseLocalDate(dateStr);
    return d.toLocaleDateString('en-US', { weekday: 'short' });
  };

  return (
    <div className="date-picker">
      {Array.from(datesByMonth.entries()).map(([monthKey, dates]) => {
        const firstDate = parseLocalDate(dates[0]);
        const monthLabel = firstDate.toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'long',
        });

        return (
          <div key={monthKey} className="month-group">
            <div className="month-label">{monthLabel}</div>
            <div className="dates-grid">
              {dates.map((date) => (
                <button
                  key={date}
                  className={`date-button ${date === selectedDate ? 'selected' : ''}`}
                  onClick={() => onSelectDate(date)}
                  title={parseLocalDate(date).toLocaleDateString()}
                >
                  <span className="day-name">{formatDayName(date)}</span>
                  <span className="day-number">{formatDay(date)}</span>
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
