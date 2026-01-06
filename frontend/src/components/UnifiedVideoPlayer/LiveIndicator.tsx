/**
 * Live indicator component showing LIVE badge or time behind live with return button.
 */

import type { LiveIndicatorProps } from './types';
import './LiveIndicator.css';

export function LiveIndicator({
  isAtLiveEdge,
  timeBehindLive,
  onReturnToLive,
}: LiveIndicatorProps) {
  const formatTimeBehind = (seconds: number): string => {
    if (seconds < 1) return '';

    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);

    if (mins > 0) {
      return `-${mins}:${secs.toString().padStart(2, '0')}`;
    }
    return `-${secs}s`;
  };

  if (isAtLiveEdge) {
    return (
      <div className="live-indicator at-edge">
        <span className="live-dot" />
        <span className="live-text">LIVE</span>
      </div>
    );
  }

  return (
    <button className="live-indicator behind" onClick={onReturnToLive} title="Return to live (L)">
      <span className="time-behind">{formatTimeBehind(timeBehindLive)}</span>
      <span className="return-text">Go Live</span>
    </button>
  );
}
