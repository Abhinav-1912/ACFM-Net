/**
 * Dashboard.jsx — Main monitoring dashboard
 *
 * Orchestrates all child components:
 *   • CameraFeed   — camera access + WebSocket data stream
 *   • CSIGauge     — SVG gauge showing Cognitive State Index
 *   • StateIndicator — current predicted cognitive state
 *   • TimelineChart  — last 60 CSI readings over time
 *
 * State is lifted here so every component stays in sync.
 */

import React, { useState, useCallback, useRef } from 'react';
import CameraFeed from './CameraFeed';
import CSIGauge from './CSIGauge';
import StateIndicator from './StateIndicator';
import TimelineChart from './TimelineChart';

// Stats card helper
function StatItem({ label, value, unit = '' }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{ fontSize: '0.72rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {label}
      </span>
      <span style={{ fontSize: '1.3rem', fontWeight: 700, color: '#e2e8f0' }}>
        {value !== null && value !== undefined ? `${value}${unit}` : '—'}
      </span>
    </div>
  );
}

const MAX_TIMELINE = 60;

export default function Dashboard({ onAlert }) {
  const [csi, setCsi] = useState(100);
  const [state, setState] = useState('unknown');
  const [timeline, setTimeline] = useState([]);
  const [stats, setStats] = useState({
    totalPredictions: 0,
    alertCount: 0,
    avgCsi: null,
    minCsi: null,
    maxCsi: null,
  });

  const alertFiredRef = useRef(false);

  // Invoked by CameraFeed whenever a prediction arrives from the WebSocket
  const handlePrediction = useCallback(
    (prediction) => {
      const { csi: newCsi, state: newState, alert } = prediction;

      setCsi(newCsi);
      setState(newState);

      setTimeline((prev) => {
        const next = [...prev, newCsi];
        return next.length > MAX_TIMELINE ? next.slice(next.length - MAX_TIMELINE) : next;
      });

      setStats((prev) => {
        const count = prev.totalPredictions + 1;
        const alerts = prev.alertCount + (alert ? 1 : 0);
        const allCsi = [...(prev._allCsi || []), newCsi];
        return {
          totalPredictions: count,
          alertCount: alerts,
          avgCsi: Math.round(allCsi.reduce((a, b) => a + b, 0) / allCsi.length),
          minCsi: Math.round(Math.min(...allCsi)),
          maxCsi: Math.round(Math.max(...allCsi)),
          _allCsi: allCsi,
        };
      });

      // Trigger alert banner for cognitive overload (CSI < 40)
      if (alert && !alertFiredRef.current) {
        alertFiredRef.current = true;
        onAlert?.('⚠️ Cognitive overload detected! Please take a break.');
        setTimeout(() => { alertFiredRef.current = false; }, 8000);
      }
    },
    [onAlert],
  );

  return (
    <div className="dashboard-grid">
      {/* Camera Feed */}
      <div className="card grid-camera">
        <div className="card-title">Camera Feed</div>
        <CameraFeed onPrediction={handlePrediction} />
      </div>

      {/* CSI Gauge */}
      <div className="card grid-gauge">
        <div className="card-title">Cognitive State Index</div>
        <CSIGauge value={csi} />
      </div>

      {/* State Indicator */}
      <div className="card grid-state">
        <div className="card-title">State</div>
        <StateIndicator state={state} />
      </div>

      {/* Session Stats */}
      <div className="card grid-stats">
        <div className="card-title">Session Stats</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 24px' }}>
          <StatItem label="Predictions" value={stats.totalPredictions} />
          <StatItem label="Alerts" value={stats.alertCount} />
          <StatItem label="Avg CSI" value={stats.avgCsi} />
          <StatItem label="Min CSI" value={stats.minCsi} />
          <StatItem label="Max CSI" value={stats.maxCsi} />
        </div>
      </div>

      {/* Timeline Chart */}
      <div className="card grid-timeline">
        <div className="card-title">CSI Timeline (last 60 readings)</div>
        <TimelineChart data={timeline} />
      </div>
    </div>
  );
}
