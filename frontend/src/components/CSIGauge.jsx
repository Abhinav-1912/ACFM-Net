/**
 * CSIGauge.jsx — SVG arc gauge for Cognitive State Index (0–100)
 *
 * Color zones:
 *   ≥ 70  → green  (healthy)
 *   40–69 → yellow (caution)
 *   < 40  → red    (alert)
 */

import React, { useEffect, useRef } from 'react';

const SIZE = 200;
const STROKE = 18;
const RADIUS = (SIZE - STROKE) / 2 - 4;
const CX = SIZE / 2;
const CY = SIZE / 2 + 20; // shifted down so arc looks like a speedometer

// Arc spans from -200° to 20° (220° total sweep, starting at bottom-left)
const START_ANGLE = -200;
const END_ANGLE = 20;
const SWEEP = END_ANGLE - START_ANGLE;

function polarToXY(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return {
    x: cx + r * Math.cos(rad),
    y: cy + r * Math.sin(rad),
  };
}

function arcPath(cx, cy, r, startAngle, endAngle) {
  const start = polarToXY(cx, cy, r, startAngle);
  const end   = polarToXY(cx, cy, r, endAngle);
  const large = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 1 ${end.x} ${end.y}`;
}

function getColor(value) {
  if (value >= 70) return '#4ade80'; // green
  if (value >= 40) return '#facc15'; // yellow
  return '#f87171';                  // red
}

export default function CSIGauge({ value = 100 }) {
  const clampedValue = Math.max(0, Math.min(100, value));
  const color = getColor(clampedValue);

  // Animated value
  const displayRef = useRef(clampedValue);
  const [displayValue, setDisplayValue] = React.useState(clampedValue);

  useEffect(() => {
    const target = clampedValue;
    const current = displayRef.current;
    const diff = target - current;
    const steps = 20;
    const stepSize = diff / steps;
    let step = 0;

    const interval = setInterval(() => {
      step++;
      displayRef.current = current + stepSize * step;
      setDisplayValue(Math.round(displayRef.current));
      if (step >= steps) {
        displayRef.current = target;
        setDisplayValue(target);
        clearInterval(interval);
      }
    }, 16);

    return () => clearInterval(interval);
  }, [clampedValue]);

  const fillAngle = START_ANGLE + (SWEEP * displayValue) / 100;

  const trackPath  = arcPath(CX, CY, RADIUS, START_ANGLE, END_ANGLE);
  const fillPath   = arcPath(CX, CY, RADIUS, START_ANGLE, fillAngle);

  // Needle tip
  const needleTip = polarToXY(CX, CY, RADIUS - 6, fillAngle);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        width={SIZE}
        height={SIZE}
        style={{ overflow: 'visible', maxWidth: '100%' }}
      >
        {/* Track */}
        <path
          d={trackPath}
          fill="none"
          stroke="rgba(148,163,184,0.15)"
          strokeWidth={STROKE}
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d={fillPath}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeLinecap="round"
          style={{ transition: 'stroke 0.4s ease' }}
        />
        {/* Needle dot */}
        <circle cx={needleTip.x} cy={needleTip.y} r={6} fill={color} />

        {/* Centre value */}
        <text
          x={CX}
          y={CY + 10}
          textAnchor="middle"
          fontSize="38"
          fontWeight="700"
          fill="#e2e8f0"
          fontFamily="'Segoe UI', system-ui, sans-serif"
        >
          {displayValue}
        </text>
        <text x={CX} y={CY + 32} textAnchor="middle" fontSize="12" fill="#94a3b8">
          CSI
        </text>

        {/* Min / Max labels */}
        {(() => {
          const pMin = polarToXY(CX, CY, RADIUS + 16, START_ANGLE);
          const pMax = polarToXY(CX, CY, RADIUS + 16, END_ANGLE);
          return (
            <>
              <text x={pMin.x} y={pMin.y} textAnchor="middle" fontSize="11" fill="#64748b">0</text>
              <text x={pMax.x} y={pMax.y} textAnchor="middle" fontSize="11" fill="#64748b">100</text>
            </>
          );
        })()}
      </svg>

      {/* Zone label */}
      <div
        style={{
          fontSize: '0.85rem',
          fontWeight: 600,
          color,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}
      >
        {clampedValue >= 70 ? 'Healthy' : clampedValue >= 40 ? 'Caution' : 'Alert'}
      </div>
    </div>
  );
}
