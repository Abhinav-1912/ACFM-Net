/**
 * StateIndicator.jsx — Displays the current cognitive state prediction
 *
 * States: normal | fatigue | stress | unknown
 * Each state has a distinct colour, icon and description.
 */

import React from 'react';

const STATE_CONFIG = {
  normal: {
    label:       'Normal',
    icon:        '🧠',
    color:       '#4ade80',
    bg:          'rgba(74,222,128,0.12)',
    border:      'rgba(74,222,128,0.3)',
    description: 'Cognitive load within healthy range',
  },
  fatigue: {
    label:       'Fatigue',
    icon:        '😴',
    color:       '#facc15',
    bg:          'rgba(250,204,21,0.12)',
    border:      'rgba(250,204,21,0.3)',
    description: 'Signs of mental fatigue detected',
  },
  stress: {
    label:       'Stress',
    icon:        '😰',
    color:       '#f87171',
    bg:          'rgba(248,113,113,0.12)',
    border:      'rgba(248,113,113,0.3)',
    description: 'Elevated cognitive stress detected',
  },
  unknown: {
    label:       'Calibrating…',
    icon:        '⏳',
    color:       '#94a3b8',
    bg:          'rgba(148,163,184,0.08)',
    border:      'rgba(148,163,184,0.2)',
    description: 'Collecting baseline data',
  },
};

export default function StateIndicator({ state = 'unknown' }) {
  const key    = (state || 'unknown').toLowerCase();
  const config = STATE_CONFIG[key] || STATE_CONFIG.unknown;

  return (
    <div
      style={{
        display:       'flex',
        flexDirection: 'column',
        alignItems:     'center',
        justifyContent: 'center',
        gap:            14,
        flex:          1,
        padding:       '16px 8px',
      }}
    >
      {/* Icon badge */}
      <div
        style={{
          width:        72,
          height:       72,
          borderRadius: '50%',
          background:   config.bg,
          border:       `2px solid ${config.border}`,
          display:      'flex',
          alignItems:   'center',
          justifyContent: 'center',
          fontSize:       '2rem',
          transition:   'all 0.4s ease',
          boxShadow:    `0 0 20px ${config.border}`,
        }}
      >
        {config.icon}
      </div>

      {/* Label */}
      <div
        style={{
          fontSize:   '1.1rem',
          fontWeight: 700,
          color:      config.color,
          letterSpacing: '0.04em',
          transition: 'color 0.4s ease',
        }}
      >
        {config.label}
      </div>

      {/* Description */}
      <div
        style={{
          fontSize:  '0.75rem',
          color:     '#64748b',
          textAlign: 'center',
          lineHeight: 1.4,
        }}
      >
        {config.description}
      </div>
    </div>
  );
}
