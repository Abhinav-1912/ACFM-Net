/**
 * TimelineChart.jsx — Real-time CSI line chart (last 60 readings)
 *
 * Uses Chart.js 4 via react-chartjs-2.
 * Dark theme with light grid lines.
 */

import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
);

const GRID_COLOR  = 'rgba(148,163,184,0.08)';
const TICK_COLOR  = '#64748b';
const LINE_COLOR  = '#38bdf8';
const FILL_COLOR  = 'rgba(56,189,248,0.08)';

export default function TimelineChart({ data = [] }) {
  const labels = useMemo(
    () => data.map((_, i) => `${i + 1}`),
    [data],
  );

  const chartData = useMemo(
    () => ({
      labels,
      datasets: [
        {
          label:           'CSI',
          data,
          borderColor:     LINE_COLOR,
          backgroundColor: FILL_COLOR,
          borderWidth:     2,
          pointRadius:     data.length > 30 ? 0 : 3,
          pointHoverRadius: 5,
          fill:            true,
          tension:         0.4,
        },
      ],
    }),
    [labels, data],
  );

  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: 'rgba(15,23,42,0.9)',
          titleColor:      '#94a3b8',
          bodyColor:       '#e2e8f0',
          borderColor:     'rgba(148,163,184,0.2)',
          borderWidth:     1,
          callbacks: {
            label: (ctx) => ` CSI: ${ctx.parsed.y}`,
          },
        },
      },
      scales: {
        x: {
          display: true,
          grid: { color: GRID_COLOR },
          ticks: {
            color:     TICK_COLOR,
            maxTicksLimit: 10,
            font: { size: 10 },
          },
        },
        y: {
          min: 0,
          max: 100,
          display: true,
          grid: { color: GRID_COLOR },
          ticks: {
            color: TICK_COLOR,
            stepSize: 20,
            font: { size: 10 },
          },
        },
      },
    }),
    [],
  );

  if (data.length === 0) {
    return (
      <div
        style={{
          height:         160,
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'center',
          color:          '#475569',
          fontSize:       '0.85rem',
        }}
      >
        Waiting for data…
      </div>
    );
  }

  return (
    <div style={{ height: 200, position: 'relative' }}>
      <Line data={chartData} options={options} />
    </div>
  );
}
