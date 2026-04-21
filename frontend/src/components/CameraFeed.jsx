/**
 * CameraFeed.jsx — Camera access + WebSocket data bridge
 *
 * • Requests camera access via getUserMedia
 * • Opens a WebSocket connection to the FastAPI backend
 * • Every 200 ms sends simulated eye-tracking metrics derived from the
 *   live video frame (blink_rate, EAR, blink_count)
 * • Parses prediction JSON messages and calls onPrediction()
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';

const WS_BASE = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
const SEND_INTERVAL_MS = 200;

// Seed pseudo-random metrics that drift over time for demo purposes
function simulateMetrics(frameRef) {
  const t = Date.now() / 1000;
  const blinkRate  = Math.max(0.05, Math.min(0.5, 0.2 + 0.15 * Math.sin(t / 12)));
  const ear        = Math.max(0.15, Math.min(0.45, 0.3 + 0.1 * Math.sin(t / 7)));
  const blinkCount = Math.floor(Math.abs(Math.sin(t / 20)) * 40);
  frameRef.current = { blink_rate: blinkRate, EAR: ear, blink_count: blinkCount };
  return frameRef.current;
}

const STATUS_COLORS = {
  connecting: '#facc15',
  connected:  '#4ade80',
  error:      '#f87171',
  closed:     '#94a3b8',
};

export default function CameraFeed({ onPrediction }) {
  const videoRef  = useRef(null);
  const wsRef     = useRef(null);
  const timerRef  = useRef(null);
  const frameRef  = useRef({});
  const userIdRef = useRef(`user_${Math.random().toString(36).slice(2, 8)}`);

  const [wsStatus, setWsStatus]         = useState('closed');
  const [cameraError, setCameraError]   = useState(null);
  const [frameCount, setFrameCount]     = useState(0);

  // ── WebSocket ──────────────────────────────────────────────────────────
  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

    const url = `${WS_BASE}/ws/${userIdRef.current}`;
    setWsStatus('connecting');
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setWsStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.state && typeof data.csi === 'number') {
          onPrediction(data);
        }
      } catch {
        /* ignore malformed frames */
      }
    };

    ws.onerror = () => setWsStatus('error');
    ws.onclose = () => {
      setWsStatus('closed');
      // Auto-reconnect after 3 s
      setTimeout(connectWs, 3000);
    };

    wsRef.current = ws;
  }, [onPrediction]);

  // ── Camera ────────────────────────────────────────────────────────────
  useEffect(() => {
    let stream;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setCameraError(`Camera access denied: ${err.message}`);
      }
    }

    startCamera();
    connectWs();

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (wsRef.current) wsRef.current.close();
      clearInterval(timerRef.current);
    };
  }, [connectWs]);

  // ── Metric sender ─────────────────────────────────────────────────────
  useEffect(() => {
    timerRef.current = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const metrics = simulateMetrics(frameRef);
        wsRef.current.send(JSON.stringify(metrics));
        setFrameCount((c) => c + 1);
      }
    }, SEND_INTERVAL_MS);

    return () => clearInterval(timerRef.current);
  }, []);

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Status bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.78rem' }}>
        <span
          style={{
            width: 10, height: 10, borderRadius: '50%',
            background: STATUS_COLORS[wsStatus],
            display: 'inline-block',
            boxShadow: `0 0 6px ${STATUS_COLORS[wsStatus]}`,
          }}
        />
        <span style={{ color: '#94a3b8', textTransform: 'capitalize' }}>{wsStatus}</span>
        <span style={{ marginLeft: 'auto', color: '#64748b' }}>Frames sent: {frameCount}</span>
      </div>

      {/* Video feed */}
      {cameraError ? (
        <div
          style={{
            background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: 8,
            padding: 16,
            color: '#fca5a5',
            fontSize: '0.85rem',
          }}
        >
          {cameraError}
          <br />
          <small style={{ color: '#94a3b8' }}>
            Metrics are still being simulated and sent to the backend.
          </small>
        </div>
      ) : (
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{
            width: '100%',
            borderRadius: 10,
            border: '1px solid rgba(148,163,184,0.15)',
            background: '#0f172a',
            aspectRatio: '4/3',
            objectFit: 'cover',
          }}
        />
      )}

      {/* User ID */}
      <div style={{ fontSize: '0.72rem', color: '#475569' }}>
        Session ID: <code style={{ color: '#94a3b8' }}>{userIdRef.current}</code>
      </div>
    </div>
  );
}
