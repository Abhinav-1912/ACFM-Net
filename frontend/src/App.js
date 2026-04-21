/**
 * App.js — ACFM-Net root component
 *
 * Renders the header, optional alert banner and the main Dashboard.
 */

import React, { useState, useCallback } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';

function App() {
  const [alert, setAlert] = useState(null);

  // Called by Dashboard when a cognitive-overload alert fires
  const handleAlert = useCallback((message) => {
    setAlert(message);
    // Auto-dismiss after 6 seconds
    setTimeout(() => setAlert(null), 6000);
  }, []);

  const dismissAlert = useCallback(() => setAlert(null), []);

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="app-header">
        <h1>ACFM-Net</h1>
        <span className="header-subtitle">Adaptive Cognitive &amp; Fatigue Monitor</span>
      </header>

      {/* ── Alert Banner ── */}
      {alert && (
        <div className="alert-banner" role="alert">
          <span className="alert-icon">⚠️</span>
          <span className="alert-text">{alert}</span>
          <button className="alert-dismiss" onClick={dismissAlert} aria-label="Dismiss alert">
            ✕
          </button>
        </div>
      )}

      {/* ── Main Content ── */}
      <main className="app-main">
        <Dashboard onAlert={handleAlert} />
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        ACFM-Net &copy; {new Date().getFullYear()} — Real-time Cognitive Monitoring
      </footer>
    </div>
  );
}

export default App;
