"""
app.py - ACFM-Net FastAPI Backend

Endpoints:
  GET  /api/health                  - server health check
  POST /api/calibrate/{user_id}     - start / reset a user session
  GET  /api/session/{user_id}/stats - retrieve session statistics
  WS   /ws/{user_id}                - real-time eye-metric stream & predictions

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import pickle
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH   = os.path.join(MODELS_DIR, "lstm_model.pth")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
MAPPING_PATH = os.path.join(MODELS_DIR, "label_mapping.json")

# ---------------------------------------------------------------------------
# LSTM + Attention (must match train_lstm.py)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class ACFMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_heads, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = MultiHeadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attended = self.attention(lstm_out)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Global model state (loaded at startup)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[ACFMNet] = None
scaler = None
label_mapping: Dict[str, Any] = {}


def _load_model() -> None:
    """Load LSTM model, scaler and label mapping from disk."""
    global model, scaler, label_mapping

    if not os.path.exists(MODEL_PATH):
        logger.warning("Model file not found at '%s'. Predictions will be random.", MODEL_PATH)
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = ACFMNet(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        num_heads=checkpoint["num_heads"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("LSTM model loaded from '%s'", MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded from '%s'", SCALER_PATH)

    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH) as f:
            label_mapping = json.load(f)
        logger.info("Label mapping loaded: %s", label_mapping.get("idx_to_label"))


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30
BUFFER_SIZE = 30
CSI_ALERT_LOW = 40    # red zone
CSI_ALERT_MED = 70    # yellow zone

sessions: Dict[str, Dict] = {}


def _new_session(user_id: str) -> Dict:
    return {
        "user_id": user_id,
        "buffer": deque(maxlen=BUFFER_SIZE),
        "predictions": [],
        "csi_history": [],
        "alert_count": 0,
        "start_time": time.time(),
        "last_state": "unknown",
    }


def _compute_csi(probabilities: np.ndarray, idx_to_label: Dict) -> float:
    """
    Derive a 0-100 Cognitive State Index from class probabilities.
    Label 0 → normal (high CSI), others reduce it.
    """
    if len(probabilities) == 0:
        return 100.0
    # normal class probability drives CSI up; other classes drag it down
    normal_idx = next(
        (int(k) for k, v in idx_to_label.items() if str(v).lower() == "normal"),
        0,
    )
    p_normal = float(probabilities[normal_idx]) if normal_idx < len(probabilities) else 0.5
    return round(p_normal * 100, 2)


def _predict(session: Dict) -> Optional[Dict]:
    """Run inference on the current frame buffer; return prediction dict or None."""
    buf = list(session["buffer"])
    if len(buf) < SEQUENCE_LENGTH:
        return None

    features = np.array(buf[-SEQUENCE_LENGTH:], dtype=np.float32)

    if scaler is not None:
        features = scaler.transform(features)

    tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, seq, 3)

    idx_to_label = label_mapping.get("idx_to_label", {"0": "normal", "1": "fatigue", "2": "stress"})

    if model is not None:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_label.get(str(pred_idx), "unknown")
    else:
        # Fallback: random prediction when model not yet trained
        probs = np.random.dirichlet(np.ones(3))
        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_label.get(str(pred_idx), "unknown")

    csi = _compute_csi(probs, idx_to_label)
    alert = csi < CSI_ALERT_LOW

    if alert:
        session["alert_count"] += 1

    result = {
        "state": pred_label,
        "csi": csi,
        "probabilities": {idx_to_label.get(str(i), str(i)): round(float(p), 4) for i, p in enumerate(probs)},
        "alert": alert,
        "timestamp": time.time(),
    }
    session["predictions"].append(result)
    session["csi_history"].append(csi)
    session["last_state"] = pred_label
    return result


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="ACFM-Net Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    _load_model()


# ---------------------------
# REST endpoints
# ---------------------------

@app.get("/api/health")
async def health():
    """Server health check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "active_sessions": len(sessions),
    }


@app.post("/api/calibrate/{user_id}")
async def calibrate(user_id: str):
    """Start or reset a monitoring session for the given user."""
    sessions[user_id] = _new_session(user_id)
    logger.info("Session started for user '%s'", user_id)
    return {"status": "session_started", "user_id": user_id}


@app.get("/api/session/{user_id}/stats")
async def session_stats(user_id: str):
    """Return aggregated statistics for an active session."""
    if user_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sess = sessions[user_id]
    history = sess["csi_history"]
    return {
        "user_id": user_id,
        "duration_seconds": round(time.time() - sess["start_time"], 1),
        "total_predictions": len(sess["predictions"]),
        "alert_count": sess["alert_count"],
        "last_state": sess["last_state"],
        "avg_csi": round(float(np.mean(history)), 2) if history else None,
        "min_csi": round(float(np.min(history)), 2) if history else None,
        "max_csi": round(float(np.max(history)), 2) if history else None,
    }


# ---------------------------
# WebSocket endpoint
# ---------------------------

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    Real-time eye-metric stream.

    Expected incoming JSON:
      { "blink_rate": float, "EAR": float, "blink_count": float }

    Outgoing JSON (prediction result or status message).
    """
    await websocket.accept()
    logger.info("WebSocket connected: user '%s'", user_id)

    if user_id not in sessions:
        sessions[user_id] = _new_session(user_id)

    sess = sessions[user_id]

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            # Validate required fields
            required = ["blink_rate", "EAR", "blink_count"]
            if not all(k in data for k in required):
                await websocket.send_json({"error": f"missing fields: {required}"})
                continue

            # Add frame to buffer
            frame = [float(data["blink_rate"]), float(data["EAR"]), float(data["blink_count"])]
            sess["buffer"].append(frame)

            # Run prediction when buffer is full
            result = _predict(sess)
            if result is not None:
                await websocket.send_json(result)
            else:
                frames_needed = SEQUENCE_LENGTH - len(sess["buffer"])
                await websocket.send_json({"status": "buffering", "frames_needed": frames_needed})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: user '%s'", user_id)
    except Exception as exc:
        logger.exception("WebSocket error for user '%s': %s", user_id, exc)
        await websocket.close(code=1011)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
