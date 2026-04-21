"""
train_lstm.py - ACFM-Net LSTM + Multi-Head Attention Trainer

Architecture:
  • 2-layer LSTM (input_size=3, hidden_size=128)
  • Multi-head attention (4 heads) applied over the sequence output
  • Fully-connected classification head (3 classes)

Training features:
  • CrossEntropyLoss
  • Adam optimiser
  • ReduceLROnPlateau scheduler (without verbose parameter for newer PyTorch)
  • Early stopping with patience=15
  • Training history plots saved to logs/

Usage:
    python train_lstm.py
"""

import os
import json
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
INPUT_SIZE = 3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 3
NUM_HEADS = 4
DROPOUT = 0.3

BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15        # early-stopping patience
SCHEDULER_PATIENCE = 7


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention over a sequence."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)          # residual + layer-norm


class ACFMNet(nn.Module):
    """
    LSTM encoder + Multi-Head Attention + MLP classifier.

    Forward pass returns logits of shape (batch, num_classes).
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attention = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)                  # (batch, seq_len, hidden)
        attended = self.attention(lstm_out)           # (batch, seq_len, hidden)
        pooled = attended.mean(dim=1)                 # mean pool over time
        return self.classifier(pooled)               # (batch, num_classes)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def load_data():
    """Load preprocessed numpy arrays and return DataLoaders."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    logger.info("Train shape: %s  Test shape: %s", X_train.shape, X_test.shape)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    test_ds  = TensorDataset(
        torch.from_numpy(X_test),  torch.from_numpy(y_test).long()
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total


def save_plots(history: dict) -> None:
    """Save loss and accuracy curves."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- loss ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"],   label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(LOGS_DIR, "loss_history.png"), dpi=120)
    plt.close(fig)

    # --- accuracy ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_acc"], label="Train Acc")
    ax.plot(epochs, history["val_acc"],   label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(LOGS_DIR, "accuracy_history.png"), dpi=120)
    plt.close(fig)

    logger.info("Saved training plots to '%s'", LOGS_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, test_loader = load_data()

    model = ACFMNet().to(device)
    logger.info("Model parameters: %s", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ReduceLROnPlateau — do NOT pass verbose= (removed in newer PyTorch)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = math.inf
    patience_counter = 0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
            "train_acc=%.4f  val_acc=%.4f  lr=%.2e",
            epoch, NUM_EPOCHS, train_loss, val_loss, train_acc, val_acc, current_lr,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    # Restore best weights and save
    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = os.path.join(MODELS_DIR, "lstm_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_classes": NUM_CLASSES,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
        },
        model_path,
    )
    logger.info("Saved model to '%s'  (best val_loss=%.4f)", model_path, best_val_loss)

    save_plots(history)
    logger.info("Training complete ✓")


if __name__ == "__main__":
    main()
