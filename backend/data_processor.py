"""
data_processor.py - ACFM-Net Data Preprocessing Pipeline

Loads the eye-tracking CSV dataset, explores it with visualizations,
normalizes features, creates LSTM sequences, and saves preprocessed arrays.

Usage:
    python data_processor.py data/final_dataset.csv
"""

import sys
import os
import json
import pickle
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
# Constants
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30          # number of time-steps fed to the LSTM
FEATURE_COLS = ["blink_rate", "EAR", "blink_count"]
LABEL_COL = "label"
TEST_SIZE = 0.20
RANDOM_STATE = 42
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs():
    """Create output directories if they do not exist."""
    for d in (LOGS_DIR, MODELS_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Read CSV, validate required columns and drop NaNs."""
    logger.info("Loading dataset from '%s'", csv_path)
    df = pd.read_csv(csv_path)
    required = FEATURE_COLS + [LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].dropna()
    logger.info("Dataset shape after cleaning: %s", df.shape)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Generate and save exploratory visualisation figures."""
    logger.info("Generating exploratory visualisations …")

    # --- label distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    df[LABEL_COL].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(LOGS_DIR, "label_distribution.png"), dpi=120)
    plt.close(fig)

    # --- feature correlation heat-map ---
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[FEATURE_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(LOGS_DIR, "feature_correlation.png"), dpi=120)
    plt.close(fig)

    # --- feature statistics box-plot ---
    fig, axes = plt.subplots(1, len(FEATURE_COLS), figsize=(12, 4))
    for ax, col in zip(axes, FEATURE_COLS):
        df.boxplot(column=col, by=LABEL_COL, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    plt.suptitle("Feature Distribution by Label")
    plt.tight_layout()
    fig.savefig(os.path.join(LOGS_DIR, "feature_by_label.png"), dpi=120)
    plt.close(fig)

    # --- descriptive statistics to console ---
    logger.info("Descriptive statistics:\n%s", df[FEATURE_COLS].describe().to_string())


def encode_labels(df: pd.DataFrame):
    """Map string/numeric labels to contiguous integers, return encoded series and mapping."""
    unique_labels = sorted(df[LABEL_COL].unique())
    label_mapping = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}
    reverse_mapping = {idx: str(lbl) for lbl, idx in label_mapping.items()}
    encoded = df[LABEL_COL].map(lambda x: label_mapping[str(x)])
    logger.info("Label mapping: %s", label_mapping)
    return encoded, label_mapping, reverse_mapping


def normalize_features(df: pd.DataFrame):
    """Fit a StandardScaler on feature columns and return (scaled_array, scaler)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS].values.astype(np.float32))
    return scaled, scaler


def create_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int):
    """
    Slide a window of *seq_len* over features/labels.

    Returns:
        X: (N, seq_len, num_features)
        y: (N,)  — label at the *last* step of each window
    """
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(labels[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: str) -> None:
    _ensure_dirs()

    # 1. Load
    df = load_dataset(csv_path)

    # 2. Explore
    explore_data(df)

    # 3. Encode labels
    encoded_labels, label_mapping, reverse_mapping = encode_labels(df)

    # 4. Normalise features
    features_scaled, scaler = normalize_features(df)

    # 5. Build sequences
    labels_array = encoded_labels.values
    X, y = create_sequences(features_scaled, labels_array, SEQUENCE_LENGTH)
    logger.info("Sequence shapes — X: %s  y: %s", X.shape, y.shape)

    # 6. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Split — train: %s  test: %s", X_train.shape[0], X_test.shape[0]
    )

    # 7. Save arrays
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    logger.info("Saved array files to '%s'", DATA_DIR)

    # 8. Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to '%s'", scaler_path)

    # 9. Save label mapping
    mapping_path = os.path.join(MODELS_DIR, "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({"label_to_idx": label_mapping, "idx_to_label": reverse_mapping}, f, indent=2)
    logger.info("Saved label mapping to '%s'", mapping_path)

    logger.info("Data processing complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACFM-Net data preprocessor")
    parser.add_argument("csv_path", help="Path to the input CSV dataset")
    args = parser.parse_args()
    main(args.csv_path)
