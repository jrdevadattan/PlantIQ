"""
PlantIQ — LSTM Autoencoder for Power Curve Anomaly Detection
================================================================
Learns the shape of normal 30-minute power consumption curves.
When a live batch's curve deviates significantly from "normal", it
produces a high reconstruction error → anomaly alert.

Architecture (per README Component 2):
  Input: power curve (1800 timesteps × 1 feature)
      ↓
  LSTM Encoder (hidden_size=64, n_layers=2)
      ↓
  Bottleneck: 64-dimensional "fingerprint" of the curve
      ↓
  LSTM Decoder (hidden_size=64 → linear → 1, n_layers=2)
      ↓
  Reconstructed curve (1800 timesteps × 1 feature)
      ↓
  Reconstruction Error = MSE(original, reconstructed)
      ↓
  Error > threshold (99th percentile of training errors) → ANOMALY

Anomaly score interpretation:
  0.00–0.15  →  Normal     (green)
  0.15–0.30  →  Watch      (amber)
  0.30–0.60  →  Warning    (orange)
  0.60+      →  Critical   (red)

Usage:
  # Train
  python models/lstm_autoencoder.py --train

  # Quick inference demo
  python models/lstm_autoencoder.py --demo
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Paths ────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
CURVES_DIR = os.path.join(DATA_DIR, "power_curves")
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")
BATCH_CSV = os.path.join(DATA_DIR, "batch_data.csv")

# ── Hyperparameters ──────────────────────────────────────────
TIMESTEPS = 1800           # 30 min at 1Hz
HIDDEN_SIZE = 64           # LSTM hidden dimension (per README)
NUM_LAYERS = 2             # LSTM depth (per README)
BOTTLENECK_DIM = 64        # Compressed representation size
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50
SUBSAMPLE_FACTOR = 6       # Downsample 1800 → 300 for training speed
SEQ_LEN = TIMESTEPS // SUBSAMPLE_FACTOR  # 300

# Anomaly threshold percentile (per README: 99th percentile of training errors)
THRESHOLD_PERCENTILE = 99

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class PowerCurveDataset(Dataset):
    """Dataset of power curve .npy files.

    Parameters
    ----------
    file_paths : list[str]
        Paths to .npy curve files.
    subsample : int
        Downsample factor (every Nth sample).
    normalize : bool
        If True, normalize each curve to [0, 1] range.
    """

    def __init__(
        self,
        file_paths: list[str],
        subsample: int = SUBSAMPLE_FACTOR,
        normalize: bool = True,
    ):
        self.file_paths = file_paths
        self.subsample = subsample
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        curve = np.load(self.file_paths[idx]).astype(np.float32)

        # Subsample for training efficiency
        if self.subsample > 1:
            curve = curve[::self.subsample]

        # Normalize to [0, 1]
        if self.normalize:
            cmin, cmax = curve.min(), curve.max()
            if cmax - cmin > 1e-6:
                curve = (curve - cmin) / (cmax - cmin)
            else:
                curve = np.zeros_like(curve)

        # Shape: (seq_len, 1) — LSTM expects (seq_len, input_size)
        return torch.tensor(curve, dtype=torch.float32).unsqueeze(-1)


# ══════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════

class LSTMEncoder(nn.Module):
    """LSTM Encoder — compresses a power curve into a fixed-length vector."""

    def __init__(self, input_size: int = 1, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, 1)

        Returns
        -------
        h_n : (num_layers, batch, hidden_size)  — final hidden states
        c_n : (num_layers, batch, hidden_size)  — final cell states
        """
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class LSTMDecoder(nn.Module):
    """LSTM Decoder — reconstructs a power curve from the bottleneck."""

    def __init__(
        self,
        output_size: int = 1,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, h_n: torch.Tensor, c_n: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Parameters
        ----------
        h_n, c_n : hidden/cell states from encoder
        batch_size : int

        Returns
        -------
        reconstructed : (batch, seq_len, 1)
        """
        # Decoder input: zeros (autoregressive from hidden state)
        decoder_input = torch.zeros(batch_size, self.seq_len, 1, device=h_n.device)
        lstm_out, _ = self.lstm(decoder_input, (h_n, c_n))
        # Project LSTM output to 1D at each timestep
        reconstructed = self.fc(lstm_out)
        return reconstructed


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for power curve anomaly detection.

    Combines encoder + decoder.  Trains to minimize reconstruction
    error on normal curves.  Anomalous curves produce high error.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.decoder = LSTMDecoder(input_size, hidden_size, num_layers, seq_len)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, 1)

        Returns
        -------
        reconstructed : (batch, seq_len, 1)
        """
        h_n, c_n = self.encoder(x)
        reconstructed = self.decoder(h_n, c_n, x.size(0))
        return reconstructed

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE reconstruction error.

        Parameters
        ----------
        x : (batch, seq_len, 1)

        Returns
        -------
        errors : (batch,) — MSE per sample
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            # MSE per sample: mean over (seq_len, 1)
            errors = torch.mean((x - x_hat) ** 2, dim=(1, 2))
        return errors


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def load_normal_curves() -> list[str]:
    """Load file paths for normal (non-fault) power curves."""
    import pandas as pd
    df = pd.read_csv(BATCH_CSV)
    normal_df = df[df["fault_type"] == "normal"]

    paths = []
    for batch_id in normal_df["batch_id"]:
        fpath = os.path.join(CURVES_DIR, f"{batch_id}.npy")
        if os.path.exists(fpath):
            paths.append(fpath)

    return paths


def load_all_curves() -> dict[str, list[str]]:
    """Load file paths grouped by fault type."""
    import pandas as pd
    df = pd.read_csv(BATCH_CSV)

    groups: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        ft = row["fault_type"]
        fpath = os.path.join(CURVES_DIR, f"{row['batch_id']}.npy")
        if os.path.exists(fpath):
            groups.setdefault(ft, []).append(fpath)

    return groups


def train_autoencoder(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    verbose: bool = True,
) -> tuple[LSTMAutoencoder, float, dict]:
    """Train the LSTM Autoencoder on normal power curves.

    Returns
    -------
    model : LSTMAutoencoder
        Trained model.
    threshold : float
        99th percentile reconstruction error (anomaly threshold).
    metadata : dict
        Training metadata (losses, threshold, config).
    """
    # Load normal curves
    normal_paths = load_normal_curves()
    if verbose:
        print(f"[PlantIQ] Found {len(normal_paths)} normal power curves for training")

    # Split 80/20 for train/val
    np.random.seed(42)
    indices = np.random.permutation(len(normal_paths))
    split = int(0.8 * len(indices))
    train_paths = [normal_paths[i] for i in indices[:split]]
    val_paths = [normal_paths[i] for i in indices[split:]]

    if verbose:
        print(f"  Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_dataset = PowerCurveDataset(train_paths)
    val_dataset = PowerCurveDataset(val_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = LSTMAutoencoder(seq_len=SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model params: {total_params:,} total, {trainable:,} trainable")
        print(f"  Device: {DEVICE}, Epochs: {epochs}, Batch size: {batch_size}")
        print()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            x_hat = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # ── Validate ─────────────────────────
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(DEVICE)
                x_hat = model(batch_x)
                loss = criterion(x_hat, batch_x)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"\n  Best validation loss: {best_val_loss:.6f}")

    # ── Compute anomaly threshold ────────────────────────────
    # Run all normal training curves through the model and take 99th percentile
    if verbose:
        print(f"  Computing anomaly threshold (p{THRESHOLD_PERCENTILE})...")

    all_normal_dataset = PowerCurveDataset(normal_paths)
    all_normal_loader = DataLoader(all_normal_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_errors = []
    with torch.no_grad():
        for batch_x in all_normal_loader:
            batch_x = batch_x.to(DEVICE)
            errors = model.get_reconstruction_error(batch_x)
            all_errors.extend(errors.cpu().numpy().tolist())

    all_errors = np.array(all_errors)
    threshold = float(np.percentile(all_errors, THRESHOLD_PERCENTILE))

    if verbose:
        print(f"  Normal curve errors: mean={all_errors.mean():.6f}, "
              f"std={all_errors.std():.6f}, max={all_errors.max():.6f}")
        print(f"  Anomaly threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}")

    # ── Test on fault curves ─────────────────────────────────
    if verbose:
        groups = load_all_curves()
        print(f"\n  Reconstruction errors by fault type:")
        for ft in ["normal", "bearing_wear", "wet_material", "calibration_needed"]:
            fpaths = groups.get(ft, [])
            if not fpaths:
                continue
            ds = PowerCurveDataset(fpaths)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            errs = []
            with torch.no_grad():
                for bx in dl:
                    bx = bx.to(DEVICE)
                    e = model.get_reconstruction_error(bx)
                    errs.extend(e.cpu().numpy().tolist())
            errs = np.array(errs)
            n_detected = (errs > threshold).sum()
            det_rate = n_detected / len(errs) * 100 if len(errs) > 0 else 0
            print(f"    {ft:25s}: mean={errs.mean():.6f}, "
                  f"detected={n_detected}/{len(errs)} ({det_rate:.1f}%)")

    # ── Build metadata ───────────────────────────────────────
    metadata = {
        "model_type": "LSTMAutoencoder",
        "timesteps_original": TIMESTEPS,
        "subsample_factor": SUBSAMPLE_FACTOR,
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "bottleneck_dim": BOTTLENECK_DIM,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "anomaly_threshold": threshold,
        "normal_error_mean": float(all_errors.mean()),
        "normal_error_std": float(all_errors.std()),
        "normal_curves_count": len(normal_paths),
        "best_val_loss": float(best_val_loss),
        "train_losses": [round(l, 6) for l in train_losses],
        "val_losses": [round(l, 6) for l in val_losses],
    }

    return model, threshold, metadata


def save_model(model: LSTMAutoencoder, threshold: float, metadata: dict):
    """Save model weights + metadata."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACT_DIR, "lstm_autoencoder.pt")
    meta_path = os.path.join(ARTIFACT_DIR, "lstm_autoencoder_meta.json")

    torch.save(model.state_dict(), model_path)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model saved: {model_path}")
    print(f"  Metadata saved: {meta_path}")
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model size: {size_mb:.2f} MB")


def load_model() -> tuple[LSTMAutoencoder, float, dict]:
    """Load trained model + metadata from disk."""
    model_path = os.path.join(ARTIFACT_DIR, "lstm_autoencoder.pt")
    meta_path = os.path.join(ARTIFACT_DIR, "lstm_autoencoder_meta.json")

    with open(meta_path) as f:
        metadata = json.load(f)

    model = LSTMAutoencoder(
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
        seq_len=metadata["seq_len"],
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    threshold = metadata["anomaly_threshold"]
    print(f"  LSTM Autoencoder loaded from {model_path}")
    print(f"  Anomaly threshold: {threshold:.6f}")

    return model, threshold, metadata


# ══════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════

def compute_anomaly_score(
    model: LSTMAutoencoder,
    curve: np.ndarray,
    threshold: float,
    normal_mean: float,
    normal_std: float,
) -> dict:
    """Score a single power curve for anomaly.

    Parameters
    ----------
    model : LSTMAutoencoder
        Trained model.
    curve : np.ndarray
        Raw power curve, shape (1800,).
    threshold : float
        Anomaly threshold from training.
    normal_mean, normal_std : float
        Mean/std of normal reconstruction errors for scaling.

    Returns
    -------
    dict with: anomaly_score (0–1), is_anomaly, reconstruction_error, severity
    """
    # Preprocess: subsample + normalize
    if len(curve) > SEQ_LEN * SUBSAMPLE_FACTOR:
        curve_sub = curve[::SUBSAMPLE_FACTOR]
    else:
        curve_sub = curve

    # Ensure correct length
    if len(curve_sub) > SEQ_LEN:
        curve_sub = curve_sub[:SEQ_LEN]
    elif len(curve_sub) < SEQ_LEN:
        curve_sub = np.pad(curve_sub, (0, SEQ_LEN - len(curve_sub)), mode="edge")

    # Normalize to [0, 1]
    cmin, cmax = curve_sub.min(), curve_sub.max()
    if cmax - cmin > 1e-6:
        curve_norm = (curve_sub - cmin) / (cmax - cmin)
    else:
        curve_norm = np.zeros_like(curve_sub)

    # To tensor
    x = torch.tensor(curve_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    # Get reconstruction error
    model.eval()
    error = model.get_reconstruction_error(x).item()

    # Scale to anomaly score (0–1)
    # Map: threshold → ~0.30 (the boundary between Watch and Warning)
    # Use sigmoid-like scaling based on how many std devs above mean
    if normal_std > 0:
        z_score = (error - normal_mean) / normal_std
        # Map z-score to [0, 1] range
        # At threshold: z_score typically ~2.3 (99th percentile)
        # We want threshold → 0.30 score
        anomaly_score = min(1.0, max(0.0, z_score / 7.5))
    else:
        anomaly_score = 0.0 if error < threshold else 0.5

    is_anomaly = error > threshold

    # Severity
    if anomaly_score < 0.15:
        severity = "NORMAL"
    elif anomaly_score < 0.30:
        severity = "WATCH"
    elif anomaly_score < 0.60:
        severity = "WARNING"
    else:
        severity = "CRITICAL"

    return {
        "anomaly_score": round(anomaly_score, 4),
        "is_anomaly": is_anomaly,
        "reconstruction_error": round(error, 6),
        "threshold": round(threshold, 6),
        "severity": severity,
    }


# ══════════════════════════════════════════════════════════════
# CLI Entry Points
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PlantIQ LSTM Autoencoder")
    parser.add_argument("--train", action="store_true", help="Train the autoencoder")
    parser.add_argument("--demo", action="store_true", help="Run inference demo")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    args = parser.parse_args()

    if args.train:
        print("=" * 65)
        print("  PlantIQ — LSTM Autoencoder Training")
        print("=" * 65)

        model, threshold, metadata = train_autoencoder(epochs=args.epochs)
        save_model(model, threshold, metadata)

        print("\n" + "=" * 65)
        print("  TRAINING COMPLETE")
        print("=" * 65)

    elif args.demo:
        print("=" * 65)
        print("  PlantIQ — LSTM Autoencoder Demo")
        print("=" * 65)

        model, threshold, metadata = load_model()
        normal_mean = metadata["normal_error_mean"]
        normal_std = metadata["normal_error_std"]

        # Load one curve of each type
        groups = load_all_curves()
        for ft in ["normal", "bearing_wear", "wet_material", "calibration_needed"]:
            fpaths = groups.get(ft, [])
            if not fpaths:
                continue
            # Test first 3 of each
            for fpath in fpaths[:3]:
                curve = np.load(fpath)
                result = compute_anomaly_score(model, curve, threshold, normal_mean, normal_std)
                fname = os.path.basename(fpath)
                print(f"  [{ft:25s}] {fname}: score={result['anomaly_score']:.4f}, "
                      f"severity={result['severity']:8s}, anomaly={result['is_anomaly']}")

        print("=" * 65)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
