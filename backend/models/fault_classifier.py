"""
PlantIQ — Fault Type Classifier (F2.2)
========================================
RandomForestClassifier trained on statistical features extracted from
power curves.  Given that an anomaly is detected, this classifier
diagnoses *what kind* of anomaly it is.

Feature extraction from curve (9 features per README spec):
  - mean, std, max, trend_slope, first_half_mean, second_half_mean,
    spike_count, area_under_curve, peak_time

Fault classes:
  Class 0 — normal             → No action
  Class 1 — bearing_wear       → "Schedule maintenance within 5 days"
  Class 2 — wet_material       → "Extend drying phase by 3–4 minutes"
  Class 3 — calibration_needed → "Machine calibration required"

Training:
    cd backend
    python models/fault_classifier.py --train

Output:
    models/trained/fault_classifier.pkl
    models/trained/fault_classifier_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
CURVES_DIR = os.path.join(DATA_DIR, "power_curves")
BATCH_CSV = os.path.join(DATA_DIR, "batch_data.csv")
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "fault_classifier.pkl")
META_PATH = os.path.join(ARTIFACT_DIR, "fault_classifier_meta.json")

# ── Label encoding ───────────────────────────────────────────
FAULT_LABELS = ["normal", "bearing_wear", "wet_material", "calibration_needed"]
LABEL_TO_INT = {label: idx for idx, label in enumerate(FAULT_LABELS)}
INT_TO_LABEL = {idx: label for idx, label in enumerate(FAULT_LABELS)}

# ── Feature names (9 statistical features per README) ────────
FEATURE_NAMES = [
    "mean",
    "std",
    "max",
    "trend_slope",
    "first_half_mean",
    "second_half_mean",
    "spike_count",
    "area_under_curve",
    "peak_time",
]


# ══════════════════════════════════════════════════════════════
# Feature Extraction
# ══════════════════════════════════════════════════════════════

def extract_features(curve: np.ndarray) -> np.ndarray:
    """Extract 9 statistical features from a single power curve.

    This matches the README Component 3 spec exactly.

    Parameters
    ----------
    curve : np.ndarray
        Power consumption readings (1D array, typically 1800 timesteps)

    Returns
    -------
    np.ndarray
        Shape (9,) — one value per feature
    """
    curve = np.asarray(curve, dtype=np.float64)
    n = len(curve)
    half = n // 2
    t = np.arange(n, dtype=np.float64)

    # 1. Mean power draw
    mean_val = float(np.mean(curve))

    # 2. Standard deviation (variability)
    std_val = float(np.std(curve))

    # 3. Maximum power draw
    max_val = float(np.max(curve))

    # 4. Trend slope (linear fit — positive slope = bearing wear)
    slope = float(np.polyfit(t, curve, 1)[0]) if n > 1 else 0.0

    # 5. First half mean (wet material anomalies concentrate here)
    first_half_mean = float(np.mean(curve[:half])) if half > 0 else mean_val

    # 6. Second half mean
    second_half_mean = float(np.mean(curve[half:])) if half > 0 else mean_val

    # 7. Spike count — number of timesteps with large jumps
    spike_count = float(np.sum(np.abs(np.diff(curve)) > 0.5)) if n > 1 else 0.0

    # 8. Area under curve (total energy proxy)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = float(_trapz(curve))

    # 9. Peak time — timestep with maximum power
    peak_time = float(np.argmax(curve))

    return np.array([
        mean_val,
        std_val,
        max_val,
        slope,
        first_half_mean,
        second_half_mean,
        spike_count,
        area,
        peak_time,
    ], dtype=np.float64)


def extract_features_batch(curves: list[np.ndarray]) -> np.ndarray:
    """Extract features from a list of curves.

    Returns
    -------
    np.ndarray
        Shape (n_curves, 9)
    """
    return np.array([extract_features(c) for c in curves])


# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════

def _load_curves_and_labels() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all power curves and their fault_type labels from CSV.

    Returns
    -------
    tuple
        X (n, 9) features, y (n,) integer labels, batch_ids list
    """
    df = pd.read_csv(BATCH_CSV)

    if "fault_type" not in df.columns:
        raise ValueError("batch_data.csv missing 'fault_type' column — regenerate data first")

    features_list = []
    labels_list = []
    batch_ids = []
    skipped = 0

    for _, row in df.iterrows():
        bid = row["batch_id"]
        fault = row["fault_type"]

        if fault not in LABEL_TO_INT:
            skipped += 1
            continue

        # Load the corresponding power curve
        curve_path = os.path.join(CURVES_DIR, f"{bid}.npy")
        if not os.path.exists(curve_path):
            skipped += 1
            continue

        curve = np.load(curve_path)
        feats = extract_features(curve)

        features_list.append(feats)
        labels_list.append(LABEL_TO_INT[fault])
        batch_ids.append(bid)

    if skipped > 0:
        print(f"  Skipped {skipped} batches (missing curve file or unknown fault type)")

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels_list, dtype=np.int64)

    return X, y, batch_ids


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def train(random_state: int = 42, test_size: float = 0.2, verbose: bool = True) -> dict:
    """Train the RandomForest fault classifier.

    Steps:
      1. Load all power curves and extract 9 features
      2. Split train/test (80/20)
      3. Train RandomForestClassifier(n_estimators=100)
      4. Evaluate (accuracy, per-class precision/recall/F1)
      5. Save model and metadata

    Returns
    -------
    dict
        Evaluation metrics
    """
    import joblib

    if verbose:
        print("\n" + "=" * 60)
        print("  PlantIQ — Fault Type Classifier Training (F2.2)")
        print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    t0 = time.time()
    if verbose:
        print(f"\n  Loading power curves from {CURVES_DIR}...")

    X, y, batch_ids = _load_curves_and_labels()

    if verbose:
        print(f"  Loaded {len(X)} curves in {time.time() - t0:.1f}s")
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls_id, cnt in zip(unique, counts):
            print(f"    Class {cls_id} ({INT_TO_LABEL[cls_id]:20s}): {cnt:5d} samples")

    # ── 2. Train/test split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Train RandomForest ────────────────────────────────
    if verbose:
        print("  Training RandomForestClassifier(n_estimators=100)...")

    t1 = time.time()
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - t1

    if verbose:
        print(f"  Training complete in {train_time:.2f}s")

    # ── 4. Evaluate ──────────────────────────────────────────
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train) * 100
    test_acc = accuracy_score(y_test, y_pred_test) * 100

    # Detailed classification report
    report_dict = classification_report(
        y_test, y_pred_test,
        target_names=FAULT_LABELS,
        output_dict=True,
        zero_division=0,
    )

    conf_mat = confusion_matrix(y_test, y_pred_test).tolist()

    if verbose:
        print(f"\n  ┌───────────────────────────────────────────────┐")
        print(f"  │ Train Accuracy: {train_acc:6.2f}%                      │")
        print(f"  │ Test Accuracy:  {test_acc:6.2f}%                      │")
        print(f"  └───────────────────────────────────────────────┘")
        print(f"\n  Per-class metrics (test set):")
        print(f"  {'Fault Type':20s} │ {'Precision':>9s} │ {'Recall':>6s} │ {'F1':>5s} │ {'Support':>7s}")
        print(f"  {'─' * 20}─┼{'─' * 11}┼{'─' * 8}┼{'─' * 7}┼{'─' * 9}")
        for label in FAULT_LABELS:
            m = report_dict[label]
            print(
                f"  {label:20s} │ {m['precision']:9.3f} │ "
                f"{m['recall']:6.3f} │ {m['f1-score']:5.3f} │ {int(m['support']):7d}"
            )
        macro = report_dict.get("macro avg", {})
        print(f"  {'─' * 20}─┼{'─' * 11}┼{'─' * 8}┼{'─' * 7}┼{'─' * 9}")
        print(
            f"  {'Overall (macro)':20s} │ {macro.get('precision', 0):9.3f} │ "
            f"{macro.get('recall', 0):6.3f} │ {macro.get('f1-score', 0):5.3f} │ "
            f"{int(macro.get('support', 0)):7d}"
        )

    # Feature importance ranking
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importance = {
        FEATURE_NAMES[i]: round(float(importances[i]), 4)
        for i in sorted_idx
    }

    if verbose:
        print(f"\n  Feature importance ranking:")
        for rank, idx in enumerate(sorted_idx, 1):
            bar = "█" * int(importances[idx] * 50)
            print(f"    {rank}. {FEATURE_NAMES[idx]:20s} {importances[idx]:.4f} {bar}")

    # ── 5. Save model + metadata ─────────────────────────────
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    meta = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "label_encoding": LABEL_TO_INT,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy_pct": round(train_acc, 2),
        "test_accuracy_pct": round(test_acc, 2),
        "per_class_metrics": {
            label: {
                "precision": round(report_dict[label]["precision"], 4),
                "recall": round(report_dict[label]["recall"], 4),
                "f1": round(report_dict[label]["f1-score"], 4),
                "support": int(report_dict[label]["support"]),
            }
            for label in FAULT_LABELS
        },
        "confusion_matrix": conf_mat,
        "feature_importance": feature_importance,
        "training_time_sec": round(train_time, 2),
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\n  Model saved: {MODEL_PATH}")
        print(f"  Metadata saved: {META_PATH}")
        print(f"  Total time: {time.time() - t0:.1f}s")

    return meta


# ══════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════

class FaultClassifier:
    """Inference wrapper for the trained fault classifier.

    Loaded once at API startup; used by the anomaly route to
    classify fault type from a power curve segment.
    """

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.is_loaded: bool = False
        self._load()

    def _load(self):
        """Load the trained model from disk."""
        if not os.path.exists(MODEL_PATH):
            print(f"[FaultClassifier] No trained model at {MODEL_PATH}")
            return

        import joblib
        self.model = joblib.load(MODEL_PATH)
        self.is_loaded = True
        print(f"  Fault classifier loaded from {MODEL_PATH}")

    def classify(self, curve: np.ndarray) -> tuple[str, float]:
        """Classify fault type from a power curve.

        Parameters
        ----------
        curve : np.ndarray
            Power consumption readings (1D array)

        Returns
        -------
        tuple[str, float]
            (fault_type label, confidence probability)
        """
        if not self.is_loaded or self.model is None:
            # Fallback: return normal with low confidence
            return "normal", 0.50

        feats = extract_features(curve).reshape(1, -1)
        proba = self.model.predict_proba(feats)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        return INT_TO_LABEL[pred_class], round(confidence, 4)

    def classify_from_features(self, features: np.ndarray) -> tuple[str, float]:
        """Classify from pre-extracted features.

        Parameters
        ----------
        features : np.ndarray
            Shape (9,) — the 9 statistical features

        Returns
        -------
        tuple[str, float]
            (fault_type label, confidence probability)
        """
        if not self.is_loaded or self.model is None:
            return "normal", 0.50

        feats = features.reshape(1, -1)
        proba = self.model.predict_proba(feats)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        return INT_TO_LABEL[pred_class], round(confidence, 4)


# ══════════════════════════════════════════════════════════════
# Standalone loading function (for main.py startup)
# ══════════════════════════════════════════════════════════════

def load_classifier() -> Optional[FaultClassifier]:
    """Load and return the FaultClassifier, or None if not trained."""
    if not os.path.exists(MODEL_PATH):
        return None
    return FaultClassifier()


# ══════════════════════════════════════════════════════════════
# CLI Entrypoint
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlantIQ Fault Type Classifier (F2.2)")
    parser.add_argument("--train", action="store_true", help="Train the classifier")
    parser.add_argument("--evaluate", action="store_true", help="Print evaluation metrics")
    args = parser.parse_args()

    if args.train:
        train(verbose=True)
    elif args.evaluate:
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
            print(json.dumps(meta, indent=2))
        else:
            print("No training metadata found. Run --train first.")
    else:
        parser.print_help()
