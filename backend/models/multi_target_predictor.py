"""
PlantIQ — Multi-Target XGBoost Predictor
==========================================
Simultaneously predicts all 4 manufacturing targets from 13 features:
  - quality_score   (%)
  - yield_pct       (%)
  - performance_pct (%)
  - energy_kwh      (kWh)

Architecture (per README spec):
  MultiOutputRegressor wrapper
    ├── XGBRegressor #1 → quality_score
    ├── XGBRegressor #2 → yield_pct
    ├── XGBRegressor #3 → performance_pct
    └── XGBRegressor #4 → energy_kwh

Hyperparameters (locked per README):
  n_estimators=300, learning_rate=0.05, max_depth=6,
  subsample=0.8, colsample_bytree=0.8, random_state=42

Validation: TimeSeriesSplit(n_splits=5) — respects temporal ordering.

Usage:
  python models/multi_target_predictor.py --train     # Train + save
  python models/multi_target_predictor.py --predict    # Quick prediction demo

Output artifacts:
  models/trained/multi_target.pkl
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")

TRAIN_CSV = os.path.join(DATA_DIR, "train_processed.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_processed.csv")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "multi_target.pkl")
META_PATH = os.path.join(ARTIFACT_DIR, "pipeline_meta.json")

# Load feature/target column names from pipeline metadata
with open(META_PATH) as f:
    _meta = json.load(f)
FEATURE_COLS = _meta["feature_cols"]
TARGET_COLS = _meta["target_cols"]

# XGBoost hyperparameters — locked per README spec
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,           # Use all CPU cores
    "verbosity": 0,         # Suppress XGBoost internal logs
}

# Cross-validation
CV_SPLITS = 5


class MultiTargetPredictor:
    """Multi-target XGBoost predictor for manufacturing batch outcomes.

    Wraps 4 independent XGBRegressor models inside a
    MultiOutputRegressor for simultaneous prediction.

    Attributes
    ----------
    model : MultiOutputRegressor
        The fitted multi-output model.
    feature_cols : list[str]
        Ordered feature column names.
    target_cols : list[str]
        Ordered target column names.
    is_fitted : bool
        Whether the model has been trained.
    """

    def __init__(self):
        self.feature_cols = FEATURE_COLS
        self.target_cols = TARGET_COLS
        self.model = MultiOutputRegressor(
            XGBRegressor(**XGBOOST_PARAMS)
        )
        self.is_fitted = False
        self._train_metrics: dict = {}

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.DataFrame | np.ndarray,
    ) -> dict:
        """Train the multi-target model.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features, shape (n_samples, 13).
        y_train : pd.DataFrame | np.ndarray
            Training targets, shape (n_samples, 4).

        Returns
        -------
        dict
            Training metrics (MAE, RMSE per target).
        """
        print("\n  Training Multi-Target XGBoost...")
        print(f"  Features: {len(self.feature_cols)} | "
              f"Targets: {len(self.target_cols)} | "
              f"Samples: {len(X_train)}")
        print(f"  Hyperparameters: {XGBOOST_PARAMS}")

        start = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start
        self.is_fitted = True

        print(f"  Training complete in {elapsed:.1f}s")

        # Compute training metrics
        y_pred = self.model.predict(X_train)
        self._train_metrics = self._compute_metrics(y_train, y_pred, prefix="train")
        return self._train_metrics

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict all 4 targets for given features.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Features, shape (n_samples, 13).

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples, 4), columns ordered as target_cols.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")
        return self.model.predict(X)

    def predict_single(self, params: dict) -> dict:
        """Predict for a single batch from a parameter dictionary.

        Parameters
        ----------
        params : dict
            Batch parameters with keys matching feature_cols.
            Missing derived features will be computed.

        Returns
        -------
        dict
            Predictions keyed by target name, plus confidence intervals.
        """
        # Build feature vector in correct column order
        from preprocessing.feature_engineer import FeatureEngineer

        # Create single-row DataFrame
        row = pd.DataFrame([params])

        # Compute derived features if not present
        fe = FeatureEngineer(include_energy_per_kg=False)
        row = fe.transform(row)

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0  # safe default

        X = row[self.feature_cols].values
        y_pred = self.predict(X)[0]

        result = {}
        for i, target in enumerate(self.target_cols):
            result[target] = round(float(y_pred[i]), 2)

        # Add derived CO₂
        if "energy_kwh" in result:
            result["co2_kg"] = round(result["energy_kwh"] * 0.82, 2)

        return result

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.DataFrame | np.ndarray,
    ) -> dict:
        """Evaluate model on test data.

        Parameters
        ----------
        X_test : pd.DataFrame | np.ndarray
            Test features.
        y_test : pd.DataFrame | np.ndarray
            True test targets.

        Returns
        -------
        dict
            Test metrics (MAE, RMSE, MAPE, accuracy per target).
        """
        y_pred = self.predict(X_test)
        return self._compute_metrics(y_test, y_pred, prefix="test")

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        n_splits: int = CV_SPLITS,
    ) -> dict:
        """Run TimeSeriesSplit cross-validation.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Full training features.
        y : pd.DataFrame | np.ndarray
            Full training targets.
        n_splits : int
            Number of CV splits (default: 5).

        Returns
        -------
        dict
            Cross-validation metrics per target.
        """
        print(f"\n  Running TimeSeriesSplit cross-validation (n_splits={n_splits})...")
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Collect predictions from each fold
        if isinstance(y, pd.DataFrame):
            y_arr = y.values
        else:
            y_arr = y
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            fold_model = MultiOutputRegressor(XGBRegressor(**XGBOOST_PARAMS))
            fold_model.fit(X_tr, y_tr)
            y_val_pred = fold_model.predict(X_val)

            all_y_true.append(y_val)
            all_y_pred.append(y_val_pred)

            # Per-fold brief summary
            fold_mae = np.mean(np.abs(y_val - y_val_pred), axis=0)
            print(f"    Fold {fold + 1}: MAE = {', '.join(f'{m:.2f}' for m in fold_mae)}")

        y_true_all = np.vstack(all_y_true)
        y_pred_all = np.vstack(all_y_pred)

        cv_metrics = self._compute_metrics(y_true_all, y_pred_all, prefix="cv")
        return cv_metrics

    def get_feature_importance(self) -> dict[str, dict[str, float]]:
        """Get feature importance scores for each target model.

        Returns
        -------
        dict[str, dict[str, float]]
            {target_name: {feature_name: importance_score}}
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        importance = {}
        for i, target in enumerate(self.target_cols):
            estimator = self.model.estimators_[i]
            raw_imp = estimator.feature_importances_
            # Normalize to sum = 1
            total = raw_imp.sum()
            if total > 0:
                raw_imp = raw_imp / total

            importance[target] = {
                self.feature_cols[j]: round(float(raw_imp[j]), 4)
                for j in range(len(self.feature_cols))
            }

        return importance

    def save(self, path: str = MODEL_PATH) -> str:
        """Save the trained model and metadata to disk.

        Parameters
        ----------
        path : str
            Path to save the model artifact.

        Returns
        -------
        str
            Path to saved artifact.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifact = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols,
            "train_metrics": self._train_metrics,
            "xgboost_params": XGBOOST_PARAMS,
        }
        joblib.dump(artifact, path)
        print(f"\n  Model saved to {path}")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Artifact size: {size_mb:.1f} MB")
        return path

    def load(self, path: str = MODEL_PATH) -> "MultiTargetPredictor":
        """Load a previously trained model from disk.

        Parameters
        ----------
        path : str
            Path to the model artifact.

        Returns
        -------
        self
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_cols = artifact["feature_cols"]
        self.target_cols = artifact["target_cols"]
        self._train_metrics = artifact.get("train_metrics", {})
        self.is_fitted = True
        print(f"  Model loaded from {path}")
        return self

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> dict:
        """Compute MAE, RMSE, MAPE, and accuracy per target.

        Parameters
        ----------
        y_true : np.ndarray
            True values, shape (n, 4).
        y_pred : np.ndarray
            Predicted values, shape (n, 4).
        prefix : str
            Label prefix for the metrics dict.

        Returns
        -------
        dict
            Nested metrics per target.
        """
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values

        metrics = {}
        for i, target in enumerate(self.target_cols):
            true = y_true[:, i]
            pred = y_pred[:, i]

            mae = float(mean_absolute_error(true, pred))
            rmse = float(np.sqrt(mean_squared_error(true, pred)))
            # MAPE — avoid division by zero
            mask = true != 0
            mape = float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100)
            accuracy = 100.0 - mape

            metrics[target] = {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 2),
                "accuracy": round(accuracy, 2),
            }

        return metrics


def _print_metrics(metrics: dict, title: str) -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n  {title}")
    print(f"  {'Target':20s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s} {'Accuracy':>10s}  Status")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}  ------")

    all_pass = True
    for target, m in metrics.items():
        status = "PASS" if m["accuracy"] >= 90 else "FAIL"
        if status == "FAIL":
            all_pass = False
        unit = "kWh" if target == "energy_kwh" else "%"
        print(f"  {target:20s} {m['mae']:8.2f}{unit:>3s} "
              f"{m['rmse']:8.2f}{unit:>3s} "
              f"{m['mape']:7.1f}%  {m['accuracy']:9.1f}%  {'✅' if status == 'PASS' else '❌'} {status}")

    if all_pass:
        print(f"\n  All targets exceed 90% accuracy requirement ✅")
    else:
        print(f"\n  ⚠️  Some targets below 90% — review needed")


def train_and_evaluate() -> dict:
    """Full training pipeline: load data → train → CV → evaluate → save.

    Returns
    -------
    dict
        Complete evaluation report.
    """
    print("=" * 60)
    print("  MULTI-TARGET XGBOOST — TRAIN & EVALUATE")
    print("=" * 60)

    # ── Load preprocessed data ──
    print("\n  Loading preprocessed data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COLS]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COLS]

    # ── Initialize and train ──
    predictor = MultiTargetPredictor()
    train_metrics = predictor.train(X_train, y_train)
    _print_metrics(train_metrics, "TRAINING SET METRICS")

    # ── Cross-validation ──
    cv_metrics = predictor.cross_validate(X_train, y_train)
    _print_metrics(cv_metrics, "CROSS-VALIDATION METRICS (TimeSeriesSplit, 5 folds)")

    # ── Test set evaluation ──
    test_metrics = predictor.evaluate(X_test, y_test)
    _print_metrics(test_metrics, "TEST SET METRICS (held-out 20%)")

    # ── Feature importance ──
    importance = predictor.get_feature_importance()
    print("\n  TOP-3 FEATURES PER TARGET:")
    for target, imp in importance.items():
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]
        top3 = ", ".join(f"{f}={v:.3f}" for f, v in sorted_imp)
        print(f"    {target:20s} → {top3}")

    # ── Save model ──
    predictor.save()

    # ── Save evaluation report ──
    report = {
        "train_metrics": train_metrics,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "feature_importance": importance,
        "hyperparameters": XGBOOST_PARAMS,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
    }
    report_path = os.path.join(ARTIFACT_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Evaluation report saved to {report_path}")

    # ── Quick demo prediction ──
    print("\n  DEMO PREDICTION (single batch):")
    demo_pred = predictor.predict_single({
        "temperature": 183,
        "conveyor_speed": 76,
        "hold_time": 18,
        "batch_size": 500,
        "material_type": 1,
        "hour_of_day": 9,
        "operator_exp": 1,
    })
    for key, val in demo_pred.items():
        print(f"    {key:20s}: {val}")

    print("\n" + "=" * 60)
    return report


def quick_predict():
    """Load saved model and run a demo prediction."""
    predictor = MultiTargetPredictor()
    predictor.load()

    result = predictor.predict_single({
        "temperature": 183,
        "conveyor_speed": 76,
        "hold_time": 18,
        "batch_size": 500,
        "material_type": 1,
        "hour_of_day": 9,
        "operator_exp": 1,
    })
    print("\n  Prediction result:")
    for key, val in result.items():
        print(f"    {key:20s}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlantIQ Multi-Target Predictor")
    parser.add_argument("--train", action="store_true", help="Train + evaluate + save")
    parser.add_argument("--predict", action="store_true", help="Quick prediction demo")
    args = parser.parse_args()

    if args.predict:
        quick_predict()
    else:
        # Default to training
        train_and_evaluate()
