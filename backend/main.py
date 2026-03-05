"""
PlantIQ — FastAPI Application Entry Point
============================================
Loads ML models at startup, registers all route modules, and serves
the PlantIQ Manufacturing Intelligence REST API.

Run with:
    cd backend
    uvicorn main:app --reload --port 8000

API docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Ensure backend is on sys.path ────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plantiq")

# ── Global model references (populated at startup) ───────────
predictor = None          # MultiTargetPredictor
shap_explainer = None     # ShapExplainer
english_converter = None  # PlainEnglishConverter
lstm_model = None         # LSTM Autoencoder artifacts dict
golden_manager = None     # GoldenSignatureManager
target_engine = None      # AdaptiveTargetEngine


def _load_models():
    """Load all ML models and explainability components.

    Called once during application startup (lifespan).
    Models are stored as module-level globals so route handlers
    can import them with `from main import predictor`.
    """
    global predictor, shap_explainer, english_converter
    global lstm_model, golden_manager, target_engine

    # ── 1. Multi-Target XGBoost Predictor ────────────────────
    logger.info("Loading Multi-Target XGBoost predictor...")
    t0 = time.time()
    from models.multi_target_predictor import MultiTargetPredictor
    predictor = MultiTargetPredictor()
    predictor.load()
    logger.info(
        "Predictor loaded in %.2fs — %d features, %d targets",
        time.time() - t0,
        len(predictor.feature_cols),
        len(predictor.target_cols),
    )

    # ── 2. SHAP Explainer ────────────────────────────────────
    logger.info("Loading SHAP explainer (TreeExplainer × 4 targets)...")
    t0 = time.time()
    from explainability.shap_explainer import ShapExplainer
    shap_explainer = ShapExplainer()
    logger.info(
        "SHAP explainer loaded in %.2fs — baselines: %s",
        time.time() - t0,
        {k: round(v, 2) for k, v in shap_explainer.baselines.items()},
    )

    # ── 3. Plain English Converter ───────────────────────────
    from explainability.plain_english import PlainEnglishConverter
    english_converter = PlainEnglishConverter()
    logger.info("Plain English converter initialized")

    # ── 4. LSTM Autoencoder (optional — only if trained) ─────
    lstm_model_path = os.path.join(BACKEND_DIR, "models", "trained", "lstm_autoencoder.pt")
    if os.path.exists(lstm_model_path):
        logger.info("Loading LSTM Autoencoder...")
        t0 = time.time()
        try:
            from models.lstm_autoencoder import load_model as load_lstm
            _model, _threshold, _meta = load_lstm()
            lstm_model = {
                "model": _model,
                "threshold": _threshold,
                "normal_mean": _meta.get("normal_error_mean", 0.0),
                "normal_std": _meta.get("normal_error_std", 1.0),
            }
            logger.info(
                "LSTM Autoencoder loaded in %.2fs — threshold: %.6f",
                time.time() - t0, _threshold,
            )
        except Exception as e:
            logger.warning("LSTM Autoencoder failed to load (will use statistical fallback): %s", e)
            lstm_model = None
    else:
        logger.info("LSTM Autoencoder not trained yet — anomaly detection will use statistical method")

    # ── 5. Golden Signature Manager ──────────────────────────
    logger.info("Initializing Golden Signature Manager...")
    t0 = time.time()
    try:
        from models.golden_signature import GoldenSignatureManager
        golden_manager = GoldenSignatureManager()
        n_sigs = len(golden_manager.signatures)
        logger.info(
            "Golden Signature Manager ready in %.2fs — %d saved signatures",
            time.time() - t0, n_sigs,
        )
    except Exception as e:
        logger.warning("Golden Signature Manager failed to initialize: %s", e)
        golden_manager = None

    # ── 6. Adaptive Target Engine ────────────────────────────
    logger.info("Initializing Adaptive Target Engine...")
    t0 = time.time()
    try:
        from models.adaptive_targets import AdaptiveTargetEngine
        target_engine = AdaptiveTargetEngine()
        has_baseline = bool(target_engine.baseline)
        logger.info(
            "Adaptive Target Engine ready in %.2fs — baseline: %s",
            time.time() - t0,
            "loaded" if has_baseline else "not yet initialized (call POST /targets/initialize)",
        )
    except Exception as e:
        logger.warning("Adaptive Target Engine failed to initialize: %s", e)
        target_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — loads models on startup."""
    logger.info("=" * 60)
    logger.info("PlantIQ API starting up...")
    logger.info("=" * 60)

    t_start = time.time()
    _load_models()
    logger.info(
        "All models loaded in %.2fs — API ready!",
        time.time() - t_start,
    )

    yield  # Application is running

    logger.info("PlantIQ API shutting down...")


# ── Create FastAPI app ───────────────────────────────────────
app = FastAPI(
    title="PlantIQ Manufacturing Intelligence API",
    description=(
        "AI-driven predictive analytics for industrial batch manufacturing. "
        "Predicts Quality, Yield, Performance, and Energy; detects anomalies; "
        "and provides SHAP-based explanations with operator recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware (Next.js dev server on port 3000) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Next.js dev
        "http://127.0.0.1:3000",
        "http://localhost:8000",    # Swagger UI
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register route modules ───────────────────────────────────
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from api.routes.anomaly import router as anomaly_router
from api.routes.explain import router as explain_router
from api.routes.features import router as features_router
from api.routes.golden_signature import router as golden_sig_router
from api.routes.targets import router as targets_router
from api.routes.hackathon import router as hackathon_router

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(anomaly_router)
app.include_router(explain_router)
app.include_router(features_router)
app.include_router(golden_sig_router)
app.include_router(targets_router)
app.include_router(hackathon_router)


# ── Root redirect to docs ────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# ── CLI entrypoint ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
