"""
PlantIQ — Database Layer (Layer 1 — Data Foundation)
======================================================
SQLite for hackathon, PostgreSQL-ready via SQLAlchemy.

Per README §4 — Component 1.3 (Prediction Store), §13 (SQLite/PostgreSQL):
  "SQLite stores the entire database as a single file on disk.
   The transition from SQLite to PostgreSQL requires changing one
   configuration value."

Usage:
    from database import get_db, init_db

    # At startup
    init_db()

    # In route handlers
    db = next(get_db())
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# ── Database file location ───────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BACKEND_DIR, "plantiq.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ── SQLAlchemy engine + session ──────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite-specific
    echo=False,  # Set True for SQL debug logging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Declarative base for ORM models ─────────────────────────
Base = declarative_base()


def init_db():
    """Create all tables if they don't exist.

    Called once at application startup.  Idempotent — safe to call
    multiple times.
    """
    from database.models import (  # noqa: F401 — import for side effects
        PredictionRecord,
        ModelVersion,
        AlertRecord,
        FeedbackMetric,
        AuditLog,
    )
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a database session for use in FastAPI dependency injection.

    Usage in route handler:
        from database import get_db
        db = next(get_db())
        ...
        db.close()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Session:
    """Context manager for database sessions (for non-route usage).

    Usage:
        with get_db_session() as db:
            db.add(record)
            db.commit()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
