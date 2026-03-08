"""
PlantIQ — API Key Authentication Middleware
=============================================
Provides optional API key authentication via X-API-Key header.

When PLANTIQ_API_KEY environment variable is set, all endpoints
(except /docs, /redoc, /openapi.json, and /health) require the
X-API-Key header to match.

When PLANTIQ_API_KEY is not set, authentication is disabled
(development mode).
"""

from __future__ import annotations

import os
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("plantiq.auth")

# Paths that never require authentication
PUBLIC_PATHS = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/health",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that validates X-API-Key header when PLANTIQ_API_KEY is set."""

    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self.api_key = api_key or os.environ.get("PLANTIQ_API_KEY")
        if self.api_key:
            logger.info("API key authentication enabled")
        else:
            logger.info("API key authentication disabled (set PLANTIQ_API_KEY to enable)")

    async def dispatch(self, request: Request, call_next):
        # Skip auth if no key configured (dev mode)
        if not self.api_key:
            return await call_next(request)

        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Skip preflight CORS requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Validate X-API-Key header
        provided_key = request.headers.get("X-API-Key")
        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-API-Key header"},
            )

        if provided_key != self.api_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)
