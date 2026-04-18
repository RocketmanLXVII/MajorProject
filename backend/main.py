"""
MulTiCheat — FastAPI Application Entry Point

Provides:
- CORS middleware
- Lifespan handler for startup/shutdown tasks
- /health endpoint
- Structured logging configuration
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import Settings, get_settings

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────

def configure_logging(settings: Settings) -> None:
    """Configure root logger with the format and level from settings."""
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    # Avoid duplicate handlers on reload
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(settings.log_level)
        formatter = logging.Formatter(settings.log_format)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    settings = get_settings()
    configure_logging(settings)

    logger.info("=" * 60)
    logger.info("  %s v%s starting up", settings.app_name, settings.app_version)
    logger.info("  Debug mode: %s", settings.debug)
    logger.info("  Log level : %s", settings.log_level)
    logger.info("=" * 60)

    # Ensure runtime directories exist
    settings.ensure_directories()
    logger.info("Upload dir  : %s", settings.upload_path)
    logger.info("Evidence dir: %s", settings.evidence_path)

    yield  # ── app is running ──

    logger.info("%s shutting down.", settings.app_name)


# ──────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────

def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-based exam cheating detection system",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health endpoint ────────────────────────
    @app.get("/health", tags=["system"])
    async def health_check() -> JSONResponse:
        """Return basic health information."""
        return JSONResponse(
            content={
                "status": "ok",
                "app_name": settings.app_name,
                "version": settings.app_version,
                "timestamp": time.time(),
            }
        )

    # ── Root redirect / info ───────────────────
    @app.get("/", tags=["system"])
    async def root() -> JSONResponse:
        """Root endpoint with basic API info."""
        return JSONResponse(
            content={
                "message": f"Welcome to {settings.app_name}",
                "docs": "/docs",
                "health": "/health",
            }
        )

    # ── Routers ─────────────────────────────────
    from backend.routers.video_router import router as video_router
    from backend.routers.analysis_router import router as analysis_router
    from backend.routers.export_router import router as export_router
    
    app.include_router(video_router)
    app.include_router(analysis_router)
    app.include_router(export_router)

    return app


# Module-level app instance used by uvicorn
app = create_app()
