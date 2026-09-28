"""
MulTiCheat — FastAPI Application Entry Point

Provides:
- CORS middleware with configurable origins
- Lifespan handler for startup/shutdown tasks & database init
- /health endpoint
- Structured logging configuration
- Job, Analysis, Video, Export, Student & WebSockets endpoints
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
from backend.database import init_db


def configure_logging(settings: Settings) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(settings.log_level)
        formatter = logging.Formatter(settings.log_format)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings)

    logger.info("=" * 60)
    logger.info("  %s v%s starting up", settings.app_name, settings.app_version)
    logger.info("  Debug mode: %s", settings.debug)
    logger.info("  Profile   : %s", settings.active_profile.value)
    logger.info("=" * 60)

    settings.ensure_directories()
    init_db()

    yield

    logger.info("%s shutting down.", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production AI-based exam cheating detection system",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    async def health_check() -> JSONResponse:
        return JSONResponse(
            content={
                "status": "ok",
                "app_name": settings.app_name,
                "version": settings.app_version,
                "profile": settings.active_profile.value,
                "timestamp": time.time(),
            }
        )

    @app.get("/", tags=["system"])
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "message": f"Welcome to {settings.app_name}",
                "docs": "/docs",
                "health": "/health",
            }
        )

    from backend.routers.video_router import router as video_router
    from backend.routers.analysis_router import router as analysis_router
    from backend.routers.export_router import router as export_router
    from backend.routers.student_router import student_router
    from backend.routers.ws_router import ws_router

    app.include_router(video_router)
    app.include_router(analysis_router)
    app.include_router(export_router)
    app.include_router(student_router)
    app.include_router(ws_router)

    return app


app = create_app()
