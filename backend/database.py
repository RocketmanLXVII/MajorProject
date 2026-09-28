"""
MulTiCheat Pro — Database Connection & Async Session Manager

Supports both PostgreSQL 16 (with TimescaleDB) and local SQLite (aiosqlite).
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Determine DB engine string
DB_URL = settings.db_url
if "sqlite" in DB_URL and "aiosqlite" not in DB_URL:
    DB_URL = DB_URL.replace("sqlite:///", "sqlite+aiosqlite:///")

is_sqlite = "sqlite" in DB_URL

# Async Engine setup
connect_args = {"check_same_thread": False} if is_sqlite else {}

async_engine = create_async_engine(
    DB_URL,
    echo=False,
    future=True,
    connect_args=connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

Base = declarative_base()


async def init_db_models() -> None:
    """Initialize all ORM tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully (%s)", DB_URL)


def init_db() -> None:
    """Synchronous init db helper for application startup."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(init_db_models())
        else:
            loop.run_until_complete(init_db_models())
    except Exception:
        pass


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for async FastAPI endpoints."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
