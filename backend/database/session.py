"""
MulTiCheat — Database Session & Engine Initialization
"""

from __future__ import annotations

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from backend.config import get_settings
from backend.database.models import Base

logger = logging.getLogger(__name__)

def get_engine():
    settings = get_settings()
    db_path = settings.db_file_path
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    return engine

def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", get_settings().db_file_path)

def get_db_session():
    engine = get_engine()
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
