"""
MulTiCheat Pro — Student ORM Model
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String, JSON
from backend.models.base import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    roll_number = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    face_embedding = Column(JSON, nullable=True)  # List[float] embedding vector
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
