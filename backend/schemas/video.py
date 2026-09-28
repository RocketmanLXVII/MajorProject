"""
MulTiCheat Pro — Video API Pydantic Schemas
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class VideoBase(BaseModel):
    filename: str
    file_size_mb: Optional[float] = None
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None


class VideoCreate(VideoBase):
    exam_id: Optional[str] = None
    file_path: str


class VideoResponse(VideoBase):
    id: str
    exam_id: Optional[str] = None
    status: str
    progress_percent: int
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
