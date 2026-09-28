"""
MulTiCheat Pro — Malpractice Event Pydantic Schemas
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class EventBase(BaseModel):
    video_id: str
    event_type: str
    severity: str
    start_frame: int
    end_frame: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    confidence: float
    description: Optional[str] = None


class EventUpdate(BaseModel):
    reviewed: Optional[bool] = None
    reviewer_notes: Optional[str] = None


class EventResponse(EventBase):
    id: str
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    seat_number: Optional[int] = None
    evidence_frames: List[int] = []
    evidence_image_paths: List[str] = []
    supporting_signals: List[str] = []
    contradicting_signals: List[str] = []
    reviewed: bool = False
    reviewer_notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
