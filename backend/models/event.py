"""
MulTiCheat Pro — Aggregated Malpractice Event ORM Model
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from backend.models.base import Base


class MalpracticeEvent(Base):
    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, index=True)
    student_id = Column(String(36), ForeignKey("students.id"), nullable=True, index=True)
    exam_id = Column(String(36), ForeignKey("exams.id"), nullable=True)
    track_id = Column(Integer, nullable=True)
    seat_number = Column(Integer, nullable=True)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    start_time_seconds = Column(Float, nullable=False)
    end_time_seconds = Column(Float, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    evidence_frames = Column(JSON, default=list)  # List[int]
    evidence_image_paths = Column(JSON, default=list)  # List[str]
    description = Column(Text, nullable=True)
    supporting_signals = Column(JSON, default=list)
    contradicting_signals = Column(JSON, default=list)
    reviewed = Column(Boolean, default=False)
    reviewer_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_events_video_type", "video_id", "event_type"),
        Index("idx_events_severity_conf", "severity", "confidence"),
    )
