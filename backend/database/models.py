"""
MulTiCheat — Database Models

SQLAlchemy ORM models for video records, background jobs, student tracks,
seats, detections, events, evidence files, and reviewer feedback.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class VideoRecord(Base):
    __tablename__ = "videos"

    video_id = Column(String(64), primary_key=True)
    original_filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    fps = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    frame_count = Column(Integer, nullable=False)
    codec = Column(String(32), default="unknown")
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String(32), default="ready")

    jobs = relationship("AnalysisJobRecord", back_populates="video", cascade="all, delete-orphan")
    events = relationship("EventRecord", back_populates="video", cascade="all, delete-orphan")


class AnalysisJobRecord(Base):
    __tablename__ = "analysis_jobs"

    job_id = Column(String(64), primary_key=True)
    video_id = Column(String(64), ForeignKey("videos.video_id"), nullable=False)
    status = Column(String(32), default="QUEUED")  # QUEUED, INITIALIZING, PROCESSING, COMPLETED, FAILED
    progress_pct = Column(Float, default=0.0)
    current_step = Column(String(128), default="Queued")
    error_detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    profile = Column(String(32), default="BALANCED")
    result_json = Column(JSON, nullable=True)

    video = relationship("VideoRecord", back_populates="jobs")


class StudentTrackRecord(Base):
    __tablename__ = "student_tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(64), ForeignKey("videos.video_id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    anonymous_label = Column(String(64), nullable=False)  # e.g. StudentTrack_023
    seat_id = Column(String(64), nullable=True)  # e.g. Seat_A12
    first_frame = Column(Integer, default=0)
    last_frame = Column(Integer, default=0)
    visibility_score = Column(Float, default=1.0)
    occlusion_state = Column(String(32), default="VISIBLE")

    events = relationship("EventRecord", back_populates="student_track")


class EventRecord(Base):
    __tablename__ = "events"

    event_id = Column(String(64), primary_key=True)
    video_id = Column(String(64), ForeignKey("videos.video_id"), nullable=False)
    track_id = Column(Integer, nullable=True)
    student_track_id = Column(Integer, ForeignKey("student_tracks.id"), nullable=True)
    event_type = Column(String(64), nullable=False)
    severity = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    start_timestamp = Column(Float, nullable=False)
    end_timestamp = Column(Float, nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    explanation_text = Column(Text, default="")
    supporting_signals = Column(JSON, default=list)
    contradicting_signals = Column(JSON, default=list)
    model_version = Column(String(64), default="v2.0")
    reviewer_status = Column(String(32), default="UNREVIEWED")  # UNREVIEWED, CONFIRMED, DISMISSED, MARK_INCORRECT

    video = relationship("VideoRecord", back_populates="events")
    student_track = relationship("StudentTrackRecord", back_populates="events")
    evidence_files = relationship("EvidenceFileRecord", back_populates="event", cascade="all, delete-orphan")


class EvidenceFileRecord(Base):
    __tablename__ = "evidence_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), ForeignKey("events.event_id"), nullable=False)
    video_id = Column(String(64), nullable=False)
    file_type = Column(String(32), nullable=False)  # raw, annotated, clip, crop
    file_path = Column(String(512), nullable=False)
    frame_index = Column(Integer, nullable=True)

    event = relationship("EventRecord", back_populates="evidence_files")


class ReviewerFeedbackRecord(Base):
    __tablename__ = "reviewer_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), ForeignKey("events.event_id"), nullable=False)
    decision = Column(String(32), nullable=False)  # CONFIRM, DISMISS, MARK_INCORRECT, NOT_SURE
    notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
