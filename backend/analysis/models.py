"""
MulTiCheat — Analysis Data Models

Pydantic models for detections, frame results, flag events, and overall analysis output.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class FlagCategory(str, Enum):
    """Cheating flag categories."""
    LOOK_AROUND = "look_around"
    PHONE_USE = "phone_use"
    NOTE_PASSING = "note_passing"
    PAPER_COPYING = "paper_copying"
    UNAUTHORIZED_OBJECT = "unauthorized_object"
    SUSPICIOUS_HAND_MOVEMENT = "suspicious_hand_movement"
    PROLONGED_NON_ATTENTIVE = "prolonged_non_attentive_behavior"
    SUSPICIOUS_MOVEMENT = "suspicious_movement"
    NORMAL = "normal"


class Severity(str, Enum):
    """Severity levels for flagged events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ──────────────────────────────────────────────
# Detection (single detection on a frame)
# ──────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Bounding box coordinates (pixels, top-left origin)."""
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., ge=0)
    h: int = Field(..., ge=0)


class Detection(BaseModel):
    """A single detection on a single frame."""
    label: str = Field(..., description="Detection class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box if applicable")
    track_id: Optional[int] = Field(None, description="Continuous temporal tracking ID representing specific entities")
    metadata: dict = Field(default_factory=dict, description="Extra detector-specific data")


# ──────────────────────────────────────────────
# Frame-level result
# ──────────────────────────────────────────────

class FrameResult(BaseModel):
    """Aggregated detection results for a single frame."""
    frame_index: int = Field(..., ge=0)
    timestamp_seconds: float = Field(..., ge=0.0)
    detections: list[Detection] = Field(default_factory=list)
    aggregate_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Combined suspicion score for this frame (0 = normal, 1 = highly suspicious)",
    )


FrameAnalysisResult = FrameResult


# ──────────────────────────────────────────────
# Flag event (aggregated from multiple frames)
# ──────────────────────────────────────────────

class FlagEvent(BaseModel):
    """A flagged suspicious event spanning one or more frames."""
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    video_id: str
    track_id: Optional[int] = Field(None, description="The unique student/person ID executing this event")
    start_timestamp: float = Field(..., ge=0.0, description="Event start in seconds")
    end_timestamp: float = Field(..., ge=0.0, description="Event end in seconds")
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)
    predicted_class: FlagCategory = Field(default=FlagCategory.SUSPICIOUS_MOVEMENT)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    severity: Severity = Field(default=Severity.LOW)
    explanation_text: str = Field(default="")
    evidence_frame_paths: list[str] = Field(default_factory=list)
    annotated_frame_paths: list[str] = Field(default_factory=list)
    model_version: str = Field(default="motion_baseline_v0.1")
    processing_latency_ms: float = Field(default=0.0, ge=0.0)
    notes: Optional[str] = None


# ──────────────────────────────────────────────
# Full analysis result
# ──────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """Complete analysis output for a video."""
    video_id: str
    events: list[FlagEvent] = Field(default_factory=list)
    total_frames_analyzed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0, ge=0.0)
    processing_time_seconds: float = Field(default=0.0, ge=0.0)
    model_version: str = Field(default="motion_baseline_v0.1")
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    frame_results: list[FrameResult] = Field(
        default_factory=list,
        description="Per-frame results (may be truncated in API responses)",
    )

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def suspicious_event_count(self) -> int:
        return sum(1 for e in self.events if e.predicted_class != FlagCategory.NORMAL)
