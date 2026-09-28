"""
MulTiCheat Pro — Frame-Level Detections ORM Model
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String
from backend.models.base import Base


class FrameDetection(Base):
    __tablename__ = "detections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp_utc = Column("time", DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, index=True)
    frame_index = Column(Integer, nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    student_id = Column(String(36), ForeignKey("students.id"), nullable=True)
    bbox = Column(JSON, nullable=False)  # [x, y, w, h]
    pose_keypoints = Column(JSON, nullable=True)  # 133 or 17 keypoints
    head_pose = Column(JSON, nullable=True)  # {yaw, pitch, roll}
    gaze_vector = Column(JSON, nullable=True)  # [dx, dy, dz]
    detected_objects = Column(JSON, nullable=True)
    behavior_scores = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_detections_video_frame", "video_id", "frame_index"),
        Index("idx_detections_track", "video_id", "track_id"),
    )
