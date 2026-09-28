"""
MulTiCheat Pro — Detection API Pydantic Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HeadPoseSchema(BaseModel):
    yaw: float
    pitch: float
    roll: float


class GazeVectorSchema(BaseModel):
    dx: float
    dy: float
    dz: float
    confidence: float = 1.0


class FrameDetectionSchema(BaseModel):
    track_id: int
    student_id: Optional[str] = None
    seat_number: Optional[int] = None
    bbox: List[int]  # [x, y, w, h]
    pose_keypoints: Optional[Dict[str, Any]] = None
    head_pose: Optional[HeadPoseSchema] = None
    gaze_vector: Optional[GazeVectorSchema] = None
    detected_objects: Optional[List[Dict[str, Any]]] = None
    behavior_scores: Optional[Dict[str, float]] = None
    severity: str = "LOW"
