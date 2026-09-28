"""
MulTiCheat Pro — Real-Time WebSocket Message Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class WSDetectionMessage(BaseModel):
    type: str = "detection_update"
    video_id: str
    timestamp: str
    frame_index: int
    detections: List[Dict[str, Any]]


class WSEventMessage(BaseModel):
    type: str = "event_flagged"
    event: Dict[str, Any]


class WSProgressMessage(BaseModel):
    type: str = "analysis_progress"
    video_id: str
    progress_percent: int
    fps_processed: float
    eta_seconds: float
