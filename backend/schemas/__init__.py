"""
MulTiCheat Pro Schemas Package
"""

from backend.schemas.video import VideoCreate, VideoResponse
from backend.schemas.detection import FrameDetectionSchema, HeadPoseSchema, GazeVectorSchema
from backend.schemas.event import EventResponse, EventUpdate
from backend.schemas.student import StudentCreate, StudentResponse, SeatAssignmentRequest
from backend.schemas.ws import WSDetectionMessage, WSEventMessage, WSProgressMessage

__all__ = [
    "VideoCreate",
    "VideoResponse",
    "FrameDetectionSchema",
    "HeadPoseSchema",
    "GazeVectorSchema",
    "EventResponse",
    "EventUpdate",
    "StudentCreate",
    "StudentResponse",
    "SeatAssignmentRequest",
    "WSDetectionMessage",
    "WSEventMessage",
    "WSProgressMessage",
]
