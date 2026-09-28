"""
MulTiCheat Pro ORM Models Package
"""

from backend.models.base import Base
from backend.models.student import Student
from backend.models.exam import Exam, ExamStudent
from backend.models.video import Video
from backend.models.detection import FrameDetection
from backend.models.event import MalpracticeEvent
from backend.models.evidence import EvidenceAsset

# Backwards-compatible aliases
VideoRecord = Video
EventRecord = MalpracticeEvent
EvidenceFileRecord = EvidenceAsset
StudentTrackRecord = FrameDetection
AnalysisJobRecord = Video
ReviewerFeedbackRecord = MalpracticeEvent

__all__ = [
    "Base",
    "Student",
    "Exam",
    "ExamStudent",
    "Video",
    "FrameDetection",
    "MalpracticeEvent",
    "EvidenceAsset",
    "VideoRecord",
    "EventRecord",
    "EvidenceFileRecord",
    "StudentTrackRecord",
    "AnalysisJobRecord",
    "ReviewerFeedbackRecord",
]
