"""
MulTiCheat — Student Track Entity

Maintains state, bounding box history, ReID feature embeddings,
occlusion status, and body pose keypoints for individual students.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class OcclusionState(str, Enum):
    VISIBLE = "VISIBLE"
    PARTIALLY_OCCLUDED = "PARTIALLY_OCCLUDED"
    FULLY_OCCLUDED = "FULLY_OCCLUDED"
    LOST = "LOST"


class StudentTrack:
    """Persistent student track representation."""

    def __init__(
        self,
        track_id: Any,
        initial_bbox: Tuple[int, int, int, int],
        seat_id: Optional[str] = None,
        confidence: float = 0.9,
    ):
        self.track_id = track_id
        if isinstance(track_id, int):
            self.anonymous_label = f"StudentTrack_{track_id:03d}"
            self.seat_id = seat_id or f"Seat_{track_id:02d}"
        else:
            self.anonymous_label = str(track_id)
            self.seat_id = seat_id or f"Seat_{track_id}"

        self.bbox = initial_bbox  # x, y, w, h
        self.confidence = confidence

        self.occlusion_state: OcclusionState = OcclusionState.VISIBLE
        self.visibility_score: float = 1.0
        self.missed_frames: int = 0
        self.total_tracked_frames: int = 1
        self.last_seen_frame: int = 0

        self.bbox_history: List[Tuple[int, int, int, int]] = [initial_bbox]
        self.pose_keypoints: List[Tuple[float, float]] = []
        self.reid_features: Optional[List[float]] = None

    @property
    def occluded(self) -> bool:
        return self.occlusion_state in (OcclusionState.PARTIALLY_OCCLUDED, OcclusionState.FULLY_OCCLUDED, OcclusionState.LOST)

    def update_detection(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        frame_index: int = 0,
        pose_kpts: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.bbox_history.append(bbox)
        self.missed_frames = 0
        self.total_tracked_frames += 1
        self.last_seen_frame = frame_index
        self.occlusion_state = OcclusionState.VISIBLE
        self.visibility_score = 1.0
        if pose_kpts:
            self.pose_keypoints = pose_kpts

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        frame_index: int = 0,
        pose_kpts: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.update_detection(bbox, confidence, frame_index, pose_kpts)

    def update_missing(self) -> None:
        self.mark_missed()

    def mark_missed(self) -> None:
        self.missed_frames += 1
        self.visibility_score = max(0.0, 1.0 - (self.missed_frames * 0.15))
        if self.missed_frames >= 10:
            self.occlusion_state = OcclusionState.LOST
        elif self.missed_frames >= 3:
            self.occlusion_state = OcclusionState.FULLY_OCCLUDED
        else:
            self.occlusion_state = OcclusionState.PARTIALLY_OCCLUDED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "anonymous_label": self.anonymous_label,
            "seat_id": self.seat_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "occlusion_state": self.occlusion_state.value,
            "visibility_score": self.visibility_score,
            "total_tracked_frames": self.total_tracked_frames,
        }
