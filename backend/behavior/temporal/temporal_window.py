"""
MulTiCheat — Temporal State Window & Feature Engine

Maintains a rolling temporal sequence window (t - Delta_t ... t) per student track
to evaluate kinematic trajectories, gaze direction, hand movements, and object presence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.analysis.models import Detection


class StudentFrameState:
    """Snapshot of a student's state at a single frame index."""

    def __init__(
        self,
        frame_index: int,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        pose_keypoints: Dict[str, Dict[str, float]],
        detections: List[Detection],
        head_yaw_offset: float = 0.0,
        head_yaw_dir: str = "center",
        hand_extension_ratio: float = 0.0,
    ):
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.bbox = bbox
        self.pose_keypoints = pose_keypoints
        self.detections = detections
        self.head_yaw_offset = head_yaw_offset
        self.head_yaw_dir = head_yaw_dir
        self.hand_extension_ratio = hand_extension_ratio

        # Extracted labels in this frame
        self.labels = {d.label for d in detections}


class TemporalStateWindow:
    """
    Rolling state sequence window for a single StudentTrack.
    """

    def __init__(self, track_id: int, max_window_seconds: float = 8.0):
        self.track_id = track_id
        self.max_window_seconds = max_window_seconds
        self.history: List[StudentFrameState] = []

    def add_frame_state(self, state: StudentFrameState) -> None:
        self.history.append(state)
        # Prune old states beyond window duration
        if self.history:
            latest_time = self.history[-1].timestamp
            self.history = [
                s for s in self.history if (latest_time - s.timestamp) <= self.max_window_seconds
            ]

    @property
    def duration(self) -> float:
        if not self.history or len(self.history) < 2:
            return 0.0
        return self.history[-1].timestamp - self.history[0].timestamp

    @property
    def frame_count(self) -> int:
        return len(self.history)

    def count_label_occurrences(self, label: str) -> int:
        return sum(1 for s in self.history if label in s.labels)

    def get_sustained_head_yaw_ratio(self, target_direction: str) -> float:
        if not self.history:
            return 0.0
        match_count = sum(1 for s in self.history if s.head_yaw_dir == target_direction and s.head_yaw_offset > 0.45)
        return match_count / len(self.history)

    def count_head_turn_glances(self) -> int:
        """Count distinct directional glances in the window."""
        glances = 0
        in_glance = False
        for s in self.history:
            if s.head_yaw_dir in ("left", "right") and s.head_yaw_offset > 0.45:
                if not in_glance:
                    glances += 1
                    in_glance = True
            else:
                in_glance = False
        return glances
