"""
MulTiCheat — Classroom Calibration & Seating Geometry Engine

Manages classroom boundaries, seating grids, depth perspective scaling,
and live monitoring coverage metrics (e.g. 97% Monitoring Coverage).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SeatRegion(BaseModel):
    seat_id: str
    row_index: int  # 1 (front) to N (rear)
    col_index: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    expected_student_size: float = 1.0  # Depth scale factor


class CoverageMetrics(BaseModel):
    expected_students: int = 50
    detected_students: int = 0
    tracked_students: int = 0
    visibly_monitored_students: int = 0
    low_visibility_count: int = 0
    occluded_count: int = 0
    coverage_percentage: float = 0.0


class ClassroomCalibration:
    """Classroom calibration model for perspective-aware threshold adjustment."""

    def __init__(
        self,
        total_rows: int = 5,
        expected_students: int = 50,
        seat_grid: Optional[List[SeatRegion]] = None,
    ):
        self.total_rows = total_rows
        self.expected_students = expected_students
        self.seat_grid = seat_grid or []

    def get_depth_scale_for_y(self, y_coordinate: float, frame_height: float) -> float:
        """
        Estimate depth scale from y coordinate (top = far/rear, bottom = near/front).
        Returns scale factor from ~0.4 (rear row) to ~1.0 (front row).
        """
        if frame_height <= 0:
            return 1.0
        normalized_y = max(0.0, min(1.0, y_coordinate / frame_height))
        # Top of frame (y -> 0) is rear (scale 0.4), bottom (y -> 1.0) is front (scale 1.0)
        scale = 0.35 + (normalized_y * 0.65)
        return round(scale, 3)

    def get_adjusted_conf_threshold(
        self,
        base_threshold: float,
        y_coordinate: float,
        frame_height: float,
    ) -> float:
        """Dynamically lower confidence threshold for rear-row students to improve recall."""
        scale = self.get_depth_scale_for_y(y_coordinate, frame_height)
        # Rear-row students (scale ~0.4) get lower threshold bounds
        adjusted = base_threshold * (0.65 + 0.35 * scale)
        return max(0.12, round(adjusted, 3))

    def compute_coverage(self, student_tracks: List[Any]) -> CoverageMetrics:
        """Compute live coverage metrics across all active student tracks."""
        detected = len(student_tracks)
        tracked = sum(1 for s in student_tracks if getattr(s, 'visibility_score', 1.0) > 0.3)
        visibly_monitored = sum(1 for s in student_tracks if getattr(s, 'occlusion_state', 'VISIBLE') == 'VISIBLE')
        low_vis = sum(1 for s in student_tracks if getattr(s, 'visibility_score', 1.0) <= 0.5)
        occluded = sum(1 for s in student_tracks if getattr(s, 'occlusion_state', 'VISIBLE') in ('PARTIALLY_OCCLUDED', 'HEAVILY_OCCLUDED'))

        cov_pct = (tracked / self.expected_students * 100.0) if self.expected_students > 0 else 0.0

        return CoverageMetrics(
            expected_students=self.expected_students,
            detected_students=detected,
            tracked_students=tracked,
            visibly_monitored_students=visibly_monitored,
            low_visibility_count=low_vis,
            occluded_count=occluded,
            coverage_percentage=round(min(100.0, cov_pct), 1),
        )
