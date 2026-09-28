"""
MulTiCheat Pro — 3D Eye Gaze Estimation Module

Computes 3D gaze vector (dx, dy, dz) from 3D head pose parameters (yaw, pitch, roll)
and MediaPipe landmark geometry. Maps gaze ray to desk surface polygons.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.utils.geometry import intersect_gaze_with_desks


class GazeEstimator:
    """3D Eye Gaze Estimator using Head Pose Vector Transformation & Desk Ray Casting."""

    def __init__(self, desk_height_m: float = 0.75):
        self.desk_height_m = desk_height_m

    def estimate_gaze_vector(
        self,
        head_yaw: float,
        head_pitch: float,
        head_roll: float,
        eye_center: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        Convert head pose angles (in degrees) to 3D unit gaze ray vector [dx, dy, dz].
        Yaw: rotation around Y-axis (left/right, positive = right)
        Pitch: rotation around X-axis (up/down, positive = up)
        Roll: rotation around Z-axis (tilt)
        """
        yaw_rad = math.radians(head_yaw)
        pitch_rad = math.radians(head_pitch)
        roll_rad = math.radians(head_roll)

        # Direction vector in camera 3D space
        # Forward axis = Z, Right axis = X, Down axis = Y
        dx = math.sin(yaw_rad) * math.cos(pitch_rad)
        dy = math.sin(pitch_rad)
        dz = math.cos(yaw_rad) * math.cos(pitch_rad)

        gaze_dir = np.array([dx, dy, dz], dtype=np.float32)
        norm = np.linalg.norm(gaze_dir)
        if norm > 1e-6:
            gaze_dir /= norm
        else:
            gaze_dir = np.array([0.0, -0.5, 1.0], dtype=np.float32)

        return gaze_dir

    def calculate_desk_intersection(
        self,
        gaze_vector: np.ndarray,
        head_3d_position: Tuple[float, float, float],
        desks: List[Dict[str, Any]],
        max_distance: float = 3.5,
    ) -> Optional[Dict[str, Any]]:
        """Find desk target intersected by student's gaze ray."""
        eye_center_3d = np.array(head_3d_position, dtype=np.float32)
        return intersect_gaze_with_desks(
            eye_center_3d=eye_center_3d,
            gaze_vector=gaze_vector,
            desks=desks,
            max_distance=max_distance,
            desk_height_m=self.desk_height_m,
        )

    def is_gaze_divergent(
        self,
        gaze_vector: np.ndarray,
        assigned_desk_id: Optional[str] = None,
        intersected_desk: Optional[Dict[str, Any]] = None,
        yaw_threshold: float = 20.0,
        pitch_threshold: float = -30.0,
        head_yaw: float = 0.0,
        head_pitch: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate if student gaze is divergent (looking at neighbor's desk, under desk, or excessive side looking).
        """
        is_looking_away = abs(head_yaw) > yaw_threshold or head_pitch < pitch_threshold
        target_desk_id = intersected_desk.get("id") if intersected_desk else None

        looking_at_other_desk = (
            target_desk_id is not None
            and assigned_desk_id is not None
            and target_desk_id != assigned_desk_id
        )

        looking_under_desk = head_pitch < -30.0

        return {
            "gaze_vector": gaze_vector.tolist() if isinstance(gaze_vector, np.ndarray) else gaze_vector,
            "is_looking_away": is_looking_away,
            "looking_at_other_desk": looking_at_other_desk,
            "looking_under_desk": looking_under_desk,
            "target_desk_id": target_desk_id,
            "severity": 0.8 if looking_at_other_desk or looking_under_desk else (0.5 if is_looking_away else 0.0),
        }
