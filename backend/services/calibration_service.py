"""
MulTiCheat Pro — Camera & Desk Calibration Service

Computes homography transform matrix H from 4 clicked floor points,
pixels-per-meter scaling factors, and desk region polygon mappings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationService:
    """Manages camera calibration, homography computation, and desk mapping."""

    @staticmethod
    def compute_homography(
        src_points: List[Tuple[float, float]],  # 4 image points [(x1,y1), (x2,y2), ...]
        real_world_width_m: float = 10.0,
        real_world_height_m: float = 10.0,
    ) -> Optional[np.ndarray]:
        """Compute 3x3 Homography matrix H using cv2.getPerspectiveTransform."""
        if len(src_points) != 4:
            logger.warning("Perspective calibration requires exactly 4 points")
            return None

        src = np.array(src_points, dtype=np.float32)
        dst = np.array([
            [0.0, 0.0],
            [real_world_width_m, 0.0],
            [real_world_width_m, real_world_height_m],
            [0.0, real_world_height_m],
        ], dtype=np.float32)

        try:
            H = cv2.getPerspectiveTransform(src, dst)
            logger.info("Computed 3x3 perspective homography matrix H")
            return H
        except Exception as exc:
            logger.error("Failed to compute homography matrix: %s", exc)
            return None

    @staticmethod
    def format_room_layout(
        homography_matrix: Optional[np.ndarray],
        pixels_per_meter: float = 42.5,
        camera_height_m: float = 2.8,
        desks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Format room layout configuration object for storage."""
        h_mat = homography_matrix.tolist() if homography_matrix is not None else None
        return {
            "homography_matrix": h_mat,
            "pixels_per_meter": pixels_per_meter,
            "camera_height_m": camera_height_m,
            "desks": desks or [],
            "forbidden_zones": [],
        }
