"""
MulTiCheat Pro — 3D Head Pose & Face Mesh Estimator

Uses MediaPipe Face Mesh (468 landmarks) + OpenCV solvePnP with 6 canonical
3D facial points to compute exact 3D head pose angles (Yaw, Pitch, Roll).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)


class HeadPoseEstimator:
    """Computes 3D head pose angles (Yaw, Pitch, Roll) using MediaPipe Face Mesh & solvePnP."""

    # 3D canonical facial model points (Nose tip, Chin, Left eye corner, Right eye corner, Left mouth, Right mouth)
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),             # Nose tip (index 1)
            (0.0, -330.0, -65.0),        # Chin (index 152)
            (-225.0, 170.0, -135.0),     # Left eye left corner (index 263)
            (225.0, 170.0, -135.0),      # Right eye right corner (index 33)
            (-150.0, -150.0, -125.0),    # Left mouth corner (index 291)
            (150.0, -150.0, -125.0),     # Right mouth corner (index 61)
        ],
        dtype=np.float64,
    )

    # Corresponding MediaPipe Face Mesh landmark indices
    FACIAL_LANDMARK_INDICES = [1, 152, 263, 33, 291, 61]

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.face_mesh = None
        self._available = False

        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=confidence_threshold,
                )
                self._available = True
                logger.info("MediaPipe Face Mesh initialized for 3D Head Pose solvePnP")
            except Exception as exc:
                logger.warning("MediaPipe Face Mesh fallback: %s", exc)

    def estimate_head_pose(
        self, face_crop: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Estimate 3D head pose (Yaw, Pitch, Roll in degrees) from a face crop.

        Returns:
            Dict {"yaw": deg, "pitch": deg, "roll": deg} or None on failure.
        """
        if not self._available or self.face_mesh is None or face_crop is None:
            return None

        h, w = face_crop.shape[:2]
        if h < 20 or w < 20:
            return None

        try:
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_crop)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            image_points = []

            for idx in self.FACIAL_LANDMARK_INDICES:
                lm = landmarks[idx]
                image_points.append((lm.x * w, lm.y * h))

            image_points = np.array(image_points, dtype=np.float64)

            # Camera intrinsics matrix (approximation)
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1))  # Assuming zero distortion

            success, rot_vec, trans_vec = cv2.solvePnP(
                self.MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                return None

            # Convert rotation vector to Euler angles
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch, yaw, roll = angles[0], angles[1], angles[2]

            return {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
            }
        except Exception as exc:
            logger.debug("Head pose solvePnP failed: %s", exc)
            return None
