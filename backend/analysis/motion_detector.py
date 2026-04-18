"""
MulTiCheat — Motion-Based Baseline Detector

Detects suspicious motion using frame differencing and contour analysis.
No external model weights required — uses only OpenCV built-in operations.

Detection logic:
1. Convert frames to grayscale, apply Gaussian blur
2. Compute absolute difference between consecutive frames
3. Threshold the difference image
4. Find contours in the thresholded image
5. Classify contours by area and position:
   - Large motion regions → suspicious_movement
   - Very large motion → possible object / person movement
   - Small / marginal motion → normal background noise
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from backend.analysis.base_detector import BaseDetector
from backend.analysis.models import BoundingBox, Detection

logger = logging.getLogger(__name__)


class MotionDetector(BaseDetector):
    """Baseline motion detector using frame differencing."""

    def __init__(
        self,
        min_contour_area: int = 500,
        large_contour_area: int = 3000,
        blur_kernel: int = 21,
        diff_threshold: int = 25,
        suspicious_threshold: float = 0.05,
    ):
        """
        Args:
            min_contour_area: Minimum contour area (px²) to consider.
            large_contour_area: Area threshold for "large" motion.
            blur_kernel: Gaussian blur kernel size (must be odd).
            diff_threshold: Binary threshold for difference image.
            suspicious_threshold: Fraction of frame area with motion
                                  to flag as suspicious.
        """
        self.min_contour_area = min_contour_area
        self.large_contour_area = large_contour_area
        self.blur_kernel = blur_kernel
        self.diff_threshold = diff_threshold
        self.suspicious_threshold = suspicious_threshold

    @property
    def name(self) -> str:
        return "motion_baseline"

    @property
    def version(self) -> str:
        return "0.1.0"

    def detect(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        frame_index: int,
        fps: float,
    ) -> list[Detection]:
        """Detect motion by comparing with the previous frame."""
        if prev_frame is None:
            return []

        # Validate inputs
        if frame.shape != prev_frame.shape:
            logger.warning(
                "Frame shape mismatch at index %d: %s vs %s",
                frame_index, frame.shape, prev_frame.shape,
            )
            return []

        try:
            detections = self._analyze_motion(frame, prev_frame, frame_index)
            return detections
        except Exception as exc:
            logger.error("Motion detection failed at frame %d: %s", frame_index, exc)
            return []

    def _analyze_motion(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        frame_index: int,
    ) -> list[Detection]:
        """Core motion analysis logic."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        prev_gray = cv2.GaussianBlur(prev_gray, (self.blur_kernel, self.blur_kernel), 0)

        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, gray)

        # Threshold
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # Dilate to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        total_motion_area = 0
        detections: list[Detection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            total_motion_area += area
            x, y, w, h = cv2.boundingRect(contour)

            # Classify by size
            if area >= self.large_contour_area:
                label = "large_motion"
                confidence = min(1.0, area / (frame_area * 0.1))
            else:
                label = "motion"
                confidence = min(1.0, area / (frame_area * 0.05))

            # Position-based heuristics
            center_y = y + h / 2
            center_x = x + w / 2

            # Motion in the center of the frame is more suspicious
            # (person-region heuristic)
            x_ratio = center_x / frame_w
            y_ratio = center_y / frame_h
            is_central = 0.2 < x_ratio < 0.8 and 0.15 < y_ratio < 0.85

            metadata = {
                "area": int(area),
                "area_ratio": round(area / frame_area, 4),
                "is_central": is_central,
                "region": "center" if is_central else "peripheral",
            }

            detections.append(Detection(
                label=label,
                confidence=round(confidence, 3),
                bbox=BoundingBox(x=x, y=y, w=w, h=h),
                metadata=metadata,
            ))

        # Add an overall motion summary detection
        motion_ratio = total_motion_area / frame_area if frame_area > 0 else 0.0
        if motion_ratio > self.suspicious_threshold:
            detections.append(Detection(
                label="high_overall_motion",
                confidence=round(min(1.0, motion_ratio / 0.15), 3),
                bbox=None,
                metadata={
                    "total_motion_area": int(total_motion_area),
                    "motion_ratio": round(motion_ratio, 4),
                    "contour_count": len(contours),
                },
            ))

        return detections
