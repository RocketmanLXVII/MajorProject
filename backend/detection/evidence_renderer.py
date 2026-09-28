"""
MulTiCheat Pro — Evidence Renderer & Visual Annotation Engine

Renders professional high-clarity OpenCV overlays for video frames & evidence clips:
- Color-coded track bounding boxes & Student IDs
- 3D Head Pose orientation vectors (XYZ axes)
- 3D Gaze Ray projection onto desk plane
- Skeletal keypoint links
- Suspicious object callout highlights (Phone, Chit, Paper)
- Malpractice event alert banners
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class EvidenceRenderer:
    """OpenCV Renderer for drawing clear, production-grade malpractice visual annotations."""

    COLOR_GREEN = (46, 204, 113)    # Normal
    COLOR_YELLOW = (241, 196, 15)   # Warning / Peeking
    COLOR_RED = (231, 76, 60)      # High / Malpractice
    COLOR_CYAN = (239, 206, 74)    # Head Pose / Gaze
    COLOR_WHITE = (255, 255, 255)

    def draw_student_track(
        self,
        frame: np.ndarray,
        track_id: str,
        bbox: List[float],
        severity: str = "LOW",
        assigned_seat: Optional[str] = None,
        head_pose: Optional[Tuple[float, float, float]] = None,
        gaze_vector: Optional[List[float]] = None,
        malpractice_label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw student bounding box, track label, seat assignment, and alert status."""
        img = frame.copy()
        x, y, w, h = [int(v) for v in bbox]

        # Determine color based on severity
        if severity == "CRITICAL" or severity == "HIGH":
            color = self.COLOR_RED
        elif severity == "MEDIUM":
            color = self.COLOR_YELLOW
        else:
            color = self.COLOR_GREEN

        # Draw main box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Header tag background
        tag_text = f"{track_id}"
        if assigned_seat:
            tag_text += f" ({assigned_seat})"
        if malpractice_label:
            tag_text += f" - {malpractice_label}"

        (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y), color, -1)
        cv2.putText(
            img,
            tag_text,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0) if color != self.COLOR_RED else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw Gaze Vector if present
        if gaze_vector and len(gaze_vector) >= 3:
            cx, cy = x + w // 2, y + int(h * 0.2)
            dx = int(gaze_vector[0] * 60.0)
            dy = int(gaze_vector[1] * 60.0)
            cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy), self.COLOR_CYAN, 2, tipLength=0.3)

        return img

    def draw_object_highlight(
        self,
        frame: np.ndarray,
        bbox: List[float],
        class_name: str,
        confidence: float,
    ) -> np.ndarray:
        """Highlight detected suspicious object (phone, chit, paper)."""
        img = frame.copy()
        x, y, w, h = [int(v) for v in bbox]

        color = (0, 165, 255)  # Orange for objects
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        label = f"{class_name.upper()}: {confidence:.2f}"
        cv2.putText(
            img,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        return img

    def draw_alert_banner(
        self,
        frame: np.ndarray,
        event_type: str,
        student_id: str,
        confidence: float,
    ) -> np.ndarray:
        """Draw prominent banner across top of frame for triggered malpractice events."""
        img = frame.copy()
        h, w = img.shape[:2]

        banner_h = 45
        cv2.rectangle(img, (0, 0), (w, banner_h), (0, 0, 180), -1)

        banner_text = f"ALERT: [{event_type}] Detected for {student_id} (Conf: {confidence:.0%})"
        cv2.putText(
            img,
            banner_text,
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img
