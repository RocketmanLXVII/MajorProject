"""
MulTiCheat — Student-Centric Crop Inference Engine

Generates high-resolution sub-crops around tracked student regions (head, hands, desk)
to run specialized object and paper detection for tiny or occluded objects.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

from backend.analysis.models import BoundingBox, Detection
from backend.perception.tracking.student_track import StudentTrack

logger = logging.getLogger(__name__)


def extract_student_crop(
    frame: np.ndarray,
    student: StudentTrack,
    expand_ratio: float = 0.25,
    min_size: int = 224,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """
    Extract a cropped sub-image for a student with contextual padding.

    Returns:
        ((crop_x, crop_y, crop_w, crop_h), crop_image)
    """
    h_frame, w_frame = frame.shape[:2]
    x, y, w, h = student.bbox

    pad_w = int(w * expand_ratio)
    pad_h = int(h * expand_ratio)

    crop_x1 = max(0, x - pad_w)
    crop_y1 = max(0, y - pad_h)
    crop_x2 = min(w_frame, x + w + pad_w)
    crop_y2 = min(h_frame, y + h + pad_h)

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w < 10 or crop_h < 10:
        return (0, 0, w_frame, h_frame), frame

    crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    return (crop_x1, crop_y1, crop_w, crop_h), crop_img


def map_crop_detection_to_student(
    detection: Detection,
    crop_offset: Tuple[int, int],
    student_track_id: int,
) -> Detection:
    """Map detection found inside a student crop back to global coordinates and tag track_id."""
    if detection.bbox is None:
        return detection

    offset_x, offset_y = crop_offset
    global_bbox = BoundingBox(
        x=detection.bbox.x + offset_x,
        y=detection.bbox.y + offset_y,
        w=detection.bbox.w,
        h=detection.bbox.h,
    )
    return Detection(
        label=detection.label,
        confidence=detection.confidence,
        bbox=global_bbox,
        track_id=student_track_id,
        metadata={**detection.metadata, "source_crop": "student_crop"},
    )
