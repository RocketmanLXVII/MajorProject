"""
MulTiCheat — Perspective-Aware Multi-Scale Spatial Tiling Engine

Solves rear-row detection by dividing wide classroom frames into overlapping
tiles, running detection per tile, mapping coordinates back to global space,
and suppressing cross-tile duplicates using NMS.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import numpy as np

from backend.analysis.models import BoundingBox, Detection

logger = logging.getLogger(__name__)


def generate_spatial_tiles(
    frame: np.ndarray,
    rows: int = 2,
    cols: int = 2,
    overlap: float = 0.20,
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Divide frame into overlapping sub-images.

    Returns:
        List of ((tile_x, tile_y, tile_w, tile_h), tile_crop)
    """
    h, w = frame.shape[:2]
    tile_w = int(w / (cols - (cols - 1) * overlap))
    tile_h = int(h / (rows - (rows - 1) * overlap))

    tiles: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []

    step_x = int(tile_w * (1.0 - overlap))
    step_y = int(tile_h * (1.0 - overlap))

    for r in range(rows):
        for c in range(cols):
            x1 = min(c * step_x, w - tile_w)
            y1 = min(r * step_y, h - tile_h)
            x2 = min(x1 + tile_w, w)
            y2 = min(y1 + tile_h, h)

            x1, y1 = max(0, x1), max(0, y1)
            tile_crop = frame[y1:y2, x1:x2]
            tiles.append(((x1, y1, x2 - x1, y2 - y1), tile_crop))

    return tiles


def map_tile_detection_to_global(
    detection: Detection,
    tile_offset: Tuple[int, int],
) -> Detection:
    """Remap detection box coordinates from tile-relative space to full frame space."""
    if detection.bbox is None:
        return detection

    offset_x, offset_y = tile_offset
    new_bbox = BoundingBox(
        x=detection.bbox.x + offset_x,
        y=detection.bbox.y + offset_y,
        w=detection.bbox.w,
        h=detection.bbox.h,
    )
    return Detection(
        label=detection.label,
        confidence=detection.confidence,
        bbox=new_bbox,
        track_id=detection.track_id,
        metadata={**detection.metadata, "remapped_from_tile": True},
    )


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    y2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - intersection

    return intersection / union if union > 0 else 0.0


def suppress_cross_tile_duplicates(
    detections: List[Detection],
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """Apply Non-Maximum Suppression across overlapping tile detections."""
    if not detections:
        return []

    # Sort detections by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []

    for det in sorted_dets:
        if det.bbox is None:
            kept.append(det)
            continue

        box = (det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h)
        duplicate = False

        for k in kept:
            if k.bbox is None or k.label != det.label:
                continue

            k_box = (k.bbox.x, k.bbox.y, k.bbox.w, k.bbox.h)
            if compute_iou(box, k_box) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept
