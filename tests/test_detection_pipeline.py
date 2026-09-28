"""
Unit & Integration Tests for Detection Pipeline (SAHI Tiling & Cross-Tile NMS)
"""

from __future__ import annotations

import numpy as np
import pytest
from backend.perception.detection.tiling import (
    generate_spatial_tiles,
    suppress_cross_tile_duplicates,
)
from backend.analysis.models import BoundingBox, Detection


def test_image_tiler_grid():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tiles = generate_spatial_tiles(img, rows=2, cols=2, overlap=0.2)

    assert len(tiles) == 4
    for (x, y, w, h), crop in tiles:
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        assert x >= 0 and y >= 0


def test_cross_tile_nms():
    # Two overlapping boxes for same student from adjacent tiles
    det1 = Detection(
        label="person",
        confidence=0.85,
        bbox=BoundingBox(x=100, y=100, w=50, h=100),
    )
    det2 = Detection(
        label="person",
        confidence=0.82,
        bbox=BoundingBox(x=102, y=102, w=50, h=100),
    )

    merged = suppress_cross_tile_duplicates([det1, det2], iou_threshold=0.5)

    assert len(merged) == 1
    assert merged[0].confidence == 0.85
