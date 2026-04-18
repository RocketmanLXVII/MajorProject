"""
Tests for YOLO Object Detector.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from backend.analysis.yolo_detector import YOLODetector, ULTRALYTICS_AVAILABLE
except ImportError:
    ULTRALYTICS_AVAILABLE = False


@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
def test_yolo_detector_init():
    detector = YOLODetector(model_version="yolov8n.pt")
    # Assuming the weights automatically download or exist
    assert detector.name == "YOLO Object Detector"
    # Even if download fails, it should gracefully fall back to unavailable
    # but let's test assuming network works.

@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
def test_yolo_detector_inference():
    detector = YOLODetector(model_version="yolov8n.pt")
    if detector.is_available():
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Add a fake phone rect to see if it hallucinates? Normally it shouldn't find anything in black
        detections = detector.detect(frame, None, 0, 30.0)
        assert len(detections) == 0
