"""
Tests for YOLO Pose Detector.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from backend.analysis.pose_detector import PoseDetector, ULTRALYTICS_AVAILABLE
except ImportError:
    ULTRALYTICS_AVAILABLE = False


@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
def test_pose_detector_init():
    detector = PoseDetector(model_version="yolov8n-pose.pt")
    assert detector.name == "YOLO Pose Detector"

@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
def test_pose_detector_empty_frame():
    detector = PoseDetector(model_version="yolov8n-pose.pt")
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame, None, 0, 30.0)
    assert len(detections) == 0

@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
def test_pose_detector_warmup():
    detector = PoseDetector(model_version="yolov8n-pose.pt")
    detector.warmup()  # Should not crash
