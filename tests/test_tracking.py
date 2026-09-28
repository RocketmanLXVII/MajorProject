"""
Unit Tests for BoT-SORT ReID Tracking and Occlusion Handling
"""

from __future__ import annotations

import pytest
from backend.detection.tracker import BoTSORTReIDTracker
from backend.perception.tracking.student_track import StudentTrack


def test_student_track_lifecycle():
    track = StudentTrack(track_id="StudentTrack_001", initial_bbox=[100, 100, 80, 160])
    assert track.track_id == "StudentTrack_001"
    assert track.occluded is False

    # Simulate missing detections
    for _ in range(5):
        track.update_missing()
    assert track.occluded is True

    # Re-identification match
    track.update_detection(bbox=[105, 102, 80, 160], confidence=0.88)
    assert track.occluded is False
    assert len(track.bbox_history) == 2


def test_botsort_tracker_update():
    tracker = BoTSORTReIDTracker()
    detections = [
        {"bbox": [100, 100, 50, 100], "conf": 0.90, "class": "person"},
        {"bbox": [300, 100, 50, 100], "conf": 0.85, "class": "person"},
    ]

    tracks = tracker.update(detections)
    assert len(tracks) == 2
    assert "track_id" in tracks[0]
