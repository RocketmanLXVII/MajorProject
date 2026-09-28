"""
Unit Tests for Behavior Rule Engine, 3D Gaze Estimator, & Temporal Classifiers
"""

from __future__ import annotations

import pytest
import numpy as np

from backend.detection.gaze_estimator import GazeEstimator
from backend.detection.interaction_engine import InteractionEngine
from backend.detection.behavior_rules import BehaviorRuleEngine
from backend.detection.temporal_classifier import TemporalPoseClassifier


def test_gaze_estimator_desk_intersection():
    estimator = GazeEstimator(desk_height_m=0.75)
    # Looking down and forward (negative Y in camera frame)
    gaze_vec = estimator.estimate_gaze_vector(head_yaw=0.0, head_pitch=-30.0, head_roll=0.0)

    desks = [
        {"id": "DESK-1", "bbox": [0.0, 0.0, 3.0, 3.0], "polygon": [(0, 0), (3, 0), (3, 3), (0, 3)]}
    ]
    intersected = estimator.calculate_desk_intersection(
        gaze_vector=gaze_vec,
        head_3d_position=(1.0, 2.0, 0.5),  # Height = 2.0m
        desks=desks,
    )
    assert intersected is not None
    assert intersected["id"] == "DESK-1"


def test_interaction_engine_phone():
    engine = InteractionEngine()
    hand_pts = [(100.0, 200.0), (105.0, 205.0)]
    detected_objs = [{"cls": "phone", "bbox": [90.0, 190.0, 30.0, 50.0], "conf": 0.92}]

    held = engine.check_hand_object_overlap(hand_pts, detected_objs)
    assert len(held) == 1
    assert held[0]["class_name"] == "phone"


def test_behavior_rule_engine_peeking():
    rules = BehaviorRuleEngine()
    track_data = {
        "track_id": "STU-01",
        "head_yaw": 45.0,
        "head_pitch": -10.0,
        "temporal_window": {"peeking_duration_sec": 2.0},
    }
    gaze_info = {"looking_at_other_desk": True}

    events = rules.evaluate_track_malpractice(track_data=track_data, gaze_info=gaze_info)
    assert len(events) >= 1
    assert any(e["type"] == "PEEKING_NEIGHBOR" for e in events)


def test_temporal_pose_classifier():
    classifier = TemporalPoseClassifier(window_size=30)
    # 30 frames of dummy keypoints (34 floats per frame)
    sequence = [[0.0] * 34 for _ in range(30)]

    result = classifier.classify_sequence(sequence)
    assert isinstance(result, dict)
    assert "NORMAL" in result
