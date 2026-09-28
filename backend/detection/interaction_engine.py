"""
MulTiCheat Pro — Hand-Object & Student Interaction Engine

Maps hand keypoints (from pose estimation) to object bounding boxes (phones, chits, papers)
and detects proximity interactions between adjacent student tracks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class InteractionEngine:
    """Detects physical and spatial interactions between hands, objects, and neighboring students."""

    def __init__(self, proximity_threshold_px: float = 120.0, hand_object_dist_px: float = 65.0):
        self.proximity_threshold_px = proximity_threshold_px
        self.hand_object_dist_px = hand_object_dist_px

    def check_hand_object_overlap(
        self,
        hand_keypoints: List[Tuple[float, float]],  # Wrist, fingertips
        detected_objects: List[Dict[str, Any]],     # [{cls: 'phone', bbox: [x,y,w,h], conf: 0.8}, ...]
    ) -> List[Dict[str, Any]]:
        """Identify which detected suspicious objects are currently held by hands."""
        held_objects = []
        for obj in detected_objects:
            obj_bbox = obj.get("bbox")
            if isinstance(obj_bbox, dict):
                bx, by, bw, bh = obj_bbox.get("x", 0), obj_bbox.get("y", 0), obj_bbox.get("w", 0), obj_bbox.get("h", 0)
            elif isinstance(obj_bbox, (list, tuple)) and len(obj_bbox) >= 4:
                bx, by, bw, bh = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            else:
                bx, by, bw, bh = 0, 0, 0, 0

            obj_center = (bx + bw / 2.0, by + bh / 2.0)

            min_dist = float("inf")
            for hand_pt in hand_keypoints:
                if hand_pt[0] > 0 and hand_pt[1] > 0:
                    dist = math.hypot(hand_pt[0] - obj_center[0], hand_pt[1] - obj_center[1])
                    if dist < min_dist:
                        min_dist = dist

            if min_dist <= self.hand_object_dist_px:
                label = obj.get("label") or obj.get("class_name") or obj.get("cls")
                held_objects.append({
                    "class_name": label,
                    "label": label,
                    "confidence": obj.get("confidence") or obj.get("conf", 0.5),
                    "bbox": obj_bbox,
                    "hand_distance_px": round(min_dist, 2),
                })
        return held_objects

    def get_student_desk_objects(
        self,
        student_bbox: List[int] | Tuple[int, int, int, int],
        detected_objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter global frame objects strictly to those within or adjacent to this student's desk region."""
        tx, ty, tw, th = int(student_bbox[0]), int(student_bbox[1]), int(student_bbox[2]), int(student_bbox[3])
        min_x = tx - 0.3 * tw
        max_x = tx + 1.3 * tw
        min_y = ty - 0.1 * th
        max_y = ty + 1.4 * th

        desk_objs = []
        for obj in detected_objects:
            obj_bbox = obj.get("bbox")
            if isinstance(obj_bbox, dict):
                bx, by, bw, bh = obj_bbox.get("x", 0), obj_bbox.get("y", 0), obj_bbox.get("w", 0), obj_bbox.get("h", 0)
            elif isinstance(obj_bbox, (list, tuple)) and len(obj_bbox) >= 4:
                bx, by, bw, bh = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            else:
                continue

            cx, cy = bx + bw / 2.0, by + bh / 2.0
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                label = obj.get("label") or obj.get("class_name") or obj.get("cls")
                desk_objs.append({
                    "class_name": label,
                    "label": label,
                    "confidence": obj.get("confidence") or obj.get("conf", 0.5),
                    "bbox": [bx, by, bw, bh],
                })
        return desk_objs

    def detect_student_proximity_pairs(
        self,
        student_tracks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect pairs of students who are physically leaning towards each other or exchanging notes.
        """
        pairs = []
        num_students = len(student_tracks)

        for i in range(num_students):
            for j in range(i + 1, num_students):
                s1 = student_tracks[i]
                s2 = student_tracks[j]

                box1 = s1.get("bbox", [0, 0, 0, 0])
                box2 = s2.get("bbox", [0, 0, 0, 0])

                c1 = (box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0)
                c2 = (box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0)

                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

                if dist <= self.proximity_threshold_px:
                    # Check hands extension between them
                    hands1 = s1.get("hand_points", [])
                    hands2 = s2.get("hand_points", [])

                    min_hand_dist = float("inf")
                    for h1 in hands1:
                        for h2 in hands2:
                            if h1[0] > 0 and h2[0] > 0:
                                h_dist = math.hypot(h1[0] - h2[0], h1[1] - h2[1])
                                if h_dist < min_hand_dist:
                                    min_hand_dist = h_dist

                    hands_touching = min_hand_dist < 50.0

                    pairs.append({
                        "student_1_id": s1.get("track_id"),
                        "student_2_id": s2.get("track_id"),
                        "distance_px": round(dist, 2),
                        "hands_touching": hands_touching,
                        "possible_note_passing": hands_touching or dist < 80.0,
                    })

        return pairs

    def map_hand_to_under_desk(
        self,
        hand_keypoints: List[Tuple[float, float]],
        desk_bbox: Tuple[float, float, float, float],
    ) -> bool:
        """Check if hand keypoints are located below the desk surface bbox boundary."""
        if not desk_bbox or len(desk_bbox) < 4:
            return False

        dx, dy, dw, dh = desk_bbox
        desk_bottom_y = dy + dh

        for h in hand_keypoints:
            if h[0] > 0 and h[1] > 0:
                # Hand x within desk horizontal span and y lower than desk line
                if dx <= h[0] <= dx + dw and h[1] >= dy + (dh * 0.7):
                    return True
        return False
