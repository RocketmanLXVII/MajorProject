"""
MulTiCheat Pro — Behavior Rule Engine

Maps 10 distinct malpractice types specified in the Section 1.2 Master Mandate:
1. PEEKING_NEIGHBOR
2. PHONE_USAGE
3. PAPER_COPYING
4. CHIT_USAGE
5. NOTE_PASSING
6. TURNING_AROUND
7. TALKING_SIGNALING
8. STAND_LEAN
9. OBJECT_EXCHANGE
10. UNAUTHORIZED_MATERIAL
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from backend.utils.geometry import compute_behavior_confidence


class BehaviorRuleEngine:
    """Evaluates rule-based criteria for all 10 malpractice categories with high precision."""

    def __init__(self):
        self.fps = 25.0

    def evaluate_track_malpractice(
        self,
        track_data: Dict[str, Any],
        gaze_info: Optional[Dict[str, Any]] = None,
        interaction_info: Optional[Dict[str, Any]] = None,
        nearby_pairs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        events = []

        track_id = track_data.get("track_id")
        head_yaw = track_data.get("head_yaw", 0.0)
        head_pitch = track_data.get("head_pitch", 0.0)
        head_roll = track_data.get("head_roll", 0.0)
        held_objects = track_data.get("held_objects", [])
        desk_objects = track_data.get("desk_objects", [])
        body_lean_angle = track_data.get("body_lean_angle", 0.0)
        is_standing = track_data.get("is_standing", False)
        temporal_window = track_data.get("temporal_window", {})

        peeking_duration = temporal_window.get("peeking_duration_sec", 0.0)
        phone_duration = temporal_window.get("phone_duration_sec", 0.0)
        turning_duration = temporal_window.get("turning_duration_sec", 0.0)

        looking_other_desk = gaze_info.get("looking_at_other_desk", False) if gaze_info else False
        is_looking_away = gaze_info.get("is_looking_away", False) if gaze_info else False

        # Rule 1: PEEKING_NEIGHBOR (Looking left/right or at neighbor desk)
        if (abs(head_yaw) > 18.0 or looking_other_desk or is_looking_away) and abs(head_yaw) <= 50.0:
            conf = compute_behavior_confidence(
                detector_confidence=0.88,
                temporal_consistency=min(1.0, max(0.6, peeking_duration / 2.0)),
                geometric_plausibility=0.92 if looking_other_desk else 0.82,
            )
            events.append({
                "type": "PEEKING_NEIGHBOR",
                "confidence": conf,
                "description": f"Student peeking / looking laterally (yaw: {head_yaw:.1f}°)",
                "severity": "MEDIUM",
            })

        # Rule 2: PHONE_USAGE (Requires explicit Phone object detected in hand/desk space)
        def _is_phone(o: Dict[str, Any]) -> bool:
            name = (o.get("class_name") or o.get("label") or "").lower()
            return name in ["phone", "cell phone", "smartphone"]

        phone_held = any(_is_phone(o) for o in held_objects)
        phone_on_desk = any(_is_phone(o) for o in desk_objects)
        hand_under_desk = track_data.get("hand_under_desk", False)

        if phone_held or (phone_on_desk and head_pitch < -15.0) or (phone_duration >= 2.0 and (phone_held or phone_on_desk)):
            conf = compute_behavior_confidence(
                detector_confidence=0.95 if phone_held else 0.85,
                temporal_consistency=min(1.0, max(0.6, phone_duration / 2.0)),
                geometric_plausibility=0.95,
            )
            events.append({
                "type": "PHONE_USAGE",
                "confidence": conf,
                "description": "Mobile phone detected in hand or desk area",
                "severity": "HIGH",
            })

        # Rule 3: PAPER_COPYING (Sustained gaze at neighbor desk + lateral body lean)
        if (looking_other_desk or abs(head_yaw) > 35.0) and peeking_duration >= 1.5:
            conf = compute_behavior_confidence(
                detector_confidence=0.89,
                temporal_consistency=min(1.0, peeking_duration / 3.0),
                geometric_plausibility=0.94,
            )
            events.append({
                "type": "PAPER_COPYING",
                "confidence": conf,
                "description": f"Sustained copying motion towards neighbor paper (yaw: {head_yaw:.1f}°)",
                "severity": "CRITICAL",
            })

        # Rule 4: CHIT_USAGE (Paper chit or small cheat sheet detected)
        def _is_chit(o: Dict[str, Any]) -> bool:
            name = (o.get("class_name") or o.get("label") or "").lower()
            return name in ["chit", "cheat_sheet", "small_note", "paper_chit", "note"]

        chit_held = any(_is_chit(o) for o in held_objects)
        chit_on_desk = any(_is_chit(o) for o in desk_objects)

        if chit_held or chit_on_desk or (head_pitch < -28.0 and hand_under_desk):
            conf = compute_behavior_confidence(
                detector_confidence=0.92 if (chit_held or chit_on_desk) else 0.82,
                temporal_consistency=0.85,
                geometric_plausibility=0.90,
            )
            events.append({
                "type": "CHIT_USAGE",
                "confidence": conf,
                "description": "Paper chit or compact cheat material detected",
                "severity": "HIGH",
            })

        # Rule 5: NOTE_PASSING (Physical hand reach contact between neighboring tracks)
        if nearby_pairs:
            for pair in nearby_pairs:
                if pair.get("student_1_id") == track_id or pair.get("student_2_id") == track_id:
                    if pair.get("possible_note_passing") or pair.get("hands_touching"):
                        conf = compute_behavior_confidence(
                            detector_confidence=0.90 if pair.get("hands_touching") else 0.80,
                            temporal_consistency=0.85,
                            geometric_plausibility=0.92,
                        )
                        events.append({
                            "type": "NOTE_PASSING",
                            "confidence": conf,
                            "description": "Hand reaching / note passing contact detected with adjacent student",
                            "severity": "CRITICAL",
                        })

        # Rule 6: TURNING_AROUND (Head rotated backwards > 50 deg)
        if abs(head_yaw) > 50.0 or turning_duration >= 1.0:
            conf = compute_behavior_confidence(
                detector_confidence=0.94,
                temporal_consistency=min(1.0, max(0.5, turning_duration / 2.0)),
                geometric_plausibility=0.92,
            )
            events.append({
                "type": "TURNING_AROUND",
                "confidence": conf,
                "description": f"Student turned around backwards (yaw: {head_yaw:.1f}°)",
                "severity": "HIGH",
            })

        # Rule 7: TALKING_SIGNALING (Mouth motion + head tilt/roll)
        is_talking = track_data.get("is_talking", False)
        if is_talking and (abs(head_yaw) > 20.0 or abs(head_roll) > 15.0):
            conf = compute_behavior_confidence(
                detector_confidence=0.82,
                temporal_consistency=0.78,
                geometric_plausibility=0.85,
            )
            events.append({
                "type": "TALKING_SIGNALING",
                "confidence": conf,
                "description": "Talking or signaling gestures detected",
                "severity": "MEDIUM",
            })

        # Rule 8: STAND_LEAN (Standing up or leaning across desk)
        if is_standing or abs(body_lean_angle) > 35.0:
            conf = compute_behavior_confidence(
                detector_confidence=0.91,
                temporal_consistency=0.85,
                geometric_plausibility=0.95,
            )
            events.append({
                "type": "STAND_LEAN",
                "confidence": conf,
                "description": "Student standing up or leaning across desk boundary",
                "severity": "MEDIUM",
            })

        # Rule 9: OBJECT_EXCHANGE (Calculator/eraser/paper exchange)
        if nearby_pairs and any(p.get("hands_touching") for p in nearby_pairs):
            def _is_exchangeable(o: Dict[str, Any]) -> bool:
                name = (o.get("class_name") or o.get("label") or "").lower()
                return name in ["calculator", "eraser", "paper", "book"]

            if any(_is_exchangeable(o) for o in held_objects):
                conf = compute_behavior_confidence(
                    detector_confidence=0.86,
                    temporal_consistency=0.82,
                    geometric_plausibility=0.90,
                )
                events.append({
                    "type": "OBJECT_EXCHANGE",
                    "confidence": conf,
                    "description": "Unauthorized material/object exchange between students",
                    "severity": "HIGH",
                })

        # Rule 10: UNAUTHORIZED_MATERIAL (Books, notes on desk)
        def _is_unauthorized(o: Dict[str, Any]) -> bool:
            name = (o.get("class_name") or o.get("label") or "").lower()
            return name in ["book", "notebook", "cheat_sheet", "tablet", "laptop"]

        unauthorized = [o for o in desk_objects if _is_unauthorized(o)]
        if unauthorized:
            mat_name = unauthorized[0].get("class_name") or unauthorized[0].get("label") or "material"
            conf = compute_behavior_confidence(
                detector_confidence=0.90,
                temporal_consistency=0.90,
                geometric_plausibility=0.95,
            )
            events.append({
                "type": "UNAUTHORIZED_MATERIAL",
                "confidence": conf,
                "description": f"Unauthorized material ({mat_name}) detected on desk",
                "severity": "HIGH",
            })

        return events
