"""
MulTiCheat — Phone Usage Malpractice Classifier

Requires multi-signal evidence: phone detection + hand proximity + head gaze + duration >= 2.0s.
Generates explicit supporting and contradicting evidence signals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from backend.analysis.models import FlagCategory, Severity
from backend.behavior.temporal.temporal_window import TemporalStateWindow


class PhoneUsageResult:
    def __init__(
        self,
        is_malpractice: bool,
        confidence: float,
        severity: Severity,
        supporting_signals: List[str],
        contradicting_signals: List[str],
        explanation: str,
    ):
        self.is_malpractice = is_malpractice
        self.confidence = confidence
        self.severity = severity
        self.supporting_signals = supporting_signals
        self.contradicting_signals = contradicting_signals
        self.explanation = explanation


class PhoneClassifier:

    @staticmethod
    def classify(window: TemporalStateWindow) -> PhoneUsageResult:
        supporting: List[str] = []
        contradicting: List[str] = []

        phone_hits = window.count_label_occurrences("cell phone") + window.count_label_occurrences("phone") + window.count_label_occurrences("smartphone_screen")
        hand_hits = window.count_label_occurrences("suspicious_hand_movement")
        head_down_hits = sum(1 for s in window.history if s.head_yaw_offset > 0.40)

        duration = window.duration
        total_frames = window.frame_count

        if total_frames == 0:
            return PhoneUsageResult(False, 0.0, Severity.LOW, [], ["No frame history"], "No data")

        phone_ratio = phone_hits / total_frames

        if phone_hits > 0:
            supporting.append(f"Phone object detected in {phone_hits}/{total_frames} window frames ({phone_ratio:.0%})")
        else:
            contradicting.append("No explicit phone object bounding box detected in window")

        if hand_hits > 0:
            supporting.append(f"Hand-to-phone interaction or reaching motion detected ({hand_hits} frames)")

        if head_down_hits > 0:
            supporting.append(f"Head orientation directed toward lap/phone area ({head_down_hits} frames)")

        if duration >= 2.0:
            supporting.append(f"Interaction duration of {duration:.1f}s exceeds min 2.0s threshold")
        else:
            contradicting.append(f"Interaction duration ({duration:.1f}s) is below 2.0s threshold")

        # Score calculation
        base_score = min(0.95, (phone_ratio * 0.5) + (min(hand_hits, 5) * 0.08) + (min(head_down_hits, 5) * 0.06))
        if duration >= 2.0:
            base_score += 0.15

        confidence = max(0.0, min(1.0, base_score))

        is_malpractice = confidence >= 0.65 and len(supporting) >= 2

        severity = Severity.CRITICAL if is_malpractice else (Severity.HIGH if confidence > 0.4 else Severity.LOW)

        explanation = (
            f"Phone usage evaluated for StudentTrack_{window.track_id:03d}: "
            f"Confidence {confidence:.0%}. "
            f"Supporting signals: {len(supporting)}, Contradicting: {len(contradicting)}."
        )

        return PhoneUsageResult(
            is_malpractice=is_malpractice,
            confidence=round(confidence, 3),
            severity=severity,
            supporting_signals=supporting,
            contradicting_signals=contradicting,
            explanation=explanation,
        )
