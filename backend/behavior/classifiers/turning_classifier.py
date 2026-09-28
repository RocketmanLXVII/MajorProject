"""
MulTiCheat — Turning Around Classifier

Distinguishes brief look backward from sustained body turn and normal posture adjustments.
Separates turning_around from paper_copying and looking_around.
"""

from __future__ import annotations

from typing import List
from backend.analysis.models import Severity
from backend.behavior.temporal.temporal_window import TemporalStateWindow


class TurningResult:
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


class TurningClassifier:

    @staticmethod
    def classify(window: TemporalStateWindow) -> TurningResult:
        supporting: List[str] = []
        contradicting: List[str] = []

        look_hits = window.count_label_occurrences("look_around")
        large_motion_hits = window.count_label_occurrences("large_motion")
        duration = window.duration
        total_frames = window.frame_count

        if total_frames == 0:
            return TurningResult(False, 0.0, Severity.LOW, [], ["No frame history"], "No data")

        turn_ratio = look_hits / total_frames

        if look_hits >= 3:
            supporting.append(f"Significant torso/head rotation detected in {look_hits}/{total_frames} frames")

        if large_motion_hits > 0:
            supporting.append("Significant upper body kinematic displacement observed")

        if duration >= 2.0:
            supporting.append(f"Sustained body turn duration {duration:.1f}s >= 2.0s threshold")
        else:
            contradicting.append(f"Brief turn ({duration:.1f}s) — likely posture adjustment or hearing check")

        confidence = max(0.0, min(1.0, (turn_ratio * 0.5) + (min(large_motion_hits, 4) * 0.1) + (0.2 if duration >= 2.0 else 0.0)))
        is_malpractice = confidence >= 0.65 and duration >= 2.0

        severity = Severity.MEDIUM if is_malpractice else Severity.LOW

        explanation = (
            f"Turning evaluation for StudentTrack_{window.track_id:03d}: "
            f"Confidence {confidence:.0%}, Duration {duration:.1f}s. "
            f"Supporting: {len(supporting)}, Contradicting: {len(contradicting)}."
        )

        return TurningResult(
            is_malpractice=is_malpractice,
            confidence=round(confidence, 3),
            severity=severity,
            supporting_signals=supporting,
            contradicting_signals=contradicting,
            explanation=explanation,
        )
