"""
MulTiCheat — Note Passing Transfer Malpractice Classifier

Requires object transfer pattern across temporal sequence:
Hand A -> Object in shared region -> Hand B.
Rejects simple centroid proximity alone.
"""

from __future__ import annotations

from typing import List, Optional
from backend.analysis.models import FlagCategory, Severity
from backend.behavior.temporal.temporal_window import TemporalStateWindow


class NotePassingResult:
    def __init__(
        self,
        is_malpractice: bool,
        confidence: float,
        severity: Severity,
        supporting_signals: List[str],
        contradicting_signals: List[str],
        explanation: str,
        target_student_id: Optional[int] = None,
    ):
        self.is_malpractice = is_malpractice
        self.confidence = confidence
        self.severity = severity
        self.supporting_signals = supporting_signals
        self.contradicting_signals = contradicting_signals
        self.explanation = explanation
        self.target_student_id = target_student_id


class NotePassingClassifier:

    @staticmethod
    def classify(
        window: TemporalStateWindow,
        target_student_id: Optional[int] = None,
        hand_transfer_detected: bool = False,
    ) -> NotePassingResult:
        supporting: List[str] = []
        contradicting: List[str] = []

        pass_hits = window.count_label_occurrences("note_passing")
        total_frames = window.frame_count

        if total_frames == 0:
            return NotePassingResult(False, 0.0, Severity.LOW, [], ["No frame history"], "No data")

        pass_ratio = pass_hits / total_frames

        if hand_transfer_detected:
            supporting.append("Object trajectory observed moving from Hand A into shared interaction zone to Hand B")
        else:
            contradicting.append("No explicit object transfer trajectory confirmed between hands")

        if pass_hits > 0:
            supporting.append(f"Synchronized wrist proximity cluster detected ({pass_hits} frames)")

        if target_student_id is not None:
            supporting.append(f"Paired target student StudentTrack_{target_student_id:03d}")

        base_score = (0.50 if hand_transfer_detected else 0.20) + (pass_ratio * 0.45)
        confidence = max(0.0, min(1.0, base_score))

        # Centroid distance alone without hand transfer caps confidence at 0.50
        if not hand_transfer_detected and pass_hits > 0:
            contradicting.append("Centroid proximity alone is insufficient for confirmed note passing")
            confidence = min(0.50, confidence)

        is_malpractice = confidence >= 0.70 and hand_transfer_detected

        severity = Severity.CRITICAL if is_malpractice else (Severity.MEDIUM if confidence > 0.4 else Severity.LOW)

        explanation = (
            f"Note passing evaluation between StudentTrack_{window.track_id:03d} and target: "
            f"Confidence {confidence:.0%}. Hand transfer: {hand_transfer_detected}. "
            f"Supporting: {len(supporting)}, Contradicting: {len(contradicting)}."
        )

        return NotePassingResult(
            is_malpractice=is_malpractice,
            confidence=round(confidence, 3),
            severity=severity,
            supporting_signals=supporting,
            contradicting_signals=contradicting,
            explanation=explanation,
            target_student_id=target_student_id,
        )
