"""
MulTiCheat — Cheat Chit / Unauthorized Note Classifier

Detects presence of small unauthorized paper notes, chits, or printed references.
Distinguishes official question/answer sheets from unauthorized material.
"""

from __future__ import annotations

from typing import List, Optional
from backend.analysis.models import FlagCategory, Severity
from backend.behavior.temporal.temporal_window import TemporalStateWindow


class ChitResult:
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


class ChitClassifier:

    @staticmethod
    def classify(window: TemporalStateWindow) -> ChitResult:
        supporting: List[str] = []
        contradicting: List[str] = []

        chit_hits = (
            window.count_label_occurrences("cheat_chit")
            + window.count_label_occurrences("small_note")
            + window.count_label_occurrences("book")
        )
        hand_hits = window.count_label_occurrences("suspicious_hand_movement")
        duration = window.duration
        total_frames = window.frame_count

        if total_frames == 0:
            return ChitResult(False, 0.0, Severity.LOW, [], ["No frame history"], "No data")

        chit_ratio = chit_hits / total_frames

        if chit_hits > 0:
            supporting.append(f"Unauthorized note/chit detected in {chit_hits}/{total_frames} window frames ({chit_ratio:.0%})")
        else:
            contradicting.append("No explicit chit/note object bounding box detected")

        if hand_hits > 0:
            supporting.append("Hand interaction with paper/chit object detected")

        if duration >= 1.5:
            supporting.append(f"Persistence duration {duration:.1f}s")
        else:
            contradicting.append("Short duration")

        confidence = max(0.0, min(1.0, (chit_ratio * 0.6) + (min(hand_hits, 4) * 0.10)))
        is_malpractice = confidence >= 0.60 and chit_hits > 0

        severity = Severity.HIGH if is_malpractice else (Severity.MEDIUM if confidence > 0.35 else Severity.LOW)

        explanation = (
            f"Cheat chit evaluation for StudentTrack_{window.track_id:03d}: "
            f"Confidence {confidence:.0%}. "
            f"Supporting: {len(supporting)}, Contradicting: {len(contradicting)}."
        )

        return ChitResult(
            is_malpractice=is_malpractice,
            confidence=round(confidence, 3),
            severity=severity,
            supporting_signals=supporting,
            contradicting_signals=contradicting,
            explanation=explanation,
        )
