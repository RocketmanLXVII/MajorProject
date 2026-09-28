"""
MulTiCheat — Paper Copying Malpractice Classifier

Models Student A head/gaze direction toward Student B's desk/paper region,
angular alignment, repeated glances, and temporal persistence.
One sideways glance = NOT CHEATING.
Repeated sustained glances = LIKELY_PAPER_COPYING.
"""

from __future__ import annotations

from typing import List, Optional
from backend.analysis.models import FlagCategory, Severity
from backend.behavior.temporal.temporal_window import TemporalStateWindow


class PaperCopyingResult:
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


class PaperCopyingClassifier:

    @staticmethod
    def classify(
        window: TemporalStateWindow,
        target_student_id: Optional[int] = None,
        target_direction: str = "right",
    ) -> PaperCopyingResult:
        supporting: List[str] = []
        contradicting: List[str] = []

        copy_hits = window.count_label_occurrences("paper_copying")
        look_hits = window.count_label_occurrences("look_around")
        glances = window.count_head_turn_glances()
        duration = window.duration
        total_frames = window.frame_count

        if total_frames == 0:
            return PaperCopyingResult(False, 0.0, Severity.LOW, [], ["No frame history"], "No data")

        yaw_ratio = window.get_sustained_head_yaw_ratio(target_direction)

        if glances == 1:
            contradicting.append("Single brief sideways glance detected — classified as normal test-taking activity")
        elif glances >= 3:
            supporting.append(f"Repeated directional glances toward neighboring desk ({glances} distinct glances)")

        if yaw_ratio >= 0.40:
            supporting.append(f"Sustained head orientation aligned with target desk in {yaw_ratio:.0%} of frames")
        elif yaw_ratio < 0.20:
            contradicting.append(f"Head orientation alignment low ({yaw_ratio:.0%})")

        if copy_hits > 0:
            supporting.append(f"Raycast geometric intersection with target desk confirmed ({copy_hits} frames)")

        if target_student_id is not None:
            supporting.append(f"Target student StudentTrack_{target_student_id:03d} located in raycast trajectory")

        if duration >= 2.5:
            supporting.append(f"Sustained behavior duration {duration:.1f}s exceeds 2.5s threshold")
        else:
            contradicting.append(f"Short duration {duration:.1f}s — insufficient temporal persistence")

        # Score calculation
        base_score = (yaw_ratio * 0.45) + (min(glances, 6) * 0.08) + ((copy_hits / max(1, total_frames)) * 0.35)
        if duration >= 2.5 and glances >= 3:
            base_score += 0.15

        confidence = max(0.0, min(1.0, base_score))

        # Single glance prevents high confidence accusation
        if glances <= 1:
            confidence = min(0.35, confidence)

        is_malpractice = confidence >= 0.65 and glances >= 2

        severity = Severity.HIGH if is_malpractice else (Severity.MEDIUM if confidence > 0.4 else Severity.LOW)

        explanation = (
            f"Paper copying evaluated for StudentTrack_{window.track_id:03d}: "
            f"Confidence {confidence:.0%}, Glances: {glances}. "
            f"Supporting: {len(supporting)}, Contradicting: {len(contradicting)}."
        )

        return PaperCopyingResult(
            is_malpractice=is_malpractice,
            confidence=round(confidence, 3),
            severity=severity,
            supporting_signals=supporting,
            contradicting_signals=contradicting,
            explanation=explanation,
            target_student_id=target_student_id,
        )
