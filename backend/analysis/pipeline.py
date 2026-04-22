"""
MulTiCheat — Analysis Pipeline Orchestrator

Coordinates frame sampling, detection, window aggregation,
event classification, and evidence generation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.analysis.base_detector import BaseDetector
from backend.analysis.evidence import save_event_evidence
from backend.analysis.models import (
    AnalysisResult,
    Detection,
    FlagCategory,
    FlagEvent,
    FrameResult,
    Severity,
)
from backend.analysis.motion_detector import MotionDetector
from backend.analysis.pose_detector import PoseDetector
from backend.analysis.yolo_detector import YOLODetector
from backend.config import get_settings
from backend.video_service import find_video_file, get_video_dir, load_metadata_cache

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Pipeline configuration
# ──────────────────────────────────────────────

class PipelineConfig:
    """Knobs for the analysis pipeline."""

    def __init__(
        self,
        sample_interval_sec: float = 0.5,
        max_frames: int = 600,
        window_size: int = 5,
        suspicion_threshold: float = 0.3,
        min_event_frames: int = 2,
        max_evidence_per_event: int = 3,
    ):
        self.sample_interval_sec = sample_interval_sec
        self.max_frames = max_frames
        self.window_size = window_size
        self.suspicion_threshold = suspicion_threshold
        self.min_event_frames = min_event_frames
        self.max_evidence_per_event = max_evidence_per_event


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

class AnalysisPipeline:
    """Orchestrates the full video analysis flow."""

    def __init__(
        self,
        detectors: list[BaseDetector] | None = None,
        config: PipelineConfig | None = None,
    ):
        self.config = config or PipelineConfig()
        
        if detectors is None:
            # Note: order is arbitrary, but it's good to have motion as a fallback
            self.detectors = [
                YOLODetector(model_version="yolov8n.pt", conf_threshold=0.3),
                PoseDetector(),
                MotionDetector()
            ]
        else:
            self.detectors = detectors

        # Log active detectors
        for d in self.detectors:
            logger.info("Detector loaded: %s (v%s, available=%s)", d.name, d.version, d.is_available())

    def run(self, video_id: str) -> AnalysisResult:
        """
        Run the full analysis pipeline for a video.

        Steps:
        1. Load video and metadata
        2. Sample frames
        3. Run detectors on each frame
        4. Compute frame-level scores
        5. Aggregate into event windows
        6. Classify events
        7. Save evidence
        8. Return structured results
        """
        start_time = time.time()
        logger.info("Starting analysis for video %s", video_id)

        # ── 1. Load video ──────────────────────
        video_path = find_video_file(video_id)
        if video_path is None:
            raise ValueError(f"Video '{video_id}' not found.")

        metadata = load_metadata_cache(video_id)
        if metadata is None:
            raise ValueError(f"Metadata for video '{video_id}' not found.")

        fps = metadata.fps if metadata.fps > 0 else 25.0

        # ── 2. Sample frames ───────────────────
        frame_data = self._sample_frames(video_path, fps)
        if not frame_data:
            logger.warning("No frames sampled from video %s", video_id)
            return AnalysisResult(
                video_id=video_id,
                total_frames_analyzed=0,
                total_duration_seconds=metadata.duration_seconds,
                processing_time_seconds=time.time() - start_time,
            )

        logger.info("Sampled %d frames from video %s", len(frame_data), video_id)

        # ── 3-4. Run detectors + score ─────────
        frame_results = self._analyze_frames(frame_data, fps)
        logger.info("Analyzed %d frames, generating events...", len(frame_results))

        # ── 5-6. Aggregate into events ─────────
        events = self._aggregate_events(frame_results, video_id)
        logger.info("Detected %d events in video %s", len(events), video_id)

        # ── 7. Save evidence ───────────────────
        self._save_event_evidence(events, frame_data, frame_results, video_id)

        # ── 8. Build result ────────────────────
        processing_time = time.time() - start_time
        result = AnalysisResult(
            video_id=video_id,
            events=events,
            total_frames_analyzed=len(frame_data),
            total_duration_seconds=metadata.duration_seconds,
            processing_time_seconds=round(processing_time, 3),
            frame_results=frame_results,
        )

        # Cache results to disk
        self._cache_results(video_id, result)

        logger.info(
            "Analysis complete for %s: %d events, %.2fs processing time",
            video_id, len(events), processing_time,
        )
        return result

    # ──────────────────────────────────────────
    # Internal steps
    # ──────────────────────────────────────────

    def _sample_frames(
        self, video_path: Path, fps: float
    ) -> list[tuple[int, float, np.ndarray]]:
        """Sample frames at regular intervals. Returns (index, timestamp, frame)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_interval = max(1, int(fps * self.config.sample_interval_sec))

            frames: list[tuple[int, float, np.ndarray]] = []
            for idx in range(0, total_frames, frame_interval):
                if len(frames) >= self.config.max_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                timestamp = idx / fps if fps > 0 else 0.0
                frames.append((idx, timestamp, frame))

            return frames
        finally:
            cap.release()

    def _analyze_frames(
        self,
        frame_data: list[tuple[int, float, np.ndarray]],
        fps: float,
    ) -> list[FrameResult]:
        """Run all detectors on each frame and compute suspicion scores."""
        results: list[FrameResult] = []
        prev_frame: np.ndarray | None = None

        for frame_index, timestamp, frame in frame_data:
            all_detections: list[Detection] = []

            for detector in self.detectors:
                if not detector.is_available():
                    continue
                try:
                    dets = detector.detect(frame, prev_frame, frame_index, fps)
                    all_detections.extend(dets)
                except Exception as exc:
                    logger.error(
                        "Detector '%s' failed at frame %d: %s",
                        detector.name, frame_index, exc,
                    )

            # Process Multi-Person Geometrical Interactions
            all_detections = self._process_interactions(all_detections)

            # Compute aggregate suspicion score for the whole frame (legacy fallback)
            score = self._compute_frame_score(all_detections)

            results.append(FrameResult(
                frame_index=frame_index,
                timestamp_seconds=round(timestamp, 3),
                detections=all_detections,
                aggregate_score=round(score, 3),
            ))

            prev_frame = frame

        return results

    def _process_interactions(self, detections: list[Detection]) -> list[Detection]:
        """Geometrically maps objects to people and computes multi-person interactions."""
        import math
        people = [d for d in detections if d.label == "person" and d.bbox is not None and d.track_id is not None]
        objects = [d for d in detections if d.label in ("cell phone", "book") and d.bbox is not None]
        
        # 1. Attribute standalone objects to tracks via Centroid Containment or Proximity
        for obj in objects:
            ox, oy, ow, oh = obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h
            obj_cx, obj_cy = ox + ow / 2, oy + oh / 2
            
            for person in people:
                px, py, pw, ph = person.bbox.x, person.bbox.y, person.bbox.w, person.bbox.h
                # Check if object centroid is strictly located inside the person's boundaries
                if px <= obj_cx <= px + pw and py <= obj_cy <= py + ph:
                    obj.track_id = person.track_id
                    break
                    
        # 2. Multi-Person Relational Anomalies (Note Passing & Copying)
        new_interactions: list[Detection] = []
        
        for i, pA in enumerate(people):
            for j, pB in enumerate(people):
                if i >= j: continue
                
                ax, ay, aw, ah = pA.bbox.x, pA.bbox.y, pA.bbox.w, pA.bbox.h
                bx, by, bw, bh = pB.bbox.x, pB.bbox.y, pB.bbox.w, pB.bbox.h
                
                cent_ax, cent_ay = ax + aw / 2, ay + ah / 2
                cent_bx, cent_by = bx + bw / 2, by + bh / 2
                dist = math.hypot(cent_ax - cent_bx, cent_ay - cent_by)
                
                # Note passing: extreme bounding cluster overlapping between distinct identities
                if dist < (aw + bw) * 0.35:
                     new_interactions.append(Detection(
                         label="note_passing", confidence=0.8, track_id=pA.track_id, bbox=pA.bbox
                     ))
                     new_interactions.append(Detection(
                         label="note_passing", confidence=0.8, track_id=pB.track_id, bbox=pB.bbox
                     ))
                     
        # 3. Paper Copying Raycast Analysis
        look_arounds = [d for d in detections if d.label == "look_around" and d.track_id is not None]
        for la in look_arounds:
            yaw = la.metadata.get("yaw_direction")
            if not yaw or not la.bbox: continue
            
            ax, aw = la.bbox.x, la.bbox.w
            acent = ax + aw / 2
            
            nearest_dist = float('inf')
            target_person = None
            
            for pB in people:
                if pB.track_id == la.track_id: continue
                bx, bw = pB.bbox.x, pB.bbox.w
                bcent = bx + bw / 2
                
                if yaw == "right" and bcent > acent:
                    dist = bcent - acent
                    if dist < nearest_dist:
                        nearest_dist = dist
                        target_person = pB
                elif yaw == "left" and bcent < acent:
                    dist = acent - bcent
                    if dist < nearest_dist:
                        nearest_dist = dist
                        target_person = pB
                        
            if target_person and nearest_dist < (aw + target_person.bbox.w) * 2.5:
                new_interactions.append(Detection(
                    label="paper_copying",
                    confidence=la.confidence * 0.9,
                    track_id=la.track_id,
                    bbox=la.bbox
                ))
                     
        detections.extend(new_interactions)
        return detections

    def _compute_frame_score(self, detections: list[Detection]) -> float:
        """Compute a combined suspicion score from all detections."""
        if not detections:
            return 0.0

        # Take the max confidence among "suspicious" detections
        suspicious_scores = []
        for det in detections:
            if det.label in ("cell phone", "book"):
                # Unauthorized objects are highly suspicious
                suspicious_scores.append(det.confidence * 1.5)
            elif det.label in ("look_around", "suspicious_hand_movement"):
                # Behavioral alerts are suspicious
                suspicious_scores.append(det.confidence * 1.2)
            elif det.label in ("large_motion", "high_overall_motion"):
                suspicious_scores.append(det.confidence * 0.8)
            elif det.label == "motion":
                suspicious_scores.append(det.confidence * 0.4)
            else:
                suspicious_scores.append(det.confidence * 0.1)

        # Combine: weighted max + count bonus
        max_score = max(suspicious_scores) if suspicious_scores else 0.0
        count_bonus = min(0.2, len(suspicious_scores) * 0.03)

        return min(1.0, max_score + count_bonus)

    def _aggregate_events(
        self,
        frame_results: list[FrameResult],
        video_id: str,
    ) -> list[FlagEvent]:
        """
        Aggregate consecutive suspicious frames into event spans per track_id
        using an entity-specific sliding window approach.
        """
        events: list[FlagEvent] = []
        min_frames = self.config.min_event_frames
        
        # Track active runs per track_id
        # None key acts as the fallback for global frame-level motion triggers
        active_runs: dict[int | None, list[tuple[FrameResult, list[Detection]]]] = {}

        for fr in frame_results:
            # Group suspicious detections in this frame by track_id
            suspicious_by_track: dict[int | None, list[Detection]] = {}
            global_suspicious = False
            
            for det in fr.detections:
                # Define strictly anomalous classifications
                if det.label in (
                    "cell phone", "book", "look_around", "suspicious_hand_movement",
                    "note_passing", "paper_copying", "large_motion", "high_overall_motion"
                ):
                    suspicious_by_track.setdefault(det.track_id, []).append(det)
                    if getattr(det, 'track_id', None) is None:
                        global_suspicious = True
            
            # Evaluate each track's continuity
            # If a track was active previously but missing from this suspect frame, finalize it.
            for t_id in list(active_runs.keys()):
                if t_id not in suspicious_by_track and not (t_id is None and global_suspicious):
                    run = active_runs.pop(t_id)
                    if len(run) >= min_frames:
                        events.extend(self._create_tracked_events(run, t_id, video_id))

            # Extend runs for actively suspicious tracks
            for t_id, dets in suspicious_by_track.items():
                active_runs.setdefault(t_id, []).append((fr, dets))

        # Close out any remaining tracks spanning through the final frame
        for t_id, run in active_runs.items():
            if len(run) >= min_frames:
                events.extend(self._create_tracked_events(run, t_id, video_id))

        return events

    def _create_tracked_events(
        self,
        run: list[tuple[FrameResult, list[Detection]]],
        track_id: int | None,
        video_id: str,
    ) -> list[FlagEvent]:
        """Create FlagEvents mapping strictly to an identity over a suspicious run."""
        
        # Calculate prolonged non-attentiveness if looks around endlessly
        look_around_count = sum(1 for _, dets in run if any(d.label == "look_around" for d in dets))
        duration = run[-1][0].timestamp_seconds - run[0][0].timestamp_seconds
        
        all_labels = []
        max_score = 0.0
        for _, dets in run:
            for d in dets:
                all_labels.append(d.label)
                if d.confidence > max_score:
                    max_score = d.confidence

        predicted_class = self._classify_event(all_labels, duration, look_around_count)
        severity = self._compute_severity(max_score, duration, len(run), predicted_class)
        explanation = self._generate_explanation(predicted_class, max_score, duration, len(run), track_id)

        return [FlagEvent(
            video_id=video_id,
            track_id=track_id,
            start_timestamp=run[0][0].timestamp_seconds,
            end_timestamp=run[-1][0].timestamp_seconds,
            start_frame=run[0][0].frame_index,
            end_frame=run[-1][0].frame_index,
            predicted_class=predicted_class,
            confidence_score=round(max_score, 3),
            severity=severity,
            explanation_text=explanation,
            model_version="multimodal_v2.0"
        )]

    def _classify_event(
        self, all_labels: list[str], duration: float, look_around_count: int
    ) -> FlagCategory:
        """Classify an event based on geometric tracked patterns."""
        if "cell phone" in all_labels:
            return FlagCategory.PHONE_USE
        if "book" in all_labels:
            return FlagCategory.UNAUTHORIZED_OBJECT
        if "note_passing" in all_labels:
            return FlagCategory.NOTE_PASSING
        if "paper_copying" in all_labels:
            return FlagCategory.PAPER_COPYING
            
        if "look_around" in all_labels:
            if duration >= 5.0 and look_around_count >= 5:
                return FlagCategory.PROLONGED_NON_ATTENTIVE
            return FlagCategory.LOOK_AROUND
            
        if "suspicious_hand_movement" in all_labels:
            return FlagCategory.SUSPICIOUS_HAND_MOVEMENT

        has_large = "large_motion" in all_labels
        if has_large:
            return FlagCategory.SUSPICIOUS_MOVEMENT
        return FlagCategory.NORMAL

    def _compute_severity(
        self, max_score: float, duration: float, frame_count: int, pclass: FlagCategory
    ) -> Severity:
        """Assign severity upgrading explicitly for physical cheating tools."""
        if pclass in (FlagCategory.PHONE_USE, FlagCategory.NOTE_PASSING):
            return Severity.CRITICAL
        if pclass in (FlagCategory.UNAUTHORIZED_OBJECT, FlagCategory.PAPER_COPYING, FlagCategory.PROLONGED_NON_ATTENTIVE):
            return Severity.HIGH
        if pclass in (FlagCategory.LOOK_AROUND, FlagCategory.SUSPICIOUS_HAND_MOVEMENT):
            return Severity.MEDIUM
        return Severity.LOW

    def _generate_explanation(
        self,
        predicted_class: FlagCategory,
        max_score: float,
        duration: float,
        frame_count: int,
        track_id: int | None
    ) -> str:
        """Generate a human-readable explanation for the event."""
        parts = []
        track_str = f"Student [Track ID: {track_id}]" if track_id is not None else "Unidentified Entity"
        parts.append(
            f"Suspicious activity ({predicted_class.value.replace('_', ' ')}) executed by {track_str}."
        )
        parts.append(f"Confidence constraint: {max_score:.0%}.")
        parts.append(f"Duration: {duration:.1f}s continuous across {frame_count} sampled frames.")

        if max_score >= 0.7:
            parts.append("High confidence cheating behavior — recommend manual proctor review.")
        elif max_score >= 0.4:
            parts.append("Moderate confidence — proctor review suggested.")
        else:
            parts.append("Low confidence — may be normal test-taking activity.")

        return " ".join(parts)

    def _save_event_evidence(
        self,
        events: list[FlagEvent],
        frame_data: list[tuple[int, float, np.ndarray]],
        frame_results: list[FrameResult],
        video_id: str,
    ) -> None:
        """Save evidence frames for each event."""
        # Index frame data by frame_index for quick lookup
        frame_lookup: dict[int, np.ndarray] = {
            idx: frame for idx, _, frame in frame_data
        }
        result_lookup: dict[int, FrameResult] = {
            fr.frame_index: fr for fr in frame_results
        }

        for event in events:
            # Collect frames that belong to this event
            evidence_frames: list[tuple[int, float, np.ndarray, list[Detection]]] = []

            for fr in frame_results:
                if event.start_frame <= fr.frame_index <= event.end_frame:
                    frame = frame_lookup.get(fr.frame_index)
                    if frame is not None:
                        evidence_frames.append((
                            fr.frame_index,
                            fr.timestamp_seconds,
                            frame,
                            fr.detections,
                        ))

            # Limit evidence per event
            if len(evidence_frames) > self.config.max_evidence_per_event:
                # Keep first, middle, and last
                indices = [
                    0,
                    len(evidence_frames) // 2,
                    len(evidence_frames) - 1,
                ]
                evidence_frames = [evidence_frames[i] for i in indices]

            if evidence_frames:
                # Build score lookup for rich annotations
                score_lookup = {
                    fr.frame_index: fr.aggregate_score for fr in frame_results
                }
                evidence_paths, annotated_paths = save_event_evidence(
                    evidence_frames, video_id, event.event_id,
                    event=event,
                    aggregate_scores=score_lookup,
                )
                event.evidence_frame_paths = evidence_paths
                event.annotated_frame_paths = annotated_paths

    # ──────────────────────────────────────────
    # Result caching
    # ──────────────────────────────────────────

    def _cache_results(self, video_id: str, result: AnalysisResult) -> None:
        """Save analysis results as JSON alongside the video."""
        video_dir = get_video_dir(video_id)
        if not video_dir.exists():
            logger.warning("Video directory not found for caching: %s", video_id)
            return

        cache_path = video_dir / "analysis_result.json"
        annotations_path = video_dir / "annotations.json"
        try:
            # Main cache without frame_results
            data = result.model_dump(exclude={"frame_results"})
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                
            # Annotations (frame_results) cache
            frame_data = [fr.model_dump() for fr in result.frame_results]
            with open(annotations_path, "w", encoding="utf-8") as f:
                json.dump(frame_data, f, indent=2, default=str)
                
            logger.info("Cached analysis results to %s and %s", cache_path, annotations_path)
        except Exception as exc:
            logger.error("Failed to cache analysis results: %s", exc)


def load_cached_results(video_id: str) -> Optional[AnalysisResult]:
    """Load cached analysis results from disk."""
    video_dir = get_video_dir(video_id)
    cache_path = video_dir / "analysis_result.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AnalysisResult(**data)
    except Exception as exc:
        logger.warning("Failed to load cached results for %s: %s", video_id, exc)
        return None

def load_annotations(video_id: str) -> Optional[list[dict]]:
    """Load cached raw frame annotations from disk."""
    video_dir = get_video_dir(video_id)
    annotations_path = video_dir / "annotations.json"
    if not annotations_path.exists():
        return None
    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load annotations for %s: %s", video_id, exc)
        return None
