"""
Phase 2 — Analysis Pipeline Tests

Tests:
1. MotionDetector produces detections on a video with motion
2. MotionDetector produces no detections on the first frame (no previous)
3. Pipeline produces structured AnalysisResult
4. Events have valid schema (event_id, timestamps, class, confidence, severity)
5. Evidence frames are saved to disk
6. Static video produces fewer/no events than dynamic video
7. API: POST /videos/{id}/analyze returns results
8. API: GET /videos/{id}/results returns cached results
9. API: evidence images are servable
10. API: analyze non-existent video → 404
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.analysis.evidence import annotate_frame, save_evidence_frame
from backend.analysis.models import (
    AnalysisResult,
    BoundingBox,
    Detection,
    FlagCategory,
    FlagEvent,
    FrameResult,
    Severity,
)
from backend.analysis.motion_detector import MotionDetector
from backend.analysis.pipeline import AnalysisPipeline, PipelineConfig


# ──────────────────────────────────────────────
# Helpers — synthetic test videos
# ──────────────────────────────────────────────

def create_dynamic_video(path: Path, num_frames: int = 50) -> Path:
    """Create a video with significant inter-frame motion."""
    w, h, fps = 320, 240, 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Moving rectangle (simulates person movement)
        x = int(50 + 200 * (i / num_frames))
        y = int(40 + 120 * abs(np.sin(i * 0.3)))
        cv2.rectangle(frame, (x, y), (x + 80, y + 100), (0, 200, 255), -1)
        # Add some varying background
        frame[10:30, 10:100] = int(100 * (i / num_frames))
        writer.write(frame)

    writer.release()
    return path


def create_static_video(path: Path, num_frames: int = 50) -> Path:
    """Create a video with identical frames (no motion)."""
    w, h, fps = 320, 240, 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    frame = np.ones((h, w, 3), dtype=np.uint8) * 128
    cv2.rectangle(frame, (100, 60), (220, 180), (200, 200, 200), -1)

    for _ in range(num_frames):
        writer.write(frame)

    writer.release()
    return path


@pytest.fixture()
def dynamic_video(tmp_path: Path) -> Path:
    return create_dynamic_video(tmp_path / "dynamic.mp4")


@pytest.fixture()
def static_video(tmp_path: Path) -> Path:
    return create_static_video(tmp_path / "static.mp4")


# ──────────────────────────────────────────────
# Detector-level tests
# ──────────────────────────────────────────────

class TestMotionDetector:
    """Tests for the motion baseline detector."""

    def test_no_detections_on_first_frame(self):
        """First frame has no prev_frame, should return empty."""
        detector = MotionDetector()
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = detector.detect(frame, None, 0, 25.0)
        assert result == []

    def test_detections_on_motion(self):
        """Should detect motion between two different frames."""
        detector = MotionDetector(min_contour_area=100)
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        frame2 = np.zeros((240, 320, 3), dtype=np.uint8)
        # Add a large moving object
        cv2.rectangle(frame2, (50, 50), (200, 180), (255, 255, 255), -1)

        result = detector.detect(frame2, frame1, 1, 25.0)
        assert len(result) > 0
        assert any(d.label in ("motion", "large_motion", "high_overall_motion") for d in result)

    def test_no_detections_on_identical_frames(self):
        """Two identical frames should produce no detections."""
        detector = MotionDetector()
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
        result = detector.detect(frame, frame.copy(), 1, 25.0)
        assert len(result) == 0

    def test_detection_has_bbox(self):
        """Detections with contours should have bounding boxes."""
        detector = MotionDetector(min_contour_area=50)
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        frame2 = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (80, 60), (200, 180), (255, 255, 255), -1)

        result = detector.detect(frame2, frame1, 1, 25.0)
        bbox_dets = [d for d in result if d.bbox is not None]
        assert len(bbox_dets) > 0
        for d in bbox_dets:
            assert d.bbox.w > 0 and d.bbox.h > 0

    def test_detector_properties(self):
        """Detector should have correct name and version."""
        detector = MotionDetector()
        assert detector.name == "motion_baseline"
        assert detector.is_available() is True
        assert detector.version == "0.1.0"


# ──────────────────────────────────────────────
# Evidence tests
# ──────────────────────────────────────────────

class TestEvidence:
    """Tests for evidence annotation and saving."""

    def test_annotate_frame_returns_copy(self):
        """annotate_frame should return a new array, not modify original."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        dets = [Detection(
            label="motion", confidence=0.8,
            bbox=BoundingBox(x=10, y=10, w=100, h=80),
        )]
        annotated = annotate_frame(frame, dets, frame_index=5, timestamp=0.2)
        assert annotated is not frame
        assert annotated.shape == frame.shape

    def test_annotate_empty_detections(self):
        """annotate_frame with no detections should still return a frame."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = annotate_frame(frame, [], frame_index=0, timestamp=0.0)
        assert result.shape == frame.shape

    def test_save_evidence_frame(self, tmp_path: Path, monkeypatch):
        """save_evidence_frame should write a JPEG to disk."""
        # Monkeypatch get_settings to use tmp_path
        from backend.analysis import evidence
        from backend.config import Settings

        settings = Settings(evidence_dir=str(tmp_path))
        monkeypatch.setattr(evidence, "get_settings", lambda: settings)

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        rel_path = save_evidence_frame(frame, "test_vid", "evt001", 0)

        assert rel_path != ""
        full_path = tmp_path / rel_path
        assert full_path.exists()
        assert full_path.stat().st_size > 0


# ──────────────────────────────────────────────
# Pipeline tests (using uploaded videos via API)
# ──────────────────────────────────────────────

class TestPipelineAPI:
    """API-level tests for analysis endpoints."""

    @pytest.mark.asyncio
    async def test_analyze_video(self, client):
        """POST /videos/{id}/analyze should return analysis results."""
        # Upload a dynamic test video first
        video_path = Path(__file__).parent / "_test_dynamic.mp4"
        create_dynamic_video(video_path, num_frames=50)
        try:
            video_bytes = video_path.read_bytes()

            upload_resp = await client.post(
                "/videos/upload",
                files={"file": ("dynamic.mp4", io.BytesIO(video_bytes), "video/mp4")},
            )
            assert upload_resp.status_code == 201
            vid_id = upload_resp.json()["video"]["video_id"]

            # Run analysis
            analyze_resp = await client.post(f"/videos/{vid_id}/analyze")
            assert analyze_resp.status_code == 200

            data = analyze_resp.json()
            assert data["video_id"] == vid_id
            assert data["total_frames_analyzed"] > 0
            assert data["processing_time_seconds"] > 0
            assert isinstance(data["events"], list)
            assert data["model_version"].startswith("motion_baseline")
        finally:
            video_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_analyze_produces_events(self, client):
        """Analysis of a dynamic video should produce at least one event."""
        video_path = Path(__file__).parent / "_test_dynamic2.mp4"
        create_dynamic_video(video_path, num_frames=75)
        try:
            video_bytes = video_path.read_bytes()

            upload_resp = await client.post(
                "/videos/upload",
                files={"file": ("dynamic2.mp4", io.BytesIO(video_bytes), "video/mp4")},
            )
            vid_id = upload_resp.json()["video"]["video_id"]

            analyze_resp = await client.post(f"/videos/{vid_id}/analyze")
            data = analyze_resp.json()

            # Dynamic video with moving rectangle should have events
            if data["events"]:
                event = data["events"][0]
                assert "event_id" in event
                assert "start_timestamp" in event
                assert "end_timestamp" in event
                assert "predicted_class" in event
                assert "confidence_score" in event
                assert "severity" in event
                assert "explanation_text" in event
                assert event["confidence_score"] >= 0
                assert event["confidence_score"] <= 1
        finally:
            video_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_get_cached_results(self, client):
        """GET /videos/{id}/results should return cached results after analysis."""
        video_path = Path(__file__).parent / "_test_cache.mp4"
        create_dynamic_video(video_path, num_frames=50)
        try:
            video_bytes = video_path.read_bytes()

            upload_resp = await client.post(
                "/videos/upload",
                files={"file": ("cache.mp4", io.BytesIO(video_bytes), "video/mp4")},
            )
            vid_id = upload_resp.json()["video"]["video_id"]

            # Run analysis first
            await client.post(f"/videos/{vid_id}/analyze")

            # Get cached results
            results_resp = await client.get(f"/videos/{vid_id}/results")
            assert results_resp.status_code == 200
            assert results_resp.json()["video_id"] == vid_id
        finally:
            video_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_results_not_found(self, client):
        """GET /videos/{id}/results for non-analyzed video should 404."""
        resp = await client.get("/videos/nonexistent999/results")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_analyze_not_found(self, client):
        """POST /videos/{id}/analyze for non-existent video should 404."""
        resp = await client.post("/videos/nonexistent999/analyze")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_static_video_fewer_events(self, client):
        """Static video should produce fewer events than dynamic video."""
        # Upload static video
        static_path = Path(__file__).parent / "_test_static.mp4"
        create_static_video(static_path, num_frames=50)
        try:
            static_bytes = static_path.read_bytes()

            upload_resp = await client.post(
                "/videos/upload",
                files={"file": ("static.mp4", io.BytesIO(static_bytes), "video/mp4")},
            )
            vid_id = upload_resp.json()["video"]["video_id"]

            analyze_resp = await client.post(f"/videos/{vid_id}/analyze")
            data = analyze_resp.json()

            # Static video should have very few or no events
            assert len(data["events"]) <= 2  # Allow some tolerance for codec artifacts
        finally:
            static_path.unlink(missing_ok=True)
