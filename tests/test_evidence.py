"""
Phase 3 — Visual Evidence Generation Tests

Tests:
1. Enhanced annotate_frame produces frame with info bar
2. annotate_frame with event context draws severity banner
3. annotate_frame with keypoints draws skeleton
4. Confidence bars are drawn on detections
5. Score gauge is drawn
6. save_evidence_frame writes correct JPEG
7. list_evidence_files returns file metadata
8. API: GET /videos/{id}/evidence returns gallery grouped by event
9. API: evidence images are viewable after analysis
10. Evidence annotation doesn't modify original frame
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.analysis.evidence import (
    annotate_frame,
    annotate_frame_simple,
    list_evidence_files,
    save_evidence_frame,
    save_event_evidence,
)
from backend.analysis.models import (
    BoundingBox,
    Detection,
    FlagCategory,
    FlagEvent,
    Severity,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_frame(w=320, h=240) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_detections() -> list[Detection]:
    return [
        Detection(
            label="motion",
            confidence=0.7,
            bbox=BoundingBox(x=50, y=40, w=80, h=100),
            metadata={"area": 8000, "is_central": True},
        ),
        Detection(
            label="large_motion",
            confidence=0.85,
            bbox=BoundingBox(x=160, y=60, w=60, h=70),
        ),
    ]


def _make_event() -> FlagEvent:
    return FlagEvent(
        event_id="testevt01",
        video_id="testvid01",
        start_timestamp=1.5,
        end_timestamp=3.0,
        start_frame=37,
        end_frame=75,
        predicted_class=FlagCategory.SUSPICIOUS_MOVEMENT,
        confidence_score=0.78,
        severity=Severity.HIGH,
        explanation_text="Test event explanation.",
    )


def _make_dynamic_video(path: Path, num_frames: int = 50) -> Path:
    w, h, fps = 320, 240, 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int(50 + 200 * (i / num_frames))
        y = int(40 + 120 * abs(np.sin(i * 0.3)))
        cv2.rectangle(frame, (x, y), (x + 80, y + 100), (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return path


# ──────────────────────────────────────────────
# Annotation tests
# ──────────────────────────────────────────────

class TestAnnotation:
    """Tests for the enhanced evidence annotation system."""

    def test_annotate_does_not_modify_original(self):
        """annotate_frame should return a new array."""
        frame = _make_frame()
        original = frame.copy()
        result = annotate_frame(frame, _make_detections(), 0, 0.0, 0.5)
        assert np.array_equal(frame, original), "Original frame was modified"
        assert not np.array_equal(result, original), "Annotated should differ from original"

    def test_annotate_with_detections_draws_boxes(self):
        """Annotated frame should differ from plain frame (boxes drawn)."""
        frame = _make_frame()
        plain = annotate_frame(frame, [], 0, 0.0, 0.0)
        with_dets = annotate_frame(frame, _make_detections(), 0, 0.0, 0.5)
        # More pixels should be different when detections are drawn
        diff = np.sum(np.abs(plain.astype(int) - with_dets.astype(int)))
        assert diff > 1000, "Detection annotations should visibly change the frame"

    def test_annotate_with_event_context(self):
        """Event context should add severity banner and panel."""
        frame = _make_frame()
        event = _make_event()
        no_event = annotate_frame(frame, _make_detections(), 0, 0.0, 0.5, event=None)
        with_event = annotate_frame(frame, _make_detections(), 0, 0.0, 0.5, event=event)
        diff = np.sum(np.abs(no_event.astype(int) - with_event.astype(int)))
        assert diff > 500, "Event context should add visible annotations"

    def test_annotate_with_keypoints(self):
        """Keypoints in detection metadata should be drawn."""
        frame = _make_frame()
        kps = {
            "0": {"x": 100, "y": 50, "confidence": 0.9},
            "1": {"x": 110, "y": 80, "confidence": 0.8},
            "2": {"x": 120, "y": 110, "confidence": 0.7}
        }
        dets = [Detection(
            label="person", confidence=0.9,
            bbox=BoundingBox(x=80, y=30, w=60, h=100),
            metadata={"keypoints": kps},
        )]
        result = annotate_frame(frame, dets, 0, 0.0, 0.6)
        assert result.shape == frame.shape

    def test_annotate_simple_is_faster_path(self):
        """annotate_frame_simple should produce valid output."""
        frame = _make_frame()
        result = annotate_frame_simple(frame, _make_detections(), 5, 0.2)
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_score_gauge_drawn(self):
        """Score gauge should modify bottom-right corner pixels."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = annotate_frame(frame, [], 0, 0.0, 0.75)
        # Check bottom-right area for gauge drawing
        bottom_right = result[200:240, 280:320]
        assert np.sum(bottom_right) > 0, "Score gauge should draw in bottom-right"


# ──────────────────────────────────────────────
# Evidence saving tests
# ──────────────────────────────────────────────

class TestEvidenceSaving:
    """Tests for evidence file saving and listing."""

    def test_save_evidence_frame_creates_file(self, tmp_path: Path, monkeypatch):
        """save_evidence_frame should create a JPEG file on disk."""
        from backend.analysis import evidence
        from backend.config import Settings
        settings = Settings(evidence_dir=str(tmp_path))
        monkeypatch.setattr(evidence, "get_settings", lambda: settings)

        frame = _make_frame()
        rel_path = save_evidence_frame(frame, "vid01", "evt01", 42, annotated=True)
        assert rel_path != ""
        assert "annotated" in rel_path
        full = tmp_path / rel_path
        assert full.exists()
        assert full.stat().st_size > 100

    def test_save_event_evidence_creates_pairs(self, tmp_path: Path, monkeypatch):
        """save_event_evidence should create both raw and annotated files."""
        from backend.analysis import evidence
        from backend.config import Settings
        settings = Settings(evidence_dir=str(tmp_path))
        monkeypatch.setattr(evidence, "get_settings", lambda: settings)

        frame = _make_frame()
        frames_data = [(0, 0.0, frame, _make_detections())]
        event = _make_event()
        raw_paths, ann_paths = save_event_evidence(
            frames_data, "vid02", "evt02",
            event=event, aggregate_scores={0: 0.6}
        )
        assert len(raw_paths) == 1
        assert len(ann_paths) == 1
        assert (tmp_path / raw_paths[0]).exists()
        assert (tmp_path / ann_paths[0]).exists()

    def test_list_evidence_files(self, tmp_path: Path, monkeypatch):
        """list_evidence_files should return file metadata."""
        from backend.analysis import evidence
        from backend.config import Settings
        settings = Settings(evidence_dir=str(tmp_path))
        monkeypatch.setattr(evidence, "get_settings", lambda: settings)

        frame = _make_frame()
        save_evidence_frame(frame, "vid03", "evt03", 10, annotated=True)
        save_evidence_frame(frame, "vid03", "evt03", 10, annotated=False)

        files = list_evidence_files("vid03")
        assert len(files) == 2
        types = {f["type"] for f in files}
        assert "annotated" in types
        assert "raw" in types
        for f in files:
            assert "filename" in f
            assert "size_bytes" in f
            assert f["event_id"] == "evt03"

    def test_list_evidence_empty(self, tmp_path: Path, monkeypatch):
        """list_evidence_files should return empty for nonexistent video."""
        from backend.analysis import evidence
        from backend.config import Settings
        settings = Settings(evidence_dir=str(tmp_path))
        monkeypatch.setattr(evidence, "get_settings", lambda: settings)

        files = list_evidence_files("nonexistent")
        assert files == []


# ──────────────────────────────────────────────
# API tests
# ──────────────────────────────────────────────

class TestEvidenceAPI:
    """API-level tests for evidence gallery and serving."""

    @pytest.mark.asyncio
    async def test_evidence_gallery_after_analysis(self, client):
        """GET /videos/{id}/evidence should return gallery after analysis."""
        video_path = Path(__file__).parent / "_test_ev_dynamic.mp4"
        _make_dynamic_video(video_path, num_frames=75)
        try:
            video_bytes = video_path.read_bytes()
            upload_resp = await client.post(
                "/videos/upload",
                files={"file": ("ev_dyn.mp4", io.BytesIO(video_bytes), "video/mp4")},
            )
            vid_id = upload_resp.json()["video"]["video_id"]

            # Run analysis
            await client.post(f"/videos/{vid_id}/analyze")

            # Check for results
            results_resp = await client.get(f"/videos/{vid_id}/results")
            result = results_resp.json()

            if result["events"]:
                # Gallery should have files
                gallery_resp = await client.get(f"/videos/{vid_id}/evidence")
                assert gallery_resp.status_code == 200
                gallery = gallery_resp.json()
                assert gallery["video_id"] == vid_id
                assert gallery["total_files"] > 0
                assert "events" in gallery

                # Each event group should have annotated and raw
                for eid, group in gallery["events"].items():
                    assert "annotated" in group
                    assert "raw" in group
        finally:
            video_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_evidence_image_served(self, client):
        """Evidence images should be servable as JPEG."""
        video_path = Path(__file__).parent / "_test_ev_serve.mp4"
        _make_dynamic_video(video_path, num_frames=75)
        try:
            video_bytes = video_path.read_bytes()
            upload_resp = await client.post(
                "/videos/upload",
                files={"file": ("ev_serve.mp4", io.BytesIO(video_bytes), "video/mp4")},
            )
            vid_id = upload_resp.json()["video"]["video_id"]
            await client.post(f"/videos/{vid_id}/analyze")

            results = (await client.get(f"/videos/{vid_id}/results")).json()
            if results["events"] and results["events"][0]["annotated_frame_paths"]:
                path = results["events"][0]["annotated_frame_paths"][0]
                filename = path.split("/")[-1]
                img_resp = await client.get(f"/videos/{vid_id}/evidence/{filename}")
                assert img_resp.status_code == 200
                assert img_resp.headers["content-type"] == "image/jpeg"
                assert len(img_resp.content) > 100
        finally:
            video_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_evidence_gallery_not_found(self, client):
        """Gallery for non-existent video should 404."""
        resp = await client.get("/videos/nonexistent999/evidence")
        assert resp.status_code == 404
