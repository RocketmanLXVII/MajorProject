"""
Phase 1 — Video Ingestion Tests

Tests:
1. Upload a valid video → 201 + correct metadata
2. Upload an invalid file type → 400
3. Upload a corrupt/empty file → 400/422
4. List videos after upload
5. Get single video metadata by ID
6. Get a frame by index → JPEG
7. Get frame with out-of-range index → 422
8. Delete a video → 200
9. Delete non-existent video → 404
10. Get non-existent video → 404
11. Video service direct tests (extract_metadata, sample_frames)
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.video_service import (
    extract_metadata,
    find_video_file,
    sample_frames,
    get_frame_at,
    cleanup_video,
    ALLOWED_EXTENSIONS,
)


# ──────────────────────────────────────────────
# Helpers — synthetic test video
# ──────────────────────────────────────────────

def create_test_video(
    path: Path,
    width: int = 320,
    height: int = 240,
    fps: float = 25.0,
    num_frames: int = 50,
    codec: str = "mp4v",
) -> Path:
    """Create a small synthetic MP4 video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create test video at {path}")

    for i in range(num_frames):
        # Create a frame with a changing color to make each frame unique
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = int(255 * i / num_frames)  # Blue gradient
        frame[:, :, 1] = 128
        frame[:, :, 2] = int(255 * (1 - i / num_frames))  # Red gradient
        # Draw frame number
        cv2.putText(
            frame, f"F{i}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )
        writer.write(frame)

    writer.release()
    return path


@pytest.fixture()
def test_video_path(tmp_path: Path) -> Path:
    """Create a temporary test video file."""
    video_path = tmp_path / "test_video.mp4"
    return create_test_video(video_path)


@pytest.fixture()
def test_video_bytes(test_video_path: Path) -> bytes:
    """Read test video as bytes."""
    return test_video_path.read_bytes()


# ──────────────────────────────────────────────
# Service-level tests
# ──────────────────────────────────────────────

class TestVideoService:
    """Direct tests for video_service functions."""

    def test_extract_metadata_valid_video(self, test_video_path: Path):
        """extract_metadata should return correct metadata for a valid video."""
        meta = extract_metadata(test_video_path, "test123", "test_video.mp4")
        assert meta.video_id == "test123"
        assert meta.original_filename == "test_video.mp4"
        assert meta.width == 320
        assert meta.height == 240
        assert meta.fps > 0
        assert meta.frame_count == 50
        assert meta.duration_seconds > 0
        assert meta.file_size_bytes > 0

    def test_extract_metadata_corrupt_file(self, tmp_path: Path):
        """extract_metadata should raise ValueError for a corrupt file."""
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"this is not a video file at all")
        with pytest.raises(ValueError, match="Cannot open video"):
            extract_metadata(corrupt, "bad1", "corrupt.mp4")

    def test_get_frame_at_valid_index(self, test_video_path: Path):
        """get_frame_at should return a frame array for a valid index."""
        frame = get_frame_at(test_video_path, 0)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)

    def test_get_frame_at_last_frame(self, test_video_path: Path):
        """get_frame_at should work for the last frame (index 49)."""
        frame = get_frame_at(test_video_path, 49)
        assert isinstance(frame, np.ndarray)

    def test_get_frame_at_out_of_range(self, test_video_path: Path):
        """get_frame_at should raise ValueError for out-of-range index."""
        with pytest.raises(ValueError, match="out of range"):
            get_frame_at(test_video_path, 999)

    def test_get_frame_at_negative_index(self, test_video_path: Path):
        """get_frame_at should raise ValueError for negative index."""
        with pytest.raises(ValueError, match="out of range"):
            get_frame_at(test_video_path, -1)

    def test_sample_frames_returns_frames(self, test_video_path: Path):
        """sample_frames should return a list of (index, frame) tuples."""
        frames = sample_frames(test_video_path, interval_seconds=0.5)
        assert len(frames) > 0
        for idx, frame in frames:
            assert isinstance(idx, int)
            assert isinstance(frame, np.ndarray)
            assert frame.shape[0] > 0 and frame.shape[1] > 0

    def test_sample_frames_respects_max(self, test_video_path: Path):
        """sample_frames should stop at max_frames."""
        frames = sample_frames(test_video_path, interval_seconds=0.04, max_frames=5)
        assert len(frames) <= 5


# ──────────────────────────────────────────────
# API-level tests
# ──────────────────────────────────────────────

class TestVideoAPI:
    """API endpoint tests for video ingestion."""

    @pytest.mark.asyncio
    async def test_upload_valid_video(self, client, test_video_bytes: bytes):
        """POST /videos/upload with a valid MP4 should return 201."""
        response = await client.post(
            "/videos/upload",
            files={"file": ("test.mp4", io.BytesIO(test_video_bytes), "video/mp4")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Video uploaded successfully"
        video = data["video"]
        assert video["width"] == 320
        assert video["height"] == 240
        assert video["frame_count"] == 50
        assert video["status"] == "ready"
        # Store for later tests
        self.__class__._uploaded_id = video["video_id"]

    @pytest.mark.asyncio
    async def test_upload_invalid_type(self, client):
        """POST /videos/upload with a .txt file should return 400."""
        response = await client.post(
            "/videos/upload",
            files={"file": ("file.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_empty_file(self, client):
        """POST /videos/upload with an empty MP4 should return 400."""
        response = await client.post(
            "/videos/upload",
            files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_corrupt_file(self, client):
        """POST /videos/upload with corrupt bytes should return 422."""
        response = await client.post(
            "/videos/upload",
            files={"file": ("corrupt.mp4", io.BytesIO(b"not-a-video"), "video/mp4")},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_videos(self, client, test_video_bytes: bytes):
        """GET /videos/ should list uploaded videos."""
        # Upload first
        await client.post(
            "/videos/upload",
            files={"file": ("list_test.mp4", io.BytesIO(test_video_bytes), "video/mp4")},
        )
        response = await client.get("/videos/")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1
        assert len(data["videos"]) >= 1

    @pytest.mark.asyncio
    async def test_get_video_metadata(self, client, test_video_bytes: bytes):
        """GET /videos/{id} should return metadata for an uploaded video."""
        upload_resp = await client.post(
            "/videos/upload",
            files={"file": ("meta_test.mp4", io.BytesIO(test_video_bytes), "video/mp4")},
        )
        vid_id = upload_resp.json()["video"]["video_id"]
        response = await client.get(f"/videos/{vid_id}")
        assert response.status_code == 200
        assert response.json()["video_id"] == vid_id

    @pytest.mark.asyncio
    async def test_get_video_not_found(self, client):
        """GET /videos/{id} for non-existent ID should return 404."""
        response = await client.get("/videos/nonexistent123")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_frame(self, client, test_video_bytes: bytes):
        """GET /videos/{id}/frame/0 should return JPEG image."""
        upload_resp = await client.post(
            "/videos/upload",
            files={"file": ("frame_test.mp4", io.BytesIO(test_video_bytes), "video/mp4")},
        )
        vid_id = upload_resp.json()["video"]["video_id"]
        response = await client.get(f"/videos/{vid_id}/frame/0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 100  # JPEG should have some data

    @pytest.mark.asyncio
    async def test_get_frame_out_of_range(self, client, test_video_bytes: bytes):
        """GET /videos/{id}/frame/9999 should return 422."""
        upload_resp = await client.post(
            "/videos/upload",
            files={"file": ("oor_test.mp4", io.BytesIO(test_video_bytes), "video/mp4")},
        )
        vid_id = upload_resp.json()["video"]["video_id"]
        response = await client.get(f"/videos/{vid_id}/frame/9999")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_video(self, client, test_video_bytes: bytes):
        """DELETE /videos/{id} should remove the video."""
        upload_resp = await client.post(
            "/videos/upload",
            files={"file": ("del_test.mp4", io.BytesIO(test_video_bytes), "video/mp4")},
        )
        vid_id = upload_resp.json()["video"]["video_id"]
        response = await client.delete(f"/videos/{vid_id}")
        assert response.status_code == 200

        # Verify it's gone
        response = await client.get(f"/videos/{vid_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, client):
        """DELETE /videos/{id} for non-existent should return 404."""
        response = await client.delete("/videos/nonexistent999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_video_full(self, client, test_video_bytes: bytes):
        """GET /videos/{id}/stream without Range should return 200 and full content."""
        upload_resp = await client.post(
            "/videos/upload",
            files={"file": ("stream_full.mp4", io.BytesIO(test_video_bytes), "video/mp4")},
        )
        vid_id = upload_resp.json()["video"]["video_id"]
        response = await client.get(f"/videos/{vid_id}/stream")
        assert response.status_code == 200
        assert len(response.content) == len(test_video_bytes)

    @pytest.mark.asyncio
    async def test_stream_video_partial_range(self, client, test_video_bytes: bytes):
        """GET /videos/{id}/stream with Range header should return 206 Partial Content."""
        upload_resp = await client.post(
            "/videos/upload",
            files={"file": ("stream_part.mp4", io.BytesIO(test_video_bytes), "video/mp4")},
        )
        vid_id = upload_resp.json()["video"]["video_id"]
        # Request a chunk
        response = await client.get(
            f"/videos/{vid_id}/stream",
            headers={"Range": "bytes=0-99"}
        )
        assert response.status_code == 206
        assert len(response.content) == 100
        assert response.headers.get("Content-Type") == "video/mp4"

