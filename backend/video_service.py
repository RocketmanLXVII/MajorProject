"""
MulTiCheat — Video Ingestion & Processing Service

Handles:
- Saving uploaded video files with size validation
- Extracting metadata via OpenCV (duration, fps, resolution, frame count, codec)
- Validating that video files are decodable
- Sampling frames at regular intervals
- Extracting individual frames by index
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile

from backend.config import get_settings
from backend.video_models import VideoMetadata, VideoStatus

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

ALLOWED_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_MIME_PREFIXES: set[str] = {"video/"}
CHUNK_SIZE: int = 1024 * 1024  # 1 MB chunks for streaming upload


# ──────────────────────────────────────────────
# Upload & Storage
# ──────────────────────────────────────────────

def generate_video_id() -> str:
    """Generate a unique video identifier."""
    return uuid.uuid4().hex[:12]


def get_video_dir(video_id: str) -> Path:
    """Return the storage directory for a given video."""
    settings = get_settings()
    return settings.upload_path / video_id


def _validate_extension(filename: str) -> str:
    """Validate file extension. Returns the lowercase extension or raises ValueError."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


async def save_upload(file: UploadFile, video_id: str) -> Path:
    """
    Stream-save an uploaded file to disk with size validation.

    Returns the path to the saved file.
    Raises ValueError on validation failure.
    """
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Validate filename
    if not file.filename:
        raise ValueError("Upload must have a filename.")
    ext = _validate_extension(file.filename)

    # Create video directory
    video_dir = get_video_dir(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    dest_path = video_dir / f"original{ext}"

    # Stream write with size check
    total_written = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_bytes:
                    out.close()
                    cleanup_video(video_id)
                    raise ValueError(
                        f"File exceeds maximum size of {settings.max_upload_size_mb} MB."
                    )
                out.write(chunk)
    except ValueError:
        raise
    except Exception as exc:
        cleanup_video(video_id)
        logger.error("Failed to save upload %s: %s", video_id, exc)
        raise ValueError(f"Failed to save upload: {exc}") from exc

    if total_written == 0:
        cleanup_video(video_id)
        raise ValueError("Uploaded file is empty.")

    logger.info("Saved upload %s (%d bytes) to %s", video_id, total_written, dest_path)
    return dest_path


# ──────────────────────────────────────────────
# Metadata Extraction
# ──────────────────────────────────────────────

def extract_metadata(video_path: Path, video_id: str, original_filename: str) -> VideoMetadata:
    """
    Extract video metadata using OpenCV.

    Raises ValueError if the video cannot be opened or is invalid.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path.name}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Decode codec fourcc
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        if fourcc_int > 0:
            codec = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
        else:
            codec = "unknown"

        # Calculate duration
        duration = (frame_count / fps) if fps > 0 else 0.0

        # Sanity: try to read the first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError("Video file is corrupt or empty — cannot read first frame.")

        file_size = video_path.stat().st_size

        metadata = VideoMetadata(
            video_id=video_id,
            original_filename=original_filename,
            file_size_bytes=file_size,
            duration_seconds=round(duration, 3),
            fps=round(fps, 3),
            width=width,
            height=height,
            frame_count=frame_count,
            codec=codec.strip(),
            upload_time=datetime.now(timezone.utc).isoformat(),
            status=VideoStatus.READY,
        )

        # Cache metadata to disk
        _save_metadata_cache(video_id, metadata)

        logger.info(
            "Extracted metadata for %s: %dx%d, %.1ffps, %d frames, %.1fs",
            video_id, width, height, fps, frame_count, duration,
        )
        return metadata

    finally:
        cap.release()


def _save_metadata_cache(video_id: str, metadata: VideoMetadata) -> None:
    """Persist metadata as JSON alongside the video."""
    cache_path = get_video_dir(video_id) / "metadata.json"
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to cache metadata for %s: %s", video_id, exc)


def load_metadata_cache(video_id: str) -> Optional[VideoMetadata]:
    """Load cached metadata from disk. Returns None if not found."""
    cache_path = get_video_dir(video_id) / "metadata.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return VideoMetadata(**data)
    except Exception as exc:
        logger.warning("Failed to load metadata cache for %s: %s", video_id, exc)
        return None


# ──────────────────────────────────────────────
# Video File Discovery
# ──────────────────────────────────────────────

def find_video_file(video_id: str) -> Optional[Path]:
    """Find the original video file for a given video_id."""
    video_dir = get_video_dir(video_id)
    if not video_dir.exists():
        return None
    for ext in ALLOWED_EXTENSIONS:
        candidate = video_dir / f"original{ext}"
        if candidate.exists():
            return candidate
    return None


def list_all_videos() -> list[VideoMetadata]:
    """List all uploaded videos with cached metadata."""
    settings = get_settings()
    upload_dir = settings.upload_path
    if not upload_dir.exists():
        return []

    videos: list[VideoMetadata] = []
    for entry in sorted(upload_dir.iterdir()):
        if entry.is_dir():
            meta = load_metadata_cache(entry.name)
            if meta is not None:
                videos.append(meta)
    return videos


# ──────────────────────────────────────────────
# Frame Extraction
# ──────────────────────────────────────────────

def get_frame_at(video_path: Path, frame_index: int) -> np.ndarray:
    """
    Extract a single frame by index.

    Raises ValueError if the frame cannot be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path.name}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError(
                f"Frame index {frame_index} out of range [0, {total_frames - 1}]."
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame {frame_index}.")

        return frame
    finally:
        cap.release()


def sample_frames(
    video_path: Path,
    interval_seconds: float = 1.0,
    max_frames: int = 500,
) -> list[tuple[int, np.ndarray]]:
    """
    Sample frames at regular time intervals.

    Returns list of (frame_index, frame_array) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path.name}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_interval = max(1, int(fps * interval_seconds))

        frames: list[tuple[int, np.ndarray]] = []
        for idx in range(0, total_frames, frame_interval):
            if len(frames) >= max_frames:
                logger.info("Reached max_frames limit (%d), stopping sampling.", max_frames)
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.debug("Skipping unreadable frame %d", idx)
                continue

            frames.append((idx, frame))

        logger.info(
            "Sampled %d frames from %s (interval=%.1fs)",
            len(frames), video_path.name, interval_seconds,
        )
        return frames

    finally:
        cap.release()


# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

def cleanup_video(video_id: str) -> bool:
    """Remove all files for a given video. Returns True if deleted."""
    video_dir = get_video_dir(video_id)
    if video_dir.exists():
        shutil.rmtree(video_dir, ignore_errors=True)
        logger.info("Cleaned up video %s", video_id)
        return True
    return False
