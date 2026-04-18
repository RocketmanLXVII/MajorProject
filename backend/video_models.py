"""
MulTiCheat — Video Data Models

Pydantic models for video metadata, upload responses, and video listings.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VideoStatus(str, Enum):
    """Processing status of an uploaded video."""
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    """Metadata extracted from an uploaded video file."""
    video_id: str = Field(..., description="Unique identifier for the video")
    original_filename: str = Field(..., description="Original upload filename")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    duration_seconds: float = Field(..., ge=0.0, description="Video duration in seconds")
    fps: float = Field(..., ge=0.0, description="Frames per second")
    width: int = Field(..., ge=0, description="Frame width in pixels")
    height: int = Field(..., ge=0, description="Frame height in pixels")
    frame_count: int = Field(..., ge=0, description="Total number of frames")
    codec: str = Field(default="unknown", description="Video codec identifier")
    upload_time: str = Field(..., description="ISO 8601 upload timestamp")
    status: VideoStatus = Field(default=VideoStatus.READY, description="Current status")

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


class VideoUploadResponse(BaseModel):
    """Response returned after a successful video upload."""
    message: str = "Video uploaded successfully"
    video: VideoMetadata


class VideoListResponse(BaseModel):
    """Response for listing all uploaded videos."""
    count: int
    videos: list[VideoMetadata]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
