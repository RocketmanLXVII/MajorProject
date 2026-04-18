"""
MulTiCheat — Video API Router

Endpoints for uploading, listing, retrieving, and managing exam videos.
"""

from __future__ import annotations

import io
import logging

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import StreamingResponse

from backend.video_models import (
    ErrorResponse,
    VideoListResponse,
    VideoMetadata,
    VideoUploadResponse,
)
from backend.video_service import (
    cleanup_video,
    extract_metadata,
    find_video_file,
    generate_video_id,
    get_frame_at,
    list_all_videos,
    load_metadata_cache,
    save_upload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])


# ──────────────────────────────────────────────
# POST /videos/upload
# ──────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or empty"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Corrupt or unreadable video"},
    },
)
async def upload_video(file: UploadFile = File(..., description="Video file to upload")):
    """
    Upload a video file for analysis.

    Validates file type and size, saves to disk, extracts metadata.
    """
    video_id = generate_video_id()
    logger.info("Upload started: video_id=%s, filename=%s", video_id, file.filename)

    # Save file to disk (validates extension and size)
    try:
        video_path = await save_upload(file, video_id)
    except ValueError as exc:
        error_msg = str(exc)
        if "size" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    # Extract metadata (validates the video is decodable)
    try:
        metadata = extract_metadata(video_path, video_id, file.filename or "unknown")
    except ValueError as exc:
        cleanup_video(video_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Video validation failed: {exc}",
        )

    logger.info("Upload complete: video_id=%s, duration=%.1fs", video_id, metadata.duration_seconds)
    return VideoUploadResponse(video=metadata)


# ──────────────────────────────────────────────
# GET /videos
# ──────────────────────────────────────────────

@router.get("/", response_model=VideoListResponse)
async def list_videos():
    """List all uploaded videos with metadata."""
    videos = list_all_videos()
    return VideoListResponse(count=len(videos), videos=videos)


# ──────────────────────────────────────────────
# GET /videos/{video_id}
# ──────────────────────────────────────────────

@router.get(
    "/{video_id}",
    response_model=VideoMetadata,
    responses={404: {"model": ErrorResponse}},
)
async def get_video(video_id: str):
    """Get metadata for a specific video."""
    meta = load_metadata_cache(video_id)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found.",
        )
    return meta


# ──────────────────────────────────────────────
# GET /videos/{video_id}/frame/{index}
# ──────────────────────────────────────────────

@router.get(
    "/{video_id}/frame/{index}",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Frame as JPEG"},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def get_video_frame(video_id: str, index: int):
    """Extract and serve a specific frame as JPEG."""
    video_path = find_video_file(video_id)
    if video_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found.",
        )

    try:
        frame = get_frame_at(video_path, index)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # Encode as JPEG
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode frame as JPEG.",
        )

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ──────────────────────────────────────────────
# GET /videos/{video_id}/stream
# ──────────────────────────────────────────────

from fastapi import Request, Response
import os

@router.get(
    "/{video_id}/stream",
    responses={
        200: {"description": "Full Video Stream"},
        206: {"description": "Partial Video Stream"},
        404: {"model": ErrorResponse},
    },
)
async def stream_video(video_id: str, request: Request):
    """
    Stream video content to the browser, supporting HTTP Range requests.
    This enables video seeking in HTML5 players.
    """
    video_path = find_video_file(video_id)
    if video_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found.",
        )

    file_size = os.stat(video_path).st_size
    range_header = request.headers.get("range")
    
    # Determine content-type from extension
    ext = video_path.suffix.lower()
    content_type = "video/mp4"
    if ext == ".webm":
        content_type = "video/webm"
    elif ext == ".mkv":
        content_type = "video/x-matroska"
    
    headers = {
        "Content-Type": content_type,
        "Accept-Ranges": "bytes",
        "Content-Encoding": "identity",
        "Content-Length": str(file_size),
        "Access-Control-Expose-Headers": "Content-Type, Accept-Ranges, Content-Length, Content-Range, Content-Encoding",
    }

    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1

        if start >= file_size or end >= file_size:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

        content_length = end - start + 1
        headers["Content-Length"] = str(content_length)
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        
        def iterfile():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(remaining, 1024 * 64))  # 64KB chunks
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iterfile(), 
            status_code=status.HTTP_206_PARTIAL_CONTENT, 
            headers=headers
        )
    
    # If no range request, just stream the whole file
    def iter_full_file():
        with open(video_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 64)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        iter_full_file(),
        status_code=status.HTTP_200_OK,
        headers=headers
    )

# ──────────────────────────────────────────────
# DELETE /videos/{video_id}
# ──────────────────────────────────────────────

@router.delete(
    "/{video_id}",
    status_code=status.HTTP_200_OK,
    responses={404: {"model": ErrorResponse}},
)
async def delete_video(video_id: str):
    """Delete a video and all associated data."""
    deleted = cleanup_video(video_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found.",
        )
    logger.info("Deleted video %s", video_id)
    return {"message": f"Video '{video_id}' deleted successfully."}
