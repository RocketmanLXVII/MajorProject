"""
MulTiCheat — Analysis API Router

Endpoints for triggering video analysis and retrieving results.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from backend.analysis.models import AnalysisResult
from backend.analysis.pipeline import AnalysisPipeline, PipelineConfig, load_cached_results, load_annotations
from backend.analysis.evidence import list_evidence_files
from backend.config import get_settings
from backend.video_service import find_video_file, load_metadata_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["analysis"])


# ──────────────────────────────────────────────
# POST /videos/{video_id}/analyze
# ──────────────────────────────────────────────

@router.post(
    "/{video_id}/analyze",
    response_model=AnalysisResult,
    response_model_exclude={"frame_results"},
    status_code=status.HTTP_200_OK,
    responses={
        404: {"description": "Video not found"},
        422: {"description": "Analysis failed"},
    },
)
async def analyze_video(video_id: str):
    """
    Trigger analysis on an uploaded video.

    Runs the detection pipeline and returns structured results.
    Evidence frames are saved to the evidence directory.
    """
    # Verify video exists
    video_path = find_video_file(video_id)
    if video_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found.",
        )

    metadata = load_metadata_cache(video_id)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metadata for video '{video_id}' not found.",
        )

    logger.info("Analysis requested for video %s", video_id)

    try:
        pipeline = AnalysisPipeline()
        result = pipeline.run(video_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Analysis failed: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected analysis error for %s: %s", video_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed unexpectedly: {exc}",
        )

    return result


# ──────────────────────────────────────────────
# GET /videos/{video_id}/results
# ──────────────────────────────────────────────

@router.get(
    "/{video_id}/results",
    response_model=AnalysisResult,
    responses={404: {"description": "Results not found"}},
)
async def get_analysis_results(video_id: str):
    """Retrieve cached analysis results for a video."""
    result = load_cached_results(video_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis results found for video '{video_id}'. Run analysis first.",
        )
    return result


# ──────────────────────────────────────────────
# GET /videos/{video_id}/annotations
# ──────────────────────────────────────────────

@router.get(
    "/{video_id}/annotations",
    responses={404: {"description": "Annotations not found"}},
)
async def get_video_annotations(video_id: str):
    """Retrieve raw frame annotations (bboxes, keypoints) for a video."""
    annotations = load_annotations(video_id)
    if annotations is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No annotations found for video '{video_id}'. Run analysis first.",
        )
    return annotations


# ──────────────────────────────────────────────
# GET /videos/{video_id}/evidence
# ──────────────────────────────────────────────

@router.get("/{video_id}/evidence")
async def get_evidence_gallery(video_id: str):
    """List all evidence files for a video, grouped by event."""
    files = list_evidence_files(video_id)
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No evidence found for video '{video_id}'.",
        )

    # Group by event_id
    events_map: dict[str, list[dict]] = {}
    for f in files:
        eid = f.get("event_id", "unknown")
        events_map.setdefault(eid, []).append(f)

    return {
        "video_id": video_id,
        "total_files": len(files),
        "events": {
            eid: {
                "annotated": [f for f in flist if f["type"] == "annotated"],
                "raw": [f for f in flist if f["type"] == "raw"],
            }
            for eid, flist in events_map.items()
        },
    }


# ──────────────────────────────────────────────
# GET /videos/{video_id}/evidence/{filename}
# ──────────────────────────────────────────────

@router.get(
    "/{video_id}/evidence/{filename}",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Evidence image"},
        404: {"description": "Evidence file not found"},
    },
)
async def get_evidence_image(video_id: str, filename: str):
    """Serve an evidence image file."""
    settings = get_settings()
    filepath = settings.evidence_path / video_id / filename

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence file '{filename}' not found for video '{video_id}'.",
        )

    # Security: ensure the file is within the evidence directory
    try:
        filepath.resolve().relative_to(settings.evidence_path.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied.",
        )

    return FileResponse(
        path=str(filepath),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )
