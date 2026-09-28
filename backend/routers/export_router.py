"""
MulTiCheat — Export Router

API endpoints for generating downloadable reports (JSON, CSV, ZIP).
"""

import csv
import io
import json
import zipfile
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from backend.analysis.pipeline import load_cached_results
from backend.video_service import get_video_dir

router = APIRouter(prefix="/export", tags=["Exports"])

@router.get("/{video_id}/json")
async def export_json(video_id: str):
    """Export complete analysis results as JSON format."""
    results = load_cached_results(video_id)
    if not results:
        raise HTTPException(status_code=404, detail="Analysis results not found or not generated.")

    response_data = results.model_dump_json(indent=2)
    
    return Response(
        content=response_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=multicheat_report_{video_id}.json"
        }
    )

@router.get("/{video_id}/csv")
async def export_csv(video_id: str):
    """Export flattened events list as a CSV."""
    results = load_cached_results(video_id)
    if not results:
        raise HTTPException(status_code=404, detail="Analysis results not found.")

    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow([
        "Event ID", "Video ID", "Category", 
        "Severity", "Confidence Score", 
        "Start Time (s)", "End Time (s)",
        "Frames Captured", "Explanation"
    ])
    
    if not results.events:
        writer.writerow([
            "N/A", video_id, "normal", "low", "0.00", "0.00", "0.00", 0, "No malpractice events detected"
        ])
    else:
        for event in results.events:
            writer.writerow([
                event.event_id,
                event.video_id,
                event.predicted_class.value,
                event.severity.value,
                f"{event.confidence_score:.2f}",
                f"{event.start_timestamp:.2f}",
                f"{event.end_timestamp:.2f}",
                event.end_frame - event.start_frame + 1,
                event.explanation_text
            ])
        
    csv_bytes = output.getvalue().encode('utf-8')

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=multicheat_events_{video_id}.csv"
        }
    )

@router.get("/{video_id}/zip")
async def export_evidence_zip(video_id: str):
    """Generate a combined ZIP archive of all evidence frames (annotated)."""
    results = load_cached_results(video_id)
    if not results:
        raise HTTPException(status_code=404, detail="Analysis results not found.")

    vid_dir = get_video_dir(video_id)
    ev_dir = vid_dir.parent.parent / "evidence" / video_id

    if not ev_dir.exists():
        ev_dir.mkdir(parents=True, exist_ok=True)
        
    # Build ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        files_added = 0
        for event in results.events:
            for ann_path in event.annotated_frame_paths:
                # Resolve true disk path based on relative paths returned from model
                # Annotated_path is something like '64e93eda832f/b8x_annotated_14.jpg'
                rel_parts = Path(ann_path).parts
                if len(rel_parts) < 2:
                    continue
                file_name = rel_parts[-1]
                disk_path = ev_dir / file_name
                
                if disk_path.exists():
                    files_added += 1
                    # Structure ZIP: event_{id}/filename.jpg
                    zip_file.write(disk_path, arcname=f"event_{event.event_id}/{file_name}")
                    
        if files_added == 0:
            # If no files, write a dummy text file to prevent invalid zip
            zip_file.writestr("info.txt", "No annotated evidence images found.")

    zip_bytes = zip_buffer.getvalue()

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=multicheat_evidence_{video_id}.zip"
        }
    )
