"""
Tests for Phase 7 Export Router endpoints.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest

from tests.test_analysis import create_dynamic_video

@pytest.mark.asyncio
class TestExportAPI:
    """Tests for export endpoints (JSON, CSV, ZIP)."""

    async def test_exports_after_analysis(self, client):
        video_path = Path(__file__).parent / "_test_export_dyn.mp4"
        create_dynamic_video(video_path, num_frames=50)
        try:
            video_bytes = video_path.read_bytes()
            upload_resp = await client.post(
                "/videos/upload",
                files={"file": ("export_dyn.mp4", io.BytesIO(video_bytes), "video/mp4")},
            )
            vid_id = upload_resp.json()["video"]["video_id"]

            await client.post(f"/videos/{vid_id}/analyze")

            # 1. Test JSON Export
            json_resp = await client.get(f"/export/{vid_id}/json")
            assert json_resp.status_code == 200
            assert json_resp.headers["content-type"] == "application/json"
            assert "attachment" in json_resp.headers["content-disposition"]
            assert "json" in json_resp.headers["content-disposition"]
            data = json_resp.json()
            assert "events" in data

            # 2. Test CSV Export
            csv_resp = await client.get(f"/export/{vid_id}/csv")
            assert csv_resp.status_code == 200
            assert csv_resp.headers["content-type"] == "text/csv; charset=utf-8"
            assert "csv" in csv_resp.headers["content-disposition"]
            csv_text = csv_resp.content.decode("utf-8")
            assert "Event ID" in csv_text
            assert vid_id in csv_text  # Make sure data is actually inside rows

            # 3. Test ZIP Export
            # The testing environment evidence script runs on Dummy, YOLODetector requires objects,
            # Motion detector usually detects moving square, so evidence might be present.
            zip_resp = await client.get(f"/export/{vid_id}/zip")
            assert zip_resp.status_code == 200
            assert zip_resp.headers["content-type"] == "application/zip"
            assert "zip" in zip_resp.headers["content-disposition"]
            assert len(zip_resp.content) > 100 # Should contain zip structure
        finally:
            video_path.unlink(missing_ok=True)

    async def test_exports_not_found(self, client):
        resp = await client.get("/export/nonexistent/json")
        assert resp.status_code == 404
        resp = await client.get("/export/nonexistent/csv")
        assert resp.status_code == 404
        resp = await client.get("/export/nonexistent/zip")
        assert resp.status_code == 404
