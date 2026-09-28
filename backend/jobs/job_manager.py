"""
MulTiCheat — Asynchronous Analysis Job Queue & Manager

Prevents API blocking during long video analysis. Handles job creation,
background processing, progress percentage tracking, and status retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.models import AnalysisJobRecord

logger = logging.getLogger(__name__)

# In-memory job state cache for high-throughput polling
_JOBS_CACHE: Dict[str, Dict[str, Any]] = {}


class JobManager:
    """Manages asynchronous video analysis jobs."""

    def __init__(self):
        self._active_tasks: Dict[str, asyncio.Task] = {}

    def create_job(self, video_id: str, profile: str = "BALANCED") -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        job_info = {
            "job_id": job_id,
            "video_id": video_id,
            "status": "QUEUED",
            "progress_pct": 0.0,
            "current_step": "Queued for processing",
            "profile": profile,
            "created_at": now,
            "updated_at": now,
            "error_message": None,
            "result": None,
        }
        _JOBS_CACHE[job_id] = job_info
        return job_id

    def update_progress(
        self,
        job_id: str,
        progress_pct: float,
        step: str,
        status: str = "PROCESSING",
        error: Optional[str] = None,
    ) -> None:
        if job_id in _JOBS_CACHE:
            _JOBS_CACHE[job_id]["progress_pct"] = round(progress_pct, 1)
            _JOBS_CACHE[job_id]["current_step"] = step
            _JOBS_CACHE[job_id]["status"] = status
            _JOBS_CACHE[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            if error:
                _JOBS_CACHE[job_id]["error_message"] = error

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        if job_id in _JOBS_CACHE:
            _JOBS_CACHE[job_id]["progress_pct"] = 100.0
            _JOBS_CACHE[job_id]["current_step"] = "Analysis Completed"
            _JOBS_CACHE[job_id]["status"] = "COMPLETED"
            _JOBS_CACHE[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            _JOBS_CACHE[job_id]["result"] = result

    def fail_job(self, job_id: str, error_message: str) -> None:
        if job_id in _JOBS_CACHE:
            _JOBS_CACHE[job_id]["progress_pct"] = _JOBS_CACHE[job_id].get("progress_pct", 0.0)
            _JOBS_CACHE[job_id]["current_step"] = "Failed"
            _JOBS_CACHE[job_id]["status"] = "FAILED"
            _JOBS_CACHE[job_id]["error_message"] = error_message
            _JOBS_CACHE[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return _JOBS_CACHE.get(job_id)


def get_job_manager() -> JobManager:
    if not hasattr(get_job_manager, "_instance"):
        get_job_manager._instance = JobManager()  # type: ignore
    return get_job_manager._instance  # type: ignore
