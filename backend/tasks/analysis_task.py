"""
MulTiCheat Pro — Celery Background Analysis Worker Task

Processes video analysis jobs asynchronously in the background.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Fallback fake celery app if Celery isn't running or imported
try:
    from celery import Celery
    celery_app = Celery("multicheat_tasks", broker="redis://localhost:6379/0")
    HAS_CELERY = True
except Exception:
    HAS_CELERY = False
    celery_app = None


async def run_analysis_async(video_id: str, job_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute analysis pipeline asynchronously."""
    from backend.analysis.pipeline import AnalysisPipeline
    pipeline = AnalysisPipeline(profile=options.get("profile", "BALANCED") if options else "BALANCED")
    result = await pipeline.process_video(video_id=video_id, job_id=job_id)
    return result


if HAS_CELERY and celery_app is not None:
    @celery_app.task(name="tasks.process_video_analysis", bind=True)
    def process_video_analysis_task(self, video_id: str, job_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Celery background worker entry point."""
        logger.info("Starting Celery task for video_id=%s, job_id=%s", video_id, job_id)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_analysis_async(video_id, job_id, options))
