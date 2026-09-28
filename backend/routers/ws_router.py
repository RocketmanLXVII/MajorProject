"""
MulTiCheat Pro — WebSockets / Socket.IO Router

Delivers real-time frame detection metadata, active track positions, and malpractice alert events.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

ws_router = APIRouter(prefix="/ws", tags=["WebSockets"])


class ConnectionManager:
    """Manages active WebSockets connections per video job session."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)
        logger.info("WebSocket client connected to job %s", job_id)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        logger.info("WebSocket client disconnected from job %s", job_id)

    async def broadcast_to_job(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            payload = json.dumps(message, default=str)
            for connection in list(self.active_connections[job_id]):
                try:
                    await connection.send_text(payload)
                except Exception as e:
                    logger.warning("Error sending WS message: %s", e)
                    self.disconnect(connection, job_id)


manager = ConnectionManager()


@ws_router.websocket("/live/{job_id}")
async def websocket_live_stream(websocket: WebSocket, job_id: str):
    """Real-time streaming endpoint for frame telemetry and active detection events."""
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep-alive receive loop
            data = await websocket.receive_text()
            # Echo back ping response
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error("WebSocket exception on job %s: %s", job_id, e)
        manager.disconnect(websocket, job_id)
