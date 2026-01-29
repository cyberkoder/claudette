"""
WebSocket handlers for Claudette web dashboard.

Provides real-time state updates, audio levels, and log streaming.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("claudette.web")


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize connection manager."""
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self._connections.append(websocket)
        logger.debug(f"WebSocket connected. Total connections: {len(self._connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.debug(f"WebSocket disconnected. Total connections: {len(self._connections)}")

    async def send_to(self, websocket: WebSocket, message: dict):
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self._connections:
            return

        disconnected = []
        for connection in self._connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    @property
    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)


def create_ws_router(manager: ConnectionManager, state_manager: Any) -> APIRouter:
    """Create WebSocket router.

    Args:
        manager: ConnectionManager instance for managing connections
        state_manager: ClaudetteStateManager instance

    Returns:
        Configured APIRouter with WebSocket endpoint
    """
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates.

        Messages sent to clients:
        - {"type": "state", "data": {...}} - State changes
        - {"type": "audio_level", "data": {"level": 0.5}} - Audio levels
        - {"type": "log", "data": {...}} - New log entries
        - {"type": "connected", "data": {"client_count": N}} - Connection established

        Messages received from clients:
        - {"type": "ping"} - Heartbeat
        - {"type": "subscribe", "channels": ["state", "audio", "logs"]} - Subscribe to specific channels
        """
        await manager.connect(websocket)

        # Send initial state
        try:
            snapshot = state_manager.get_snapshot()
            claude_activity = state_manager.get_claude_activity()
            await manager.send_to(
                websocket,
                {
                    "type": "connected",
                    "data": {
                        "client_count": manager.connection_count,
                        "initial_state": {
                            "state": snapshot.state,
                            "conversation_mode": snapshot.conversation_mode,
                            "awaiting_confirmation": snapshot.awaiting_confirmation,
                            "last_transcription": snapshot.last_transcription,
                            "last_response": snapshot.last_response,
                            "uptime_seconds": snapshot.uptime_seconds,
                            "timestamp": snapshot.timestamp,
                        },
                        "claude_activity": {
                            "active": claude_activity.active,
                            "query": claude_activity.query,
                            "status": claude_activity.status,
                            "current_output": claude_activity.current_output,
                            "progress_lines": claude_activity.progress_lines,
                            "started_at": claude_activity.started_at,
                            "elapsed_seconds": claude_activity.elapsed_seconds,
                        },
                    },
                },
            )
        except Exception as e:
            logger.error(f"Failed to send initial state: {e}")

        try:
            while True:
                # Wait for messages from client
                data = await websocket.receive_text()

                try:
                    message = json.loads(data)
                    msg_type = message.get("type", "")

                    if msg_type == "ping":
                        # Respond to ping with pong
                        await manager.send_to(websocket, {"type": "pong"})

                    elif msg_type == "get_state":
                        # Send current state on request
                        snapshot = state_manager.get_snapshot()
                        await manager.send_to(
                            websocket,
                            {
                                "type": "state",
                                "data": {
                                    "state": snapshot.state,
                                    "conversation_mode": snapshot.conversation_mode,
                                    "awaiting_confirmation": snapshot.awaiting_confirmation,
                                    "last_transcription": snapshot.last_transcription,
                                    "last_response": snapshot.last_response,
                                    "uptime_seconds": snapshot.uptime_seconds,
                                    "timestamp": snapshot.timestamp,
                                },
                            },
                        )

                    elif msg_type == "get_logs":
                        # Send recent logs on request
                        limit = message.get("limit", 50)
                        logs = state_manager.get_recent_logs(limit=limit)
                        await manager.send_to(
                            websocket,
                            {
                                "type": "logs",
                                "data": {
                                    "entries": [
                                        {
                                            "timestamp": log.timestamp,
                                            "level": log.level,
                                            "message": log.message,
                                        }
                                        for log in logs
                                    ]
                                },
                            },
                        )

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except WebSocketDisconnect:
            manager.disconnect(websocket)
            logger.debug("WebSocket client disconnected")
        except Exception as e:
            manager.disconnect(websocket)
            logger.error(f"WebSocket error: {e}")

    return router
