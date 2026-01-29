"""
Claudette Web Server

FastAPI-based web server for the Claudette dashboard.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("claudette.web")

# Lazy imports for FastAPI/Uvicorn
_fastapi = None
_uvicorn = None


def _import_fastapi():
    """Lazy import FastAPI to avoid import errors if not installed."""
    global _fastapi
    if _fastapi is None:
        try:
            import fastapi

            _fastapi = fastapi
        except ImportError:
            raise ImportError(
                "FastAPI is required for web dashboard. "
                "Install with: pip install claudette-voice[web]"
            )
    return _fastapi


def _import_uvicorn():
    """Lazy import Uvicorn to avoid import errors if not installed."""
    global _uvicorn
    if _uvicorn is None:
        try:
            import uvicorn

            _uvicorn = uvicorn
        except ImportError:
            raise ImportError(
                "Uvicorn is required for web dashboard. "
                "Install with: pip install claudette-voice[web]"
            )
    return _uvicorn


class WebServer:
    """Web server for Claudette dashboard.

    Provides a web-based interface for:
    - Viewing current status and audio levels
    - Managing configuration settings
    - Browsing conversation history
    - Viewing skills and logs
    """

    def __init__(
        self,
        state_manager: Any,
        host: str = "127.0.0.1",
        port: int = 8420,
    ):
        """Initialize web server.

        Args:
            state_manager: ClaudetteStateManager instance
            host: Host to bind to (default: localhost only)
            port: Port to listen on (default: 8420)
        """
        self.state_manager = state_manager
        self.host = host
        self.port = port

        self._app = None
        self._server = None
        self._thread = None
        self._running = False
        self._loop = None

        # Connection manager for WebSockets
        self._ws_manager = None

    def _create_app(self):
        """Create and configure the FastAPI application."""
        fastapi = _import_fastapi()
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        app = fastapi.FastAPI(
            title="Claudette Dashboard",
            description="Web dashboard for Claudette voice assistant",
            version="0.1.0",
        )

        # Get static files directory
        static_dir = Path(__file__).parent / "static"

        # Import and include routes
        from .routes.api import create_api_router
        from .routes.websocket import create_ws_router, ConnectionManager

        # Create connection manager
        self._ws_manager = ConnectionManager()

        # Include API routes
        api_router = create_api_router(self.state_manager)
        app.include_router(api_router, prefix="/api")

        # Include WebSocket routes
        ws_router = create_ws_router(self._ws_manager, self.state_manager)
        app.include_router(ws_router)

        # Serve static files
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # Root route serves the dashboard
        @app.get("/")
        async def root():
            index_file = static_dir / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            return {"message": "Claudette Dashboard", "status": "ok"}

        # Health check endpoint
        @app.get("/health")
        async def health():
            return {"status": "ok", "service": "claudette-web"}

        return app

    def _setup_state_observers(self):
        """Set up observers to broadcast state changes via WebSocket."""

        def on_state_change(snapshot):
            if self._ws_manager and self._loop and self._running:
                try:
                    if not self._loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._ws_manager.broadcast(
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
                                }
                            ),
                            self._loop,
                        )
                except RuntimeError:
                    pass  # Loop closed, ignore

        def on_audio_level(level):
            if self._ws_manager and self._loop and self._running:
                try:
                    if not self._loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._ws_manager.broadcast({"type": "audio_level", "data": {"level": level}}),
                            self._loop,
                        )
                except RuntimeError:
                    pass  # Loop closed, ignore

        def on_log_entry(entry):
            if self._ws_manager and self._loop and self._running:
                try:
                    if not self._loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._ws_manager.broadcast(
                                {
                                    "type": "log",
                                    "data": {
                                        "timestamp": entry.timestamp,
                                        "level": entry.level,
                                        "message": entry.message,
                                    },
                                }
                            ),
                            self._loop,
                        )
                except RuntimeError:
                    pass  # Loop closed, ignore

        def on_claude_activity(activity):
            if self._ws_manager and self._loop and self._running:
                try:
                    if not self._loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._ws_manager.broadcast(
                                {
                                    "type": "claude_activity",
                                    "data": {
                                        "active": activity.active,
                                        "query": activity.query,
                                        "status": activity.status,
                                        "current_output": activity.current_output,
                                        "progress_lines": activity.progress_lines,
                                        "started_at": activity.started_at,
                                        "elapsed_seconds": activity.elapsed_seconds,
                                    },
                                }
                            ),
                            self._loop,
                        )
                except RuntimeError:
                    pass  # Loop closed, ignore

        self.state_manager.add_state_observer(on_state_change)
        self.state_manager.add_audio_observer(on_audio_level)
        self.state_manager.add_log_observer(on_log_entry)
        self.state_manager.add_claude_activity_observer(on_claude_activity)

    def _run_server(self):
        """Run the server in a thread."""
        uvicorn = _import_uvicorn()

        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Create app
        self._app = self._create_app()

        # Set up state observers
        self._setup_state_observers()

        # Configure uvicorn
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="warning",  # Reduce uvicorn logging noise
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        logger.info(f"Starting web server at http://{self.host}:{self.port}")

        # Run server
        self._running = True
        try:
            self._loop.run_until_complete(self._server.serve())
        except Exception as e:
            logger.error(f"Web server error: {e}")
        finally:
            self._running = False
            self._loop.close()

    def start(self):
        """Start the web server in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Web server already running")
            return

        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        # Wait briefly to ensure server starts
        import time

        time.sleep(0.5)

        if self._running:
            logger.info(f"Web dashboard available at http://{self.host}:{self.port}")
        else:
            logger.error("Web server failed to start")

    def stop(self):
        """Stop the web server."""
        if not self._running:
            return

        # Set running to False first to stop observers from broadcasting
        self._running = False

        if self._server:
            self._server.should_exit = True

        if self._thread:
            self._thread.join(timeout=5.0)

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"
