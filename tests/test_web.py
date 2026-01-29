"""
Tests for Claudette Web Dashboard
"""

import pytest
import time
from unittest.mock import MagicMock, patch


class TestClaudetteStateManager:
    """Tests for the state manager."""

    def test_state_manager_creation(self):
        """Test state manager can be created."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        assert manager is not None

    def test_state_snapshot(self):
        """Test getting state snapshot."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        snapshot = manager.get_snapshot()

        assert snapshot.state == "idle"
        assert snapshot.conversation_mode is False
        assert snapshot.awaiting_confirmation is False
        assert snapshot.audio_level == 0.0
        assert snapshot.uptime_seconds >= 0

    def test_update_state(self):
        """Test updating state."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        manager.update_state("Recording...")

        snapshot = manager.get_snapshot()
        assert snapshot.state == "Recording..."

    def test_update_audio_level(self):
        """Test updating audio level."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()

        # Test normal range
        manager.update_audio_level(0.5)
        assert manager.get_snapshot().audio_level == 0.5

        # Test clamping to max
        manager.update_audio_level(1.5)
        assert manager.get_snapshot().audio_level == 1.0

        # Test clamping to min
        manager.update_audio_level(-0.5)
        assert manager.get_snapshot().audio_level == 0.0

    def test_update_conversation_mode(self):
        """Test updating conversation mode."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        manager.update_conversation_mode(True)

        snapshot = manager.get_snapshot()
        assert snapshot.conversation_mode is True

    def test_update_transcription(self):
        """Test updating last transcription."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        manager.update_last_transcription("Hello Claudette")

        snapshot = manager.get_snapshot()
        assert snapshot.last_transcription == "Hello Claudette"

    def test_update_response(self):
        """Test updating last response."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        manager.update_last_response("Yes, sir?")

        snapshot = manager.get_snapshot()
        assert snapshot.last_response == "Yes, sir?"

    def test_state_observers(self):
        """Test state observer notifications."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        received_snapshots = []

        def observer(snapshot):
            received_snapshots.append(snapshot)

        manager.add_state_observer(observer)
        manager.update_state("Testing")

        assert len(received_snapshots) == 1
        assert received_snapshots[0].state == "Testing"

    def test_remove_state_observer(self):
        """Test removing state observer."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        received_snapshots = []

        def observer(snapshot):
            received_snapshots.append(snapshot)

        manager.add_state_observer(observer)
        manager.remove_state_observer(observer)
        manager.update_state("Testing")

        assert len(received_snapshots) == 0

    def test_audio_observers(self):
        """Test audio level observer notifications."""
        from src.claudette.web.state import ClaudetteStateManager

        manager = ClaudetteStateManager()
        received_levels = []

        def observer(level):
            received_levels.append(level)

        manager.add_audio_observer(observer)
        manager.update_audio_level(0.75)

        assert len(received_levels) == 1
        assert received_levels[0] == 0.75

    def test_recent_logs(self):
        """Test log entry storage."""
        from src.claudette.web.state import ClaudetteStateManager, LogEntry

        manager = ClaudetteStateManager()

        # Manually add log entries
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00",
            level="INFO",
            message="Test message"
        )
        manager._add_log_entry(entry)

        logs = manager.get_recent_logs(limit=10)
        assert len(logs) >= 1


class TestWebServer:
    """Tests for the web server."""

    @pytest.fixture
    def state_manager(self):
        """Create a state manager for testing."""
        from src.claudette.web.state import ClaudetteStateManager
        return ClaudetteStateManager()

    def test_server_creation(self, state_manager):
        """Test web server can be created."""
        from src.claudette.web.server import WebServer

        server = WebServer(state_manager, host="127.0.0.1", port=18420)
        assert server is not None
        assert server.host == "127.0.0.1"
        assert server.port == 18420

    def test_server_url(self, state_manager):
        """Test server URL property."""
        from src.claudette.web.server import WebServer

        server = WebServer(state_manager, host="127.0.0.1", port=18420)
        assert server.url == "http://127.0.0.1:18420"

    def test_server_not_running_initially(self, state_manager):
        """Test server is not running before start."""
        from src.claudette.web.server import WebServer

        server = WebServer(state_manager, host="127.0.0.1", port=18420)
        assert server.is_running is False


class TestAPIEndpoints:
    """Tests for API endpoints using TestClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from src.claudette.web.state import ClaudetteStateManager
        from src.claudette.web.routes.api import create_api_router

        # Create a mock FastAPI app for testing
        from fastapi import FastAPI
        app = FastAPI()
        state_manager = ClaudetteStateManager()
        router = create_api_router(state_manager)
        app.include_router(router, prefix="/api")

        return TestClient(app)

    def test_status_endpoint(self, client):
        """Test /api/status endpoint."""
        response = client.get("/api/status")
        assert response.status_code == 200

        data = response.json()
        assert "state" in data
        assert "conversation_mode" in data
        assert "audio_level" in data
        assert "uptime_seconds" in data

    def test_voices_endpoint(self, client):
        """Test /api/voices endpoint."""
        response = client.get("/api/voices")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Check first voice has expected fields
        voice = data[0]
        assert "id" in voice
        assert "name" in voice
        assert "language" in voice

    def test_personalities_endpoint(self, client):
        """Test /api/personalities endpoint."""
        response = client.get("/api/personalities")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Check claudette personality exists
        names = [p["name"] for p in data]
        assert "claudette" in names

    def test_system_endpoint(self, client):
        """Test /api/system endpoint."""
        response = client.get("/api/system")
        assert response.status_code == 200

        data = response.json()
        assert "platform" in data
        assert "python_version" in data
        assert "gpu_available" in data

    def test_logs_endpoint(self, client):
        """Test /api/logs endpoint."""
        response = client.get("/api/logs")
        assert response.status_code == 200

        data = response.json()
        assert "entries" in data
        assert isinstance(data["entries"], list)

    def test_skills_endpoint(self, client):
        """Test /api/skills endpoint - returns empty without Claudette."""
        response = client.get("/api/skills")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_history_endpoint(self, client):
        """Test /api/history endpoint."""
        response = client.get("/api/history")
        assert response.status_code == 200

        data = response.json()
        assert "entries" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_config_endpoint_without_claudette(self, client):
        """Test /api/config returns 503 without Claudette."""
        response = client.get("/api/config")
        # Without Claudette instance, should return 503
        assert response.status_code == 503


class TestConnectionManager:
    """Tests for WebSocket connection manager."""

    def test_connection_manager_creation(self):
        """Test connection manager can be created."""
        from src.claudette.web.routes.websocket import ConnectionManager

        manager = ConnectionManager()
        assert manager is not None
        assert manager.connection_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
