"""Tests for the offline fallback module."""

import pytest
from unittest.mock import patch
from claudette.offline import OfflineFallback, check_network


class TestCheckNetwork:
    """Test the network checking function."""

    @patch('socket.socket')
    def test_network_available(self, mock_socket):
        """Test when network is available."""
        mock_socket.return_value.connect.return_value = None
        
        result = check_network(timeout=1.0)
        assert result is True

    @patch('socket.socket')
    def test_network_unavailable(self, mock_socket):
        """Test when network is unavailable."""
        import socket
        mock_socket.return_value.connect.side_effect = socket.error("No connection")
        
        result = check_network(timeout=1.0)
        assert result is False


class TestOfflineFallback:
    """Test the OfflineFallback class."""

    def test_init_enabled(self):
        """Test initialization with enabled=True."""
        fallback = OfflineFallback(enabled=True)
        assert fallback.enabled is True

    def test_init_disabled(self):
        """Test initialization with enabled=False."""
        fallback = OfflineFallback(enabled=False)
        assert fallback.enabled is False

    def test_get_offline_response_greeting(self):
        """Test offline response for greetings."""
        fallback = OfflineFallback(enabled=True)
        
        response = fallback.get_offline_response("hello there")
        assert response is not None
        assert "offline" in response.lower()

    def test_get_offline_response_farewell(self):
        """Test offline response for farewells."""
        fallback = OfflineFallback(enabled=True)
        
        response = fallback.get_offline_response("goodbye")
        assert response is not None

    def test_get_offline_response_thanks(self):
        """Test offline response for thanks."""
        fallback = OfflineFallback(enabled=True)
        
        response = fallback.get_offline_response("thank you")
        assert response is not None
        assert "welcome" in response.lower()

    def test_get_offline_response_status(self):
        """Test offline response for status."""
        fallback = OfflineFallback(enabled=True)
        
        response = fallback.get_offline_response("how are you")
        assert response is not None
        assert "operational" in response.lower() or "offline" in response.lower()

    def test_get_offline_response_unknown(self):
        """Test offline response for unknown command."""
        fallback = OfflineFallback(enabled=True)
        
        response = fallback.get_offline_response("calculate the square root of 42")
        assert response is not None
        assert "offline" in response.lower()

    def test_get_offline_response_disabled(self):
        """Test that disabled fallback returns None."""
        fallback = OfflineFallback(enabled=False)
        
        response = fallback.get_offline_response("hello")
        assert response is None

    def test_get_reconnected_message(self):
        """Test reconnection message."""
        fallback = OfflineFallback(enabled=True)
        
        message = fallback.get_reconnected_message()
        assert "restored" in message.lower() or "connection" in message.lower()

    def test_was_recently_offline(self):
        """Test the was_recently_offline flag."""
        fallback = OfflineFallback(enabled=True)
        
        # Initially should be False
        assert fallback.was_recently_offline() is False
        
        # Simulate going offline then online
        fallback._was_offline = True
        assert fallback.was_recently_offline() is True
        
        # Should reset after checking
        assert fallback.was_recently_offline() is False
