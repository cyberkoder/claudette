"""Tests for the notifications module."""

import pytest
from unittest.mock import patch, MagicMock
from claudette.notifications import NotificationManager


class TestNotificationManager:
    """Test the NotificationManager class."""

    def test_init_disabled(self):
        """Test initialization with notifications disabled."""
        manager = NotificationManager(enabled=False)
        assert manager.enabled is False

    @patch('claudette.notifications.PLYER_AVAILABLE', False)
    def test_init_no_plyer(self):
        """Test initialization when plyer is not available."""
        manager = NotificationManager(enabled=True)
        assert manager.enabled is False

    def test_notify_disabled(self):
        """Test that notify does nothing when disabled."""
        manager = NotificationManager(enabled=False)
        # Should not raise any errors
        manager.notify("Test", "Test message")

    @patch('claudette.notifications.PLYER_AVAILABLE', True)
    @patch('claudette.notifications.plyer_notification')
    def test_notify_enabled(self, mock_plyer):
        """Test notification when enabled."""
        manager = NotificationManager.__new__(NotificationManager)
        manager.enabled = True
        manager._app_name = "Claudette"
        
        manager.notify("Test Title", "Test Message", timeout=5)
        
        mock_plyer.notify.assert_called_once()
        call_kwargs = mock_plyer.notify.call_args[1]
        assert call_kwargs['title'] == "Test Title"
        assert call_kwargs['message'] == "Test Message"

    def test_notify_wake(self):
        """Test wake notification helper."""
        manager = NotificationManager(enabled=False)
        # Should not raise any errors
        manager.notify_wake()

    def test_notify_processing(self):
        """Test processing notification helper."""
        manager = NotificationManager(enabled=False)
        manager.notify_processing("test command")

    def test_notify_response(self):
        """Test response notification helper."""
        manager = NotificationManager(enabled=False)
        manager.notify_response("This is a test response")

    def test_notify_error(self):
        """Test error notification helper."""
        manager = NotificationManager(enabled=False)
        manager.notify_error("Test error message")

    def test_notify_started(self):
        """Test started notification helper."""
        manager = NotificationManager(enabled=False)
        manager.notify_started()

    def test_notify_shutdown(self):
        """Test shutdown notification helper."""
        manager = NotificationManager(enabled=False)
        manager.notify_shutdown()

    @patch('claudette.notifications.PLYER_AVAILABLE', True)
    @patch('claudette.notifications.plyer_notification')
    def test_notify_truncates_long_message(self, mock_plyer):
        """Test that long messages are truncated."""
        manager = NotificationManager.__new__(NotificationManager)
        manager.enabled = True
        manager._app_name = "Claudette"
        
        long_response = "A" * 200
        manager.notify_response(long_response)
        
        call_kwargs = mock_plyer.notify.call_args[1]
        assert len(call_kwargs['message']) <= 103  # 100 chars + "..."
