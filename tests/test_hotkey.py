"""Tests for the hotkey module."""

import pytest
from unittest.mock import patch, MagicMock
from claudette.hotkey import HotkeyManager, get_default_hotkey


class TestGetDefaultHotkey:
    """Test the default hotkey function."""

    @patch('platform.system')
    def test_macos_default(self, mock_system):
        """Test default hotkey on macOS."""
        mock_system.return_value = "Darwin"
        
        hotkey = get_default_hotkey()
        assert "<cmd>" in hotkey
        assert "<shift>" in hotkey
        assert "c" in hotkey

    @patch('platform.system')
    def test_linux_default(self, mock_system):
        """Test default hotkey on Linux."""
        mock_system.return_value = "Linux"
        
        hotkey = get_default_hotkey()
        assert "<ctrl>" in hotkey
        assert "<shift>" in hotkey
        assert "c" in hotkey

    @patch('platform.system')
    def test_windows_default(self, mock_system):
        """Test default hotkey on Windows."""
        mock_system.return_value = "Windows"
        
        hotkey = get_default_hotkey()
        assert "<ctrl>" in hotkey
        assert "<shift>" in hotkey


class TestHotkeyManager:
    """Test the HotkeyManager class."""

    def test_init_disabled(self):
        """Test initialization with hotkeys disabled."""
        manager = HotkeyManager(enabled=False)
        assert manager.enabled is False

    def test_init_no_pynput(self):
        """Test initialization when pynput is not available."""
        with patch.dict('sys.modules', {'pynput': None}):
            manager = HotkeyManager(enabled=True)
            # Should gracefully disable itself
            assert manager._listener is None

    def test_callback_set(self):
        """Test that callback is properly set."""
        callback = MagicMock()
        manager = HotkeyManager(enabled=False, callback=callback)
        
        assert manager.callback == callback

    def test_hotkey_stored(self):
        """Test that hotkey string is stored."""
        manager = HotkeyManager(enabled=False, hotkey="<ctrl>+<alt>+x")
        assert manager.hotkey == "<ctrl>+<alt>+x"

    def test_is_running_false_initially(self):
        """Test that manager is not running initially."""
        manager = HotkeyManager(enabled=False)
        assert manager.is_running is False

    def test_stop_when_not_started(self):
        """Test that stop doesn't fail when not started."""
        manager = HotkeyManager(enabled=False)
        # Should not raise any errors
        manager.stop()

    @patch('claudette.hotkey.PYNPUT_AVAILABLE', False)
    def test_pynput_not_available_warning(self):
        """Test warning when pynput is not installed."""
        manager = HotkeyManager(enabled=True)
        assert manager.enabled is False
