"""Tests for the tray module."""

import pytest
from unittest.mock import patch, MagicMock
from claudette.tray import TrayIcon, WaveformWindow


class TestTrayIcon:
    """Test the TrayIcon class."""

    def test_init_disabled(self):
        """Test initialization with tray disabled."""
        tray = TrayIcon(enabled=False)
        assert tray.enabled is False

    @patch('claudette.tray.TRAY_AVAILABLE', False)
    def test_init_no_pystray(self):
        """Test initialization when pystray is not available."""
        tray = TrayIcon(enabled=True)
        assert tray.enabled is False

    def test_callbacks_set(self):
        """Test that callbacks are properly set."""
        on_activate = MagicMock()
        on_quit = MagicMock()
        
        tray = TrayIcon(enabled=False, on_activate=on_activate, on_quit=on_quit)
        
        assert tray.on_activate == on_activate
        assert tray.on_quit == on_quit

    def test_state_colors_defined(self):
        """Test that all state colors are defined."""
        expected_states = ["idle", "listening", "recording", "processing", "speaking", "error"]
        
        for state in expected_states:
            assert state in TrayIcon.COLORS

    def test_set_state_disabled(self):
        """Test set_state when disabled."""
        tray = TrayIcon(enabled=False)
        # Should not raise any errors
        tray.set_state("listening")
        tray.set_state("recording")
        tray.set_state("invalid_state")

    def test_start_disabled(self):
        """Test start when disabled."""
        tray = TrayIcon(enabled=False)
        # Should not raise any errors
        tray.start()

    def test_stop_when_not_started(self):
        """Test stop when not started."""
        tray = TrayIcon(enabled=False)
        # Should not raise any errors
        tray.stop()

    @patch('claudette.tray.TRAY_AVAILABLE', True)
    @patch('claudette.tray.Image')
    @patch('claudette.tray.ImageDraw')
    def test_create_icon_image(self, mock_draw, mock_image):
        """Test icon image creation."""
        mock_img = MagicMock()
        mock_image.new.return_value = mock_img
        mock_draw.Draw.return_value = MagicMock()
        
        tray = TrayIcon.__new__(TrayIcon)
        tray.enabled = True
        
        result = tray._create_icon_image("#ff0000", size=64)
        
        mock_image.new.assert_called_once()
        assert result == mock_img


class TestWaveformWindow:
    """Test the WaveformWindow class."""

    def test_init_disabled(self):
        """Test initialization with window disabled."""
        window = WaveformWindow(enabled=False)
        assert window.enabled is False

    def test_init_no_tkinter(self):
        """Test initialization when tkinter is not available."""
        with patch.dict('sys.modules', {'tkinter': None}):
            window = WaveformWindow(enabled=True)
            # Should gracefully handle missing tkinter

    def test_start_disabled(self):
        """Test start when disabled."""
        window = WaveformWindow(enabled=False)
        # Should not raise any errors
        window.start()

    def test_stop_when_not_started(self):
        """Test stop when not started."""
        window = WaveformWindow(enabled=False)
        # Should not raise any errors
        window.stop()

    def test_update_waveform_disabled(self):
        """Test update_waveform when disabled."""
        import numpy as np
        
        window = WaveformWindow(enabled=False)
        # Should not raise any errors
        window.update_waveform(np.zeros(100))
