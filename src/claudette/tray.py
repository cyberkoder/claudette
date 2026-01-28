"""
System tray application for Claudette.

Provides a system tray icon with status indicator and optional waveform visualization.
"""

import logging
import threading
from pathlib import Path
from typing import Callable, Optional
import io

logger = logging.getLogger("claudette")

# Try to import system tray libraries
try:
    import pystray
    from PIL import Image, ImageDraw

    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    pystray = None
    Image = None
    ImageDraw = None


class TrayIcon:
    """System tray icon for Claudette."""

    # State colors
    COLORS = {
        "idle": "#4a90d9",  # Blue - listening
        "listening": "#4a90d9",  # Blue - actively listening
        "recording": "#e74c3c",  # Red - recording speech
        "processing": "#f39c12",  # Orange - processing
        "speaking": "#27ae60",  # Green - speaking
        "error": "#c0392b",  # Dark red - error
    }

    def __init__(
        self,
        enabled: bool = True,
        on_activate: Optional[Callable] = None,
        on_quit: Optional[Callable] = None,
    ):
        self.enabled = enabled and TRAY_AVAILABLE
        self.on_activate = on_activate
        self.on_quit = on_quit
        self._icon: Optional["pystray.Icon"] = None
        self._state = "idle"
        self._thread: Optional[threading.Thread] = None

        if enabled and not TRAY_AVAILABLE:
            logger.warning(
                "System tray disabled: pystray or PIL not installed. "
                "Install with: pip install pystray pillow"
            )

    def _create_icon_image(self, color: str, size: int = 64) -> "Image.Image":
        """Create a tray icon image with the given color."""
        # Create a circular icon with the state color
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw outer circle
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin], fill=color, outline="#ffffff", width=2
        )

        # Draw "C" letter in center
        font_size = size // 2
        try:
            from PIL import ImageFont

            # Try to use a nice font, fall back to default
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
                )
            except Exception:
                font = ImageFont.load_default()
        except Exception:
            font = None

        # Draw centered "C"
        text = "C"
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = font_size // 2, font_size
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - 2
        draw.text((x, y), text, fill="#ffffff", font=font)

        return image

    def _create_menu(self) -> "pystray.Menu":
        """Create the tray context menu."""
        return pystray.Menu(
            pystray.MenuItem("Claudette", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Activate", self._on_activate_clicked),
            pystray.MenuItem("Status", self._on_status_clicked),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit_clicked),
        )

    def _on_activate_clicked(self, icon, item):
        """Handle activate menu click."""
        if self.on_activate:
            threading.Thread(target=self.on_activate, daemon=True).start()

    def _on_status_clicked(self, icon, item):
        """Handle status menu click."""
        # Could show a status window
        logger.info(f"Tray status: {self._state}")

    def _on_quit_clicked(self, icon, item):
        """Handle quit menu click."""
        if self.on_quit:
            self.on_quit()
        self.stop()

    def set_state(self, state: str):
        """Update the tray icon state."""
        if not self.enabled or not self._icon:
            return

        self._state = state
        color = self.COLORS.get(state, self.COLORS["idle"])

        try:
            self._icon.icon = self._create_icon_image(color)
            self._icon.title = f"Claudette - {state.capitalize()}"
        except Exception as e:
            logger.debug(f"Failed to update tray icon: {e}")

    def start(self):
        """Start the system tray icon."""
        if not self.enabled or not TRAY_AVAILABLE:
            return

        try:
            # Create the icon
            image = self._create_icon_image(self.COLORS["idle"])
            menu = self._create_menu()

            self._icon = pystray.Icon("claudette", image, "Claudette - Listening", menu)

            # Run in background thread
            self._thread = threading.Thread(target=self._icon.run, daemon=True)
            self._thread.start()
            logger.info("System tray icon started")

        except Exception as e:
            logger.error(f"Failed to start system tray: {e}")
            self.enabled = False

    def stop(self):
        """Stop the system tray icon."""
        if self._icon:
            try:
                self._icon.stop()
                logger.info("System tray icon stopped")
            except Exception as e:
                logger.debug(f"Error stopping tray icon: {e}")


class WaveformWindow:
    """Optional floating window with audio waveform visualization."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._window = None
        self._canvas = None
        self._running = False

        # Check for tkinter
        if enabled:
            try:
                import tkinter as tk

                self._tk = tk
            except ImportError:
                logger.warning("Waveform window disabled: tkinter not available")
                self.enabled = False
                self._tk = None

    def start(self):
        """Start the waveform window."""
        if not self.enabled or not self._tk:
            return

        def run_window():
            self._window = self._tk.Tk()
            self._window.title("Claudette")
            self._window.geometry("300x80")
            self._window.attributes("-topmost", True)
            self._window.configure(bg="#1a1a2e")

            # Create canvas for waveform
            self._canvas = self._tk.Canvas(
                self._window, width=280, height=60, bg="#1a1a2e", highlightthickness=0
            )
            self._canvas.pack(pady=10)

            # Draw initial flat line
            self._draw_waveform([0] * 50)

            self._running = True
            self._window.mainloop()

        self._thread = threading.Thread(target=run_window, daemon=True)
        self._thread.start()
        logger.info("Waveform window started")

    def _draw_waveform(self, samples: list):
        """Draw waveform on canvas."""
        if not self._canvas or not self._running:
            return

        try:
            self._canvas.delete("all")
            width = 280
            height = 60
            center_y = height // 2

            # Draw waveform line
            points = []
            for i, sample in enumerate(samples):
                x = int(i * width / len(samples))
                y = int(center_y - sample * center_y * 0.8)
                points.extend([x, y])

            if len(points) >= 4:
                self._canvas.create_line(points, fill="#4a90d9", width=2, smooth=True)
        except Exception:
            pass

    def update_waveform(self, audio_samples):
        """Update waveform with new audio samples."""
        if not self.enabled or not self._running:
            return

        try:
            import numpy as np

            # Downsample to 50 points for display
            if len(audio_samples) > 50:
                indices = np.linspace(0, len(audio_samples) - 1, 50, dtype=int)
                samples = audio_samples[indices]
            else:
                samples = audio_samples

            # Normalize
            max_val = np.max(np.abs(samples)) or 1
            normalized = (samples / max_val).tolist()

            # Schedule update on main thread
            if self._window:
                self._window.after(0, lambda: self._draw_waveform(normalized))
        except Exception as e:
            logger.debug(f"Waveform update error: {e}")

    def stop(self):
        """Stop the waveform window."""
        self._running = False
        if self._window:
            try:
                self._window.quit()
            except Exception:
                pass
