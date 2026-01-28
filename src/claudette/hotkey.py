"""
Hotkey support for Claudette.

Allows triggering voice commands via keyboard shortcuts.
"""

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger("claudette")

# Try to import pynput for cross-platform hotkey support
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    keyboard = None


class HotkeyManager:
    """Manages global hotkey bindings for Claudette."""

    def __init__(
        self,
        enabled: bool = True,
        hotkey: str = "<ctrl>+<shift>+c",
        callback: Optional[Callable] = None
    ):
        self.enabled = enabled and PYNPUT_AVAILABLE
        self.hotkey = hotkey
        self.callback = callback
        self._listener: Optional["keyboard.GlobalHotKeys"] = None
        self._running = False

        if enabled and not PYNPUT_AVAILABLE:
            logger.warning("Hotkey support disabled: pynput not installed. "
                          "Install with: pip install pynput")

    def _parse_hotkey(self, hotkey_str: str) -> set:
        """Parse hotkey string into pynput key set."""
        if not PYNPUT_AVAILABLE:
            return set()

        parts = hotkey_str.lower().split("+")
        keys = set()

        for part in parts:
            part = part.strip()
            if part in ("<ctrl>", "ctrl", "control"):
                keys.add(keyboard.Key.ctrl)
            elif part in ("<shift>", "shift"):
                keys.add(keyboard.Key.shift)
            elif part in ("<alt>", "alt"):
                keys.add(keyboard.Key.alt)
            elif part in ("<cmd>", "cmd", "command", "super"):
                keys.add(keyboard.Key.cmd)
            elif part.startswith("<") and part.endswith(">"):
                # Try to get special key like <f12>
                key_name = part[1:-1]
                try:
                    keys.add(getattr(keyboard.Key, key_name))
                except AttributeError:
                    logger.warning(f"Unknown special key: {part}")
            elif len(part) == 1:
                # Regular character key
                keys.add(keyboard.KeyCode.from_char(part))
            else:
                logger.warning(f"Unknown key in hotkey: {part}")

        return keys

    def _on_hotkey(self):
        """Called when hotkey is pressed."""
        if self.callback:
            logger.info(f"Hotkey pressed: {self.hotkey}")
            # Run callback in a separate thread to not block the listener
            threading.Thread(target=self.callback, daemon=True).start()

    def start(self):
        """Start listening for hotkeys."""
        if not self.enabled or not PYNPUT_AVAILABLE:
            return

        try:
            # Create hotkey listener
            self._listener = keyboard.GlobalHotKeys({
                self.hotkey: self._on_hotkey
            })
            self._listener.start()
            self._running = True
            logger.info(f"Hotkey listener started: {self.hotkey}")
        except Exception as e:
            logger.error(f"Failed to start hotkey listener: {e}")
            self.enabled = False

    def stop(self):
        """Stop listening for hotkeys."""
        if self._listener:
            self._listener.stop()
            self._running = False
            logger.info("Hotkey listener stopped")

    @property
    def is_running(self) -> bool:
        """Check if hotkey listener is active."""
        return self._running


def get_default_hotkey() -> str:
    """Get platform-appropriate default hotkey."""
    import platform
    system = platform.system()

    if system == "Darwin":  # macOS
        return "<cmd>+<shift>+c"
    else:  # Linux/Windows
        return "<ctrl>+<shift>+c"
