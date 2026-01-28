"""
Offline fallback support for Claudette.

Provides basic functionality when network is unavailable.
"""

import logging
import socket
from typing import Optional

logger = logging.getLogger("claudette")


def check_network(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    """Check if network is available.

    Args:
        host: Host to check connectivity (default: Google DNS)
        port: Port to check
        timeout: Connection timeout in seconds

    Returns:
        True if network is available, False otherwise
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.error, socket.timeout):
        return False


class OfflineFallback:
    """Handles offline mode responses when Claude is unavailable."""

    # Canned responses for offline mode
    OFFLINE_RESPONSES = {
        "greeting": [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening"
        ],
        "farewell": [
            "goodbye", "bye", "see you", "later", "good night"
        ],
        "thanks": [
            "thank you", "thanks", "appreciate"
        ],
        "status": [
            "how are you", "are you there", "you okay"
        ],
    }

    OFFLINE_REPLIES = {
        "greeting": "Hello, sir. I'm currently offline but still at your service for basic commands.",
        "farewell": "Goodbye, sir. I'll be here when you return.",
        "thanks": "You're most welcome, sir.",
        "status": "I'm operational, sir, though currently without network access. Basic commands are still available.",
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._was_offline = False
        self._network_available = True

    def check_and_notify(self) -> bool:
        """Check network status and log changes.

        Returns:
            True if network is available, False otherwise
        """
        is_online = check_network()

        if is_online != self._network_available:
            if is_online:
                logger.info("Network connection restored")
                self._was_offline = True
            else:
                logger.warning("Network connection lost - entering offline mode")

        self._network_available = is_online
        return is_online

    def is_online(self) -> bool:
        """Check if currently online."""
        return check_network()

    def was_recently_offline(self) -> bool:
        """Check if we were recently offline (and just came back)."""
        if self._was_offline:
            self._was_offline = False
            return True
        return False

    def get_offline_response(self, command: str) -> Optional[str]:
        """Get a canned response for offline mode.

        Args:
            command: User's command

        Returns:
            Appropriate offline response, or None if no match
        """
        if not self.enabled:
            return None

        command_lower = command.lower()

        for category, triggers in self.OFFLINE_RESPONSES.items():
            for trigger in triggers:
                if trigger in command_lower:
                    return self.OFFLINE_REPLIES.get(category)

        # Generic offline message for unmatched commands
        return "I'm currently offline, sir. I can still help with basic commands like time, date, and system status, but I'll need network access for more complex requests."

    def get_reconnected_message(self) -> str:
        """Get message for when connection is restored."""
        return "Network connection restored, sir. Full functionality is now available."
