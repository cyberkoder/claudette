"""
Desktop notifications for Claudette.

Provides system notifications for important events.
"""

import logging

logger = logging.getLogger("claudette")

# Try to import notification libraries
try:
    from plyer import notification as plyer_notification

    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    plyer_notification = None


class NotificationManager:
    """Manages desktop notifications for Claudette."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PLYER_AVAILABLE
        self._app_name = "Claudette"

        if enabled and not PLYER_AVAILABLE:
            logger.warning(
                "Desktop notifications disabled: plyer not installed. "
                "Install with: pip install plyer"
            )

    def notify(self, title: str, message: str, timeout: int = 5, icon_path: str | None = None):
        """Show a desktop notification.

        Args:
            title: Notification title
            message: Notification body text
            timeout: Time in seconds to show notification
            icon_path: Optional path to notification icon
        """
        if not self.enabled:
            return

        try:
            plyer_notification.notify(
                title=title,
                message=message,
                app_name=self._app_name,
                timeout=timeout,
                app_icon=icon_path,
            )
            logger.debug(f"Notification shown: {title}")
        except Exception as e:
            logger.debug(f"Failed to show notification: {e}")

    def notify_wake(self):
        """Notify that wake word was detected."""
        self.notify("Claudette Activated", "Listening for your command...", timeout=3)

    def notify_processing(self, command: str):
        """Notify that a command is being processed."""
        self.notify(
            "Processing Command",
            f"Working on: {command[:50]}{'...' if len(command) > 50 else ''}",
            timeout=5,
        )

    def notify_response(self, response: str):
        """Notify with Claudette's response."""
        self.notify(
            "Claudette", response[:100] + ("..." if len(response) > 100 else ""), timeout=10
        )

    def notify_error(self, error: str):
        """Notify about an error."""
        self.notify("Claudette Error", error, timeout=10)

    def notify_started(self):
        """Notify that Claudette has started."""
        self.notify("Claudette Ready", "Voice assistant is now listening.", timeout=5)

    def notify_shutdown(self):
        """Notify that Claudette is shutting down."""
        self.notify("Claudette", "Shutting down. Goodbye!", timeout=3)
