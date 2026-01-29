"""
Thread-safe state manager for Claudette web dashboard.

Provides centralized state management with observer pattern for real-time updates.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("claudette.web")


@dataclass
class StateSnapshot:
    """Immutable snapshot of Claudette's current state."""

    state: str = "idle"
    conversation_mode: bool = False
    awaiting_confirmation: bool = False
    audio_level: float = 0.0
    last_transcription: str = ""
    last_response: str = ""
    uptime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LogEntry:
    """A log entry for the web dashboard."""

    timestamp: str
    level: str
    message: str


@dataclass
class ClaudeActivity:
    """Tracks Claude CLI backend activity."""

    active: bool = False
    query: str = ""
    status: str = ""  # "thinking", "searching", "reading", "writing", "done"
    current_output: str = ""
    progress_lines: list = field(default_factory=list)
    started_at: str = ""
    elapsed_seconds: float = 0.0


class ClaudetteStateManager:
    """Thread-safe state manager with observer pattern for real-time updates."""

    def __init__(self, claudette: Any = None):
        """Initialize state manager.

        Args:
            claudette: Reference to the Claudette instance (set later during integration)
        """
        self._claudette = claudette
        self._lock = threading.RLock()
        self._start_time = time.time()

        # Current state
        self._state = "idle"
        self._conversation_mode = False
        self._awaiting_confirmation = False
        self._audio_level = 0.0
        self._last_transcription = ""
        self._last_response = ""

        # Claude CLI activity tracking
        self._claude_active = False
        self._claude_query = ""
        self._claude_status = ""
        self._claude_output = ""
        self._claude_progress: list[str] = []
        self._claude_started_at = ""
        self._claude_start_time = 0.0

        # Observers for state changes
        self._state_observers: list[Callable[[StateSnapshot], None]] = []
        self._audio_observers: list[Callable[[float], None]] = []
        self._log_observers: list[Callable[[LogEntry], None]] = []
        self._claude_activity_observers: list[Callable[[ClaudeActivity], None]] = []

        # Recent logs buffer
        self._recent_logs: deque[LogEntry] = deque(maxlen=100)

        # Log handler integration
        self._setup_log_handler()

    def _setup_log_handler(self):
        """Set up log handler to capture Claudette logs."""

        class WebLogHandler(logging.Handler):
            def __init__(handler_self, state_manager: "ClaudetteStateManager"):
                super().__init__()
                handler_self.state_manager = state_manager
                handler_self._in_emit = False  # Prevent recursion

            def emit(handler_self, record: logging.LogRecord):
                # Prevent recursion - don't capture web module logs
                if handler_self._in_emit or record.name.startswith("claudette.web"):
                    return
                handler_self._in_emit = True
                try:
                    entry = LogEntry(
                        timestamp=datetime.fromtimestamp(record.created).isoformat(),
                        level=record.levelname,
                        message=record.getMessage(),
                    )
                    handler_self.state_manager._add_log_entry(entry)
                finally:
                    handler_self._in_emit = False

        # Add handler to claudette logger
        handler = WebLogHandler(self)
        handler.setLevel(logging.DEBUG)
        logging.getLogger("claudette").addHandler(handler)

    def set_claudette(self, claudette: Any):
        """Set the Claudette instance reference."""
        with self._lock:
            self._claudette = claudette

    @property
    def claudette(self) -> Any:
        """Get the Claudette instance."""
        with self._lock:
            return self._claudette

    # State update methods

    def update_state(self, state: str):
        """Update the current voice state."""
        with self._lock:
            self._state = state
        self._notify_state_observers()

    def update_conversation_mode(self, active: bool):
        """Update conversation mode status."""
        with self._lock:
            self._conversation_mode = active
        self._notify_state_observers()

    def update_awaiting_confirmation(self, awaiting: bool):
        """Update awaiting confirmation status."""
        with self._lock:
            self._awaiting_confirmation = awaiting
        self._notify_state_observers()

    def update_audio_level(self, level: float):
        """Update current audio level (0.0-1.0)."""
        with self._lock:
            self._audio_level = max(0.0, min(1.0, level))
        self._notify_audio_observers()

    def update_last_transcription(self, text: str):
        """Update the last transcription."""
        with self._lock:
            self._last_transcription = text
        self._notify_state_observers()

    def update_last_response(self, text: str):
        """Update the last response."""
        with self._lock:
            self._last_response = text
        self._notify_state_observers()

    # Claude activity methods

    def start_claude_activity(self, query: str):
        """Start tracking Claude CLI activity."""
        with self._lock:
            self._claude_active = True
            self._claude_query = query
            self._claude_status = "thinking"
            self._claude_output = ""
            self._claude_progress = []
            self._claude_started_at = datetime.now().isoformat()
            self._claude_start_time = time.time()
        self._notify_claude_activity_observers()

    def update_claude_status(self, status: str):
        """Update Claude's current status (thinking, searching, reading, writing, done)."""
        with self._lock:
            self._claude_status = status
        self._notify_claude_activity_observers()

    def update_claude_output(self, output: str):
        """Update Claude's current output text."""
        with self._lock:
            self._claude_output = output
        self._notify_claude_activity_observers()

    def add_claude_progress(self, line: str):
        """Add a progress line from Claude."""
        with self._lock:
            self._claude_progress.append(line)
            # Keep last 50 lines
            if len(self._claude_progress) > 50:
                self._claude_progress = self._claude_progress[-50:]
        self._notify_claude_activity_observers()

    def end_claude_activity(self):
        """End Claude CLI activity tracking."""
        with self._lock:
            self._claude_active = False
            self._claude_status = "done"
        self._notify_claude_activity_observers()

    def get_claude_activity(self) -> ClaudeActivity:
        """Get current Claude activity state."""
        with self._lock:
            elapsed = time.time() - self._claude_start_time if self._claude_active else 0.0
            return ClaudeActivity(
                active=self._claude_active,
                query=self._claude_query,
                status=self._claude_status,
                current_output=self._claude_output,
                progress_lines=list(self._claude_progress),
                started_at=self._claude_started_at,
                elapsed_seconds=elapsed,
            )

    # State getters

    def get_snapshot(self) -> StateSnapshot:
        """Get an immutable snapshot of the current state."""
        with self._lock:
            return StateSnapshot(
                state=self._state,
                conversation_mode=self._conversation_mode,
                awaiting_confirmation=self._awaiting_confirmation,
                audio_level=self._audio_level,
                last_transcription=self._last_transcription,
                last_response=self._last_response,
                uptime_seconds=time.time() - self._start_time,
            )

    def get_recent_logs(self, limit: int = 50) -> list[LogEntry]:
        """Get recent log entries."""
        with self._lock:
            logs = list(self._recent_logs)
        return logs[-limit:]

    # Observer management

    def add_state_observer(self, callback: Callable[[StateSnapshot], None]):
        """Add observer for state changes."""
        with self._lock:
            self._state_observers.append(callback)

    def remove_state_observer(self, callback: Callable[[StateSnapshot], None]):
        """Remove state observer."""
        with self._lock:
            if callback in self._state_observers:
                self._state_observers.remove(callback)

    def add_audio_observer(self, callback: Callable[[float], None]):
        """Add observer for audio level changes."""
        with self._lock:
            self._audio_observers.append(callback)

    def remove_audio_observer(self, callback: Callable[[float], None]):
        """Remove audio observer."""
        with self._lock:
            if callback in self._audio_observers:
                self._audio_observers.remove(callback)

    def add_log_observer(self, callback: Callable[[LogEntry], None]):
        """Add observer for new log entries."""
        with self._lock:
            self._log_observers.append(callback)

    def remove_log_observer(self, callback: Callable[[LogEntry], None]):
        """Remove log observer."""
        with self._lock:
            if callback in self._log_observers:
                self._log_observers.remove(callback)

    def add_claude_activity_observer(self, callback: Callable[[ClaudeActivity], None]):
        """Add observer for Claude activity updates."""
        with self._lock:
            self._claude_activity_observers.append(callback)

    def remove_claude_activity_observer(self, callback: Callable[[ClaudeActivity], None]):
        """Remove Claude activity observer."""
        with self._lock:
            if callback in self._claude_activity_observers:
                self._claude_activity_observers.remove(callback)

    # Notification methods

    def _notify_state_observers(self):
        """Notify all state observers."""
        snapshot = self.get_snapshot()
        with self._lock:
            observers = list(self._state_observers)
        for callback in observers:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"State observer error: {e}")

    def _notify_audio_observers(self):
        """Notify all audio observers."""
        with self._lock:
            level = self._audio_level
            observers = list(self._audio_observers)
        for callback in observers:
            try:
                callback(level)
            except Exception as e:
                logger.error(f"Audio observer error: {e}")

    def _add_log_entry(self, entry: LogEntry):
        """Add a log entry and notify observers."""
        with self._lock:
            self._recent_logs.append(entry)
            observers = list(self._log_observers)
        for callback in observers:
            try:
                callback(entry)
            except Exception:
                # Don't log here to avoid recursion
                pass

    def _notify_claude_activity_observers(self):
        """Notify all Claude activity observers."""
        activity = self.get_claude_activity()
        with self._lock:
            observers = list(self._claude_activity_observers)
        for callback in observers:
            try:
                callback(activity)
            except Exception as e:
                logger.error(f"Claude activity observer error: {e}")

    # Config access (delegates to Claudette)

    def get_config(self) -> dict:
        """Get current configuration."""
        with self._lock:
            if not self._claudette:
                return {}
            return {
                "whisper": {
                    "mode": self._claudette.whisper_mode,
                    "url": self._claudette.whisper_url,
                    "language": self._claudette.whisper_language,
                },
                "vad": {
                    "threshold": self._claudette.vad_threshold,
                    "min_speech_ms": self._claudette.min_speech_ms,
                    "silence_duration": self._claudette.silence_duration,
                },
                "tts": {
                    "voice": self._claudette.tts_voice,
                    "rate": self._claudette.tts_rate,
                    "pitch": self._claudette.tts_pitch,
                },
                "wake_word": {
                    "word": self._claudette.wake_word,
                    "variants": self._claudette.wake_word_variants,
                },
                "memory": {
                    "enabled": self._claudette.memory is not None,
                    "max_exchanges": (
                        self._claudette.memory.max_exchanges if self._claudette.memory else 20
                    ),
                    "exchange_count": (
                        len(self._claudette.memory.exchanges) if self._claudette.memory else 0
                    ),
                },
                "sounds": {
                    "enabled": self._claudette.sounds.enabled,
                    "volume": self._claudette.sounds.volume,
                },
                "hotkey": {
                    "enabled": self._claudette.hotkey_manager.enabled,
                    "key": self._claudette.hotkey_manager.hotkey,
                },
            }

    def update_config(self, updates: dict) -> dict:
        """Update configuration at runtime.

        Args:
            updates: Dict with config updates (supports nested keys like "tts.voice")

        Returns:
            Updated config
        """
        with self._lock:
            if not self._claudette:
                return {}

            # Process updates
            for key, value in updates.items():
                try:
                    if key == "tts.voice":
                        self._claudette.tts_voice = value
                        self._claudette._audio_cache.clear()
                    elif key == "tts.rate":
                        self._claudette.tts_rate = value
                    elif key == "tts.pitch":
                        self._claudette.tts_pitch = value
                    elif key == "vad.threshold":
                        self._claudette.vad_threshold = float(value)
                    elif key == "vad.silence_duration":
                        self._claudette.silence_duration = float(value)
                    elif key == "sounds.enabled":
                        self._claudette.sounds.enabled = bool(value)
                    elif key == "sounds.volume":
                        self._claudette.sounds.volume = float(value)
                    elif key == "wake_word.word":
                        self._claudette.wake_word = value.lower()
                    elif key.startswith("wake_word.variants"):
                        if isinstance(value, list):
                            self._claudette.wake_word_variants = [v.lower() for v in value]
                    else:
                        logger.warning(f"Unknown config key: {key}")
                except Exception as e:
                    logger.error(f"Failed to update {key}: {e}")

            return self.get_config()

    # Memory/History access

    def get_history(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get conversation history."""
        with self._lock:
            if not self._claudette or not self._claudette.memory:
                return []
            exchanges = self._claudette.memory.exchanges
        # Apply pagination
        start = max(0, len(exchanges) - offset - limit)
        end = len(exchanges) - offset
        return exchanges[start:end][::-1]  # Return newest first

    def clear_history(self):
        """Clear conversation history."""
        with self._lock:
            if self._claudette and self._claudette.memory:
                self._claudette.memory.clear()

    # Skills access

    def get_skills(self) -> list[dict]:
        """Get list of available skills."""
        with self._lock:
            if not self._claudette:
                return []
            return self._claudette.skills.list_skills()
