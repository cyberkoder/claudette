"""
Claudette - A sophisticated AI voice assistant

A 1940s British bombshell with wit, charm, and intelligence.
"""

__version__ = "0.1.3"
__author__ = "Claudette Contributors"

from .assistant import Claudette, main
from .audio_processing import AudioProcessor
from .hotkey import HotkeyManager
from .notifications import NotificationManager
from .offline import OfflineFallback, check_network
from .personalities import PERSONALITIES, get_personality, list_personalities
from .skills import Skill, SkillManager
from .sounds import SoundEffects
from .tray import TrayIcon, WaveformWindow

# Web module (optional, requires fastapi/uvicorn)
try:
    from .web import ClaudetteStateManager, WebServer

    _WEB_AVAILABLE = True
except ImportError:
    ClaudetteStateManager = None
    WebServer = None
    _WEB_AVAILABLE = False

__all__ = [
    "Claudette",
    "main",
    "Skill",
    "SkillManager",
    "SoundEffects",
    "HotkeyManager",
    "TrayIcon",
    "WaveformWindow",
    "NotificationManager",
    "PERSONALITIES",
    "get_personality",
    "list_personalities",
    "AudioProcessor",
    "OfflineFallback",
    "check_network",
    "WebServer",
    "ClaudetteStateManager",
    "__version__",
]
