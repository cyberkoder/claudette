"""
Claudette - A sophisticated AI voice assistant

A 1940s British bombshell with wit, charm, and intelligence.
"""

__version__ = "0.1.2"
__author__ = "Claudette Contributors"

from .assistant import Claudette, main
from .skills import Skill, SkillManager
from .sounds import SoundEffects
from .hotkey import HotkeyManager
from .tray import TrayIcon, WaveformWindow
from .notifications import NotificationManager
from .personalities import PERSONALITIES, get_personality, list_personalities
from .audio_processing import AudioProcessor
from .offline import OfflineFallback, check_network

__all__ = [
    "Claudette", "main", "Skill", "SkillManager", "SoundEffects",
    "HotkeyManager", "TrayIcon", "WaveformWindow", "NotificationManager",
    "PERSONALITIES", "get_personality", "list_personalities", "AudioProcessor",
    "OfflineFallback", "check_network",
    "__version__"
]
