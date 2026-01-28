"""
Claudette Skills/Plugins System

Allows extending Claudette with custom voice-activated skills.
"""

import importlib.util
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger("claudette")


class Skill(ABC):
    """Base class for Claudette skills.

    To create a skill:
    1. Create a Python file in the skills/ directory
    2. Define a class that inherits from Skill
    3. Implement the required methods

    Example:
        class WeatherSkill(Skill):
            name = "weather"
            triggers = ["what's the weather", "weather forecast", "is it going to rain"]

            def execute(self, command: str, claudette) -> str:
                return "I'm afraid I don't have weather data, sir."
    """

    # Skill metadata - override in subclass
    name: str = "base_skill"
    description: str = "A Claudette skill"
    triggers: list[str] = []  # Phrases that trigger this skill

    def matches(self, command: str) -> bool:
        """Check if command matches any trigger phrases."""
        command_lower = command.lower()
        for trigger in self.triggers:
            if trigger.lower() in command_lower:
                return True
        return False

    @abstractmethod
    def execute(self, command: str, claudette) -> str | None:
        """Execute the skill and return a response.

        Args:
            command: The user's voice command
            claudette: Reference to the Claudette instance for access to config, etc.

        Returns:
            Response string to speak, or None to pass to Claude
        """
        pass


class SkillManager:
    """Manages loading and executing Claudette skills."""

    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = skills_dir or Path.cwd() / "skills"
        self.skills: list[Skill] = []
        self._load_builtin_skills()
        self._load_custom_skills()

    def _load_builtin_skills(self):
        """Load built-in skills."""

        # Time skill
        class TimeSkill(Skill):
            name = "time"
            description = "Tell the current time"
            triggers = ["what time is it", "what's the time", "current time", "tell me the time"]

            def execute(self, command: str, claudette) -> str:
                from datetime import datetime

                now = datetime.now()
                hour = now.hour
                minute = now.minute

                # Convert to 12-hour format with period
                period = (
                    "in the morning"
                    if hour < 12
                    else "in the afternoon" if hour < 17 else "in the evening"
                )
                if hour == 0:
                    hour_12 = 12
                elif hour > 12:
                    hour_12 = hour - 12
                else:
                    hour_12 = hour

                if minute == 0:
                    return f"It's {hour_12} o'clock {period}, sir."
                elif minute == 30:
                    return f"It's half past {hour_12} {period}, sir."
                else:
                    return f"It's {hour_12}:{minute:02d} {period}, sir."

        # Date skill
        class DateSkill(Skill):
            name = "date"
            description = "Tell the current date"
            triggers = ["what's the date", "what day is it", "current date", "today's date"]

            def execute(self, command: str, claudette) -> str:
                from datetime import datetime

                now = datetime.now()
                day_name = now.strftime("%A")
                month_name = now.strftime("%B")
                day = now.day
                year = now.year

                # Add ordinal suffix
                if 10 <= day % 100 <= 20:
                    suffix = "th"
                else:
                    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

                return f"Today is {day_name}, the {day}{suffix} of {month_name}, {year}, sir."

        # Clear memory skill
        class ClearMemorySkill(Skill):
            name = "clear_memory"
            description = "Clear conversation memory"
            triggers = ["clear memory", "forget everything", "clear conversation", "start fresh"]

            def execute(self, command: str, claudette) -> str:
                if claudette.memory:
                    claudette.memory.clear()
                    return "I've cleared my memory, sir. Starting fresh."
                return "Memory is not enabled, sir."

        # Status skill
        class StatusSkill(Skill):
            name = "status"
            description = "Report Claudette's status"
            triggers = ["what's your status", "system status", "how are you doing", "are you okay"]

            def execute(self, command: str, claudette) -> str:
                import torch

                parts = ["All systems operational, sir."]

                # GPU status
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    parts.append(f"Running on {gpu_name}.")
                else:
                    parts.append("Running on CPU.")

                # Memory status
                if claudette.memory:
                    count = len(claudette.memory.exchanges)
                    parts.append(f"I have {count} conversation exchanges in memory.")

                # Whisper mode
                parts.append(f"Using {claudette.whisper_mode} transcription.")

                return " ".join(parts)

        # System info skill
        class SystemInfoSkill(Skill):
            name = "system_info"
            description = "Report system information"
            triggers = ["system info", "computer info", "system information", "how is my computer"]

            def execute(self, command: str, claudette) -> str:
                import platform
                import subprocess

                parts = []

                # OS info
                parts.append(f"You're running {platform.system()} {platform.release()}, sir.")

                # Uptime (Linux/macOS)
                try:
                    result = subprocess.run(
                        ["uptime", "-p"], capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        uptime = result.stdout.strip().replace("up ", "")
                        parts.append(f"System has been running for {uptime}.")
                except Exception:
                    pass

                # Memory usage
                try:
                    import psutil

                    mem = psutil.virtual_memory()
                    mem.used / (1024**3)
                    total_gb = mem.total / (1024**3)
                    parts.append(f"Memory usage is {mem.percent:.0f}% of {total_gb:.1f} gigabytes.")
                except ImportError:
                    pass

                # CPU usage
                try:
                    import psutil

                    cpu = psutil.cpu_percent(interval=0.5)
                    parts.append(f"CPU is at {cpu:.0f}%.")
                except ImportError:
                    pass

                return " ".join(parts) if parts else "Unable to retrieve system information, sir."

        # Battery skill
        class BatterySkill(Skill):
            name = "battery"
            description = "Check battery status"
            triggers = ["battery level", "battery status", "how much battery", "am i plugged in"]

            def execute(self, command: str, claudette) -> str:
                try:
                    import psutil

                    battery = psutil.sensors_battery()
                    if battery is None:
                        return "I don't detect a battery, sir. This appears to be a desktop system."

                    percent = battery.percent
                    plugged = battery.power_plugged

                    if plugged:
                        if percent >= 100:
                            return "Battery is fully charged and plugged in, sir."
                        else:
                            return f"Battery is at {percent:.0f}% and charging, sir."
                    else:
                        secs_left = battery.secsleft
                        if secs_left > 0:
                            hours = secs_left // 3600
                            mins = (secs_left % 3600) // 60
                            if hours > 0:
                                return f"Battery is at {percent:.0f}% with about {hours} hours and {mins} minutes remaining, sir."
                            else:
                                return f"Battery is at {percent:.0f}% with about {mins} minutes remaining, sir."
                        else:
                            return f"Battery is at {percent:.0f}%, sir."

                except ImportError:
                    return "I need the psutil library to check battery status, sir."
                except Exception as e:
                    return f"I couldn't check the battery status, sir. {str(e)}"

        # Volume control skill
        class VolumeSkill(Skill):
            name = "volume"
            description = "Control system volume"
            triggers = [
                "volume up",
                "louder",
                "turn it up",
                "increase volume",
                "volume down",
                "quieter",
                "turn it down",
                "decrease volume",
                "mute",
                "unmute",
                "toggle mute",
                "silence",
                "what's the volume",
                "volume level",
            ]

            def execute(self, command: str, claudette) -> str:
                import platform
                import subprocess

                command_lower = command.lower()
                system = platform.system()

                try:
                    if (
                        "up" in command_lower
                        or "louder" in command_lower
                        or "increase" in command_lower
                    ):
                        if system == "Linux":
                            subprocess.run(
                                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"], timeout=5
                            )
                        elif system == "Darwin":
                            subprocess.run(
                                [
                                    "osascript",
                                    "-e",
                                    "set volume output volume (output volume of (get volume settings) + 10)",
                                ],
                                timeout=5,
                            )
                        return "Volume increased, sir."

                    elif (
                        "down" in command_lower
                        or "quieter" in command_lower
                        or "decrease" in command_lower
                    ):
                        if system == "Linux":
                            subprocess.run(
                                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"], timeout=5
                            )
                        elif system == "Darwin":
                            subprocess.run(
                                [
                                    "osascript",
                                    "-e",
                                    "set volume output volume (output volume of (get volume settings) - 10)",
                                ],
                                timeout=5,
                            )
                        return "Volume decreased, sir."

                    elif "mute" in command_lower or "silence" in command_lower:
                        if "unmute" in command_lower:
                            if system == "Linux":
                                subprocess.run(
                                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], timeout=5
                                )
                            elif system == "Darwin":
                                subprocess.run(
                                    ["osascript", "-e", "set volume without output muted"],
                                    timeout=5,
                                )
                            return "Unmuted, sir."
                        else:
                            if system == "Linux":
                                subprocess.run(
                                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                                    timeout=5,
                                )
                            elif system == "Darwin":
                                subprocess.run(
                                    ["osascript", "-e", "set volume with output muted"], timeout=5
                                )
                            return "Muted, sir."

                    elif "level" in command_lower or "what" in command_lower:
                        if system == "Linux":
                            result = subprocess.run(
                                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            # Parse output like "Volume: front-left: 65536 / 100%..."
                            import re

                            match = re.search(r"(\d+)%", result.stdout)
                            if match:
                                return f"Volume is at {match.group(1)}%, sir."
                        elif system == "Darwin":
                            result = subprocess.run(
                                ["osascript", "-e", "output volume of (get volume settings)"],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            return f"Volume is at {result.stdout.strip()}%, sir."
                        return "I couldn't determine the volume level, sir."

                except Exception as e:
                    logger.error(f"Volume control error: {e}")
                    return "I had trouble controlling the volume, sir."

                return "I didn't catch that volume command, sir."

        # Lock screen skill
        class LockScreenSkill(Skill):
            name = "lock_screen"
            description = "Lock the computer screen"
            triggers = ["lock screen", "lock computer", "lock my computer", "lock the screen"]

            def execute(self, command: str, claudette) -> str:
                import platform
                import subprocess

                system = platform.system()

                try:
                    if system == "Linux":
                        # Try common Linux screen lockers
                        for locker in [
                            "gnome-screensaver-command -l",
                            "xdg-screensaver lock",
                            "loginctl lock-session",
                        ]:
                            try:
                                subprocess.run(locker.split(), timeout=5)
                                return "Locking the screen, sir."
                            except Exception:
                                continue
                    elif system == "Darwin":
                        subprocess.run(
                            [
                                "osascript",
                                "-e",
                                'tell application "System Events" to keystroke "q" using {control down, command down}',
                            ],
                            timeout=5,
                        )
                        return "Locking the screen, sir."

                except Exception as e:
                    logger.error(f"Lock screen error: {e}")

                return "I'm afraid I couldn't lock the screen, sir."

        # Screenshot skill
        class ScreenshotSkill(Skill):
            name = "screenshot"
            description = "Take a screenshot"
            triggers = ["take a screenshot", "screenshot", "capture screen", "screen capture"]

            def execute(self, command: str, claudette) -> str:
                import platform
                import subprocess
                from datetime import datetime
                from pathlib import Path

                system = platform.system()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = Path.home() / f"screenshot_{timestamp}.png"

                try:
                    if system == "Linux":
                        # Try common screenshot tools
                        for tool in [
                            f"gnome-screenshot -f {filename}",
                            f"scrot {filename}",
                            f"import -window root {filename}",
                        ]:
                            try:
                                subprocess.run(tool.split(), timeout=10)
                                if filename.exists():
                                    return f"Screenshot saved to {filename}, sir."
                            except Exception:
                                continue
                    elif system == "Darwin":
                        subprocess.run(["screencapture", str(filename)], timeout=10)
                        if filename.exists():
                            return f"Screenshot saved to {filename}, sir."

                except Exception as e:
                    logger.error(f"Screenshot error: {e}")

                return "I'm afraid I couldn't capture the screen, sir."

        # Voice control skill
        class VoiceChangeSkill(Skill):
            name = "voice_change"
            description = "Change Claudette's voice"
            triggers = [
                "change voice",
                "switch voice",
                "different voice",
                "use voice",
                "speak with",
                "change accent",
            ]

            # Available voice presets
            VOICE_PRESETS = {
                "sonia": ("en-GB-SoniaNeural", "British English - Sonia (default)"),
                "libby": ("en-GB-LibbyNeural", "British English - Libby (warm)"),
                "maisie": ("en-GB-MaisieNeural", "British English - Maisie (young)"),
                "aria": ("en-US-AriaNeural", "American English - Aria"),
                "jenny": ("en-US-JennyNeural", "American English - Jenny"),
                "emma": ("en-AU-NatashaNeural", "Australian English - Natasha"),
                "guy": ("en-GB-RyanNeural", "British English - Ryan (male)"),
            }

            def execute(self, command: str, claudette) -> str:
                command_lower = command.lower()

                # Check for list voices
                if (
                    "list" in command_lower
                    or "available" in command_lower
                    or "what" in command_lower
                ):
                    voices = []
                    for name, (voice_id, desc) in self.VOICE_PRESETS.items():
                        current = " (current)" if voice_id == claudette.tts_voice else ""
                        voices.append(f"{name}: {desc}{current}")
                    voice_list = ", ".join(voices)
                    return f"Available voices are: {voice_list}. Say 'change voice to' followed by the name."

                # Check for specific voice request
                for name, (voice_id, _desc) in self.VOICE_PRESETS.items():
                    if name in command_lower:
                        claudette.tts_voice = voice_id
                        # Clear audio cache since voice changed
                        claudette._audio_cache.clear()
                        return f"Very well, sir. I've switched to the {name} voice."

                return "Which voice would you like, sir? Say 'list voices' to hear the options."

        # Personality change skill
        class PersonalitySkill(Skill):
            name = "personality"
            description = "Change Claudette's personality"
            triggers = [
                "change personality",
                "switch personality",
                "different personality",
                "be more",
                "act like",
                "personality mode",
            ]

            def execute(self, command: str, claudette) -> str:
                from .personalities import PERSONALITIES, get_personality

                command_lower = command.lower()

                # Check for list personalities
                if (
                    "list" in command_lower
                    or "available" in command_lower
                    or "what" in command_lower
                ):
                    names = [f"{name}: {desc}" for name, (desc, _) in PERSONALITIES.items()]
                    return f"Available personalities: {', '.join(names)}. Say 'change personality to' followed by the name."

                # Check for specific personality request
                for name in PERSONALITIES.keys():
                    if name in command_lower:
                        claudette.system_prompt = get_personality(name)
                        return f"Personality switched to {name}. My demeanor has been adjusted accordingly."

                return "Which personality would you like? Say 'list personalities' to hear the options."

        # Wake word management skill
        class WakeWordSkill(Skill):
            name = "wake_word"
            description = "Manage wake word variants"
            triggers = [
                "add wake word",
                "new wake word",
                "wake word variant",
                "teach wake word",
                "learn my voice",
            ]

            def execute(self, command: str, claudette) -> str:
                command_lower = command.lower()

                # Check for add/teach patterns
                if "add" in command_lower or "teach" in command_lower or "learn" in command_lower:
                    # Extract the variant from the command
                    # Pattern: "add wake word [variant]" or "teach wake word [variant]"
                    import re

                    patterns = [
                        r"add wake word[:\s]+(.+)",
                        r"teach wake word[:\s]+(.+)",
                        r"add[:\s]+(.+)\s+as wake word",
                        r"new wake word[:\s]+(.+)",
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, command_lower)
                        if match:
                            variant = match.group(1).strip()
                            if variant:
                                claudette.wake_word_variants.append(variant)
                                return f"Added '{variant}' as a wake word variant, sir. I'll respond to that as well."

                    return "To add a wake word variant, say something like 'add wake word cloud' if I'm mishearing you."

                # List current variants
                all_variants = [claudette.wake_word] + claudette.wake_word_variants[:5]
                return f"Current wake word is '{claudette.wake_word}'. I also respond to variations like: {', '.join(all_variants)}."

        # List skills skill
        class ListSkillsSkill(Skill):
            name = "list_skills"
            description = "List available skills"
            triggers = ["list skills", "what can you do", "available commands", "help me"]

            def execute(self, command: str, claudette) -> str:
                skill_names = [s.name for s in claudette.skills.skills]
                return f"I have {len(skill_names)} skills available, sir: {', '.join(skill_names)}. For anything else, I'll consult Claude."

        self.skills.extend(
            [
                TimeSkill(),
                DateSkill(),
                ClearMemorySkill(),
                StatusSkill(),
                SystemInfoSkill(),
                BatterySkill(),
                VolumeSkill(),
                LockScreenSkill(),
                ScreenshotSkill(),
                VoiceChangeSkill(),
                PersonalitySkill(),
                WakeWordSkill(),
                ListSkillsSkill(),
            ]
        )
        logger.info(f"Loaded {len(self.skills)} built-in skills")

    def _load_custom_skills(self):
        """Load custom skills from the skills directory."""
        if not self.skills_dir.exists():
            logger.debug(f"Skills directory not found: {self.skills_dir}")
            return

        loaded = 0
        for skill_file in self.skills_dir.glob("*.py"):
            if skill_file.name.startswith("_"):
                continue

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(skill_file.stem, skill_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find Skill subclasses
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, Skill) and obj is not Skill:
                        skill = obj()
                        self.skills.append(skill)
                        logger.info(f"Loaded custom skill: {skill.name}")
                        loaded += 1

            except Exception as e:
                logger.error(f"Failed to load skill {skill_file}: {e}")

        if loaded:
            logger.info(f"Loaded {loaded} custom skills from {self.skills_dir}")

    def find_skill(self, command: str) -> Skill | None:
        """Find a skill that matches the command."""
        for skill in self.skills:
            if skill.matches(command):
                logger.debug(f"Command matched skill: {skill.name}")
                return skill
        return None

    def execute(self, command: str, claudette) -> str | None:
        """Try to execute a matching skill.

        Returns:
            Response string if skill handled the command, None otherwise
        """
        skill = self.find_skill(command)
        if skill:
            try:
                return skill.execute(command, claudette)
            except Exception as e:
                logger.error(f"Skill {skill.name} failed: {e}")
                return None
        return None

    def list_skills(self) -> list[dict]:
        """List all available skills."""
        return [
            {"name": s.name, "description": s.description, "triggers": s.triggers}
            for s in self.skills
        ]
