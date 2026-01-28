"""
Claudette Skills/Plugins System

Allows extending Claudette with custom voice-activated skills.
"""

import importlib.util
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

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
    def execute(self, command: str, claudette) -> Optional[str]:
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

    def __init__(self, skills_dir: Optional[Path] = None):
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
                period = "in the morning" if hour < 12 else "in the afternoon" if hour < 17 else "in the evening"
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

        self.skills.extend([
            TimeSkill(),
            DateSkill(),
            ClearMemorySkill(),
            StatusSkill(),
        ])
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
                spec = importlib.util.spec_from_file_location(
                    skill_file.stem, skill_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find Skill subclasses
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and
                        issubclass(obj, Skill) and
                        obj is not Skill):
                        skill = obj()
                        self.skills.append(skill)
                        logger.info(f"Loaded custom skill: {skill.name}")
                        loaded += 1

            except Exception as e:
                logger.error(f"Failed to load skill {skill_file}: {e}")

        if loaded:
            logger.info(f"Loaded {loaded} custom skills from {self.skills_dir}")

    def find_skill(self, command: str) -> Optional[Skill]:
        """Find a skill that matches the command."""
        for skill in self.skills:
            if skill.matches(command):
                logger.debug(f"Command matched skill: {skill.name}")
                return skill
        return None

    def execute(self, command: str, claudette) -> Optional[str]:
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
            {
                "name": s.name,
                "description": s.description,
                "triggers": s.triggers
            }
            for s in self.skills
        ]
