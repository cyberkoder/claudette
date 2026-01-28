"""
Example Custom Skill for Claudette

This file demonstrates how to create custom skills.
Place your skill files in the skills/ directory.
"""

from claudette import Skill


class GreetingSkill(Skill):
    """A simple greeting skill example."""

    name = "greeting"
    description = "Respond to greetings"
    triggers = [
        "good morning",
        "good afternoon",
        "good evening",
        "hello claudette",
        "hi claudette",
    ]

    def execute(self, command: str, claudette) -> str:
        command_lower = command.lower()

        if "morning" in command_lower:
            return "Good morning, sir. I trust you slept well?"
        elif "afternoon" in command_lower:
            return "Good afternoon, sir. How may I assist you?"
        elif "evening" in command_lower:
            return "Good evening, sir. Lovely time of day, isn't it?"
        else:
            return "Hello, sir. Claudette at your service."


class JokeSkill(Skill):
    """Tell a joke skill."""

    name = "joke"
    description = "Tell a joke"
    triggers = ["tell me a joke", "make me laugh", "say something funny"]

    def execute(self, command: str, claudette) -> str:
        import random
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything, sir.",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Why did the scarecrow win an award? Because he was outstanding in his field, sir.",
            "I'm reading a book about anti-gravity. It's impossible to put down.",
        ]
        return random.choice(jokes)
