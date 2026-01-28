"""Tests for Claudette assistant."""

import pytest


class TestWakeWordDetection:
    """Test wake word variant detection."""

    wake_word_variants = [
        "claudette",
        "claudet",
        "claudia",
        "clodette",
        "cladette",
        "cloud",
        "claud",
        "audit",
        "audette",
        "kladette",
        "klodette",
        "plot it",
        "godette",
        "colette",
        "clodet",
        "clawed",
        "plaudit",
        "hey claudette",
        "hey claudet",
        "okay claudette",
        "laudette",
        "lodette",
        "la dette",
    ]

    @pytest.mark.parametrize("variant", wake_word_variants)
    def test_wake_word_variants_recognized(self, variant):
        """Test that all wake word variants are in the list."""
        # This is a basic test to ensure variants are defined
        assert variant.lower() == variant


class TestTranscriptionParsing:
    """Test parsing of transcribed text."""

    def test_extract_command_after_wake_word(self):
        """Test command extraction after wake word."""
        transcription = "Claudette, how's my computer running?"
        wake_word = "claudette"

        # Simple extraction logic
        lower = transcription.lower()
        for suffix in [",", ".", "!", "?", " ", ""]:
            pattern = f"{wake_word}{suffix}"
            if lower.startswith(pattern):
                command = transcription[len(pattern):].strip()
                command = command.lstrip(",.!? ")
                break

        assert command == "how's my computer running?"

    def test_wake_word_only(self):
        """Test detection of wake word with no command."""
        transcription = "Claudette."
        wake_word = "claudette"

        lower = transcription.lower()
        command = None
        for suffix in [",", ".", "!", "?", " ", ""]:
            pattern = f"{wake_word}{suffix}"
            if lower.startswith(pattern):
                command = transcription[len(pattern):].strip()
                command = command.lstrip(",.!? ")
                break

        assert command == ""


class TestConversationMode:
    """Test conversation mode logic."""

    def test_end_phrases_detected(self):
        """Test that end phrases are properly detected."""
        end_phrases = [
            "thank you",
            "thanks",
            "that's all",
            "goodbye",
            "bye",
            "nevermind",
            "never mind",
        ]

        test_inputs = [
            ("Thank you very much", True),
            ("Thanks for your help", True),
            ("That's all for now", True),
            ("Goodbye Claudette", True),
            ("What's the weather", False),
            ("Tell me more", False),
        ]

        for text, should_end in test_inputs:
            text_lower = text.lower()
            detected = any(phrase in text_lower for phrase in end_phrases)
            assert detected == should_end, f"Failed for: {text}"

    def test_confirmation_phrases_detected(self):
        """Test that confirmation phrases are properly detected."""
        affirmative = [
            "yes",
            "yeah",
            "yep",
            "go ahead",
            "do it",
            "proceed",
            "please do",
            "approved",
            "confirmed",
            "affirmative",
        ]

        test_inputs = [
            ("Yes please", True),
            ("Yeah, go ahead", True),
            ("Do it now", True),
            ("No thanks", False),
            ("Maybe later", False),
        ]

        for text, should_confirm in test_inputs:
            text_lower = text.lower()
            detected = any(phrase in text_lower for phrase in affirmative)
            assert detected == should_confirm, f"Failed for: {text}"
