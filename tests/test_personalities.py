"""Tests for the personalities module."""

import pytest
from claudette.personalities import (
    PERSONALITIES,
    get_personality,
    list_personalities,
    CLAUDETTE_DEFAULT,
    PROFESSIONAL,
    FRIENDLY,
    BUTLER,
    PIRATE
)


class TestPersonalities:
    """Test personality presets."""

    def test_all_personalities_defined(self):
        """Test that all expected personalities are defined."""
        expected = ["claudette", "professional", "friendly", "butler", "pirate"]
        for name in expected:
            assert name in PERSONALITIES

    def test_get_personality_valid(self):
        """Test getting a valid personality."""
        prompt = get_personality("claudette")
        assert prompt == CLAUDETTE_DEFAULT
        assert "Claudette" in prompt
        assert "1940s" in prompt

    def test_get_personality_invalid(self):
        """Test getting an invalid personality returns default."""
        prompt = get_personality("nonexistent")
        assert prompt == CLAUDETTE_DEFAULT

    def test_list_personalities(self):
        """Test listing all personalities."""
        personalities = list_personalities()
        
        assert isinstance(personalities, dict)
        assert "claudette" in personalities
        assert "professional" in personalities

    def test_personality_content(self):
        """Test that each personality has meaningful content."""
        for name, (desc, prompt) in PERSONALITIES.items():
            assert len(prompt) > 100, f"{name} prompt too short"
            assert "personality" in prompt.lower() or "traits" in prompt.lower(), \
                f"{name} should describe personality traits"

    def test_claudette_personality(self):
        """Test Claudette's specific personality traits."""
        prompt = CLAUDETTE_DEFAULT
        assert "British" in prompt
        assert "sir" in prompt.lower()
        assert "witty" in prompt.lower() or "wit" in prompt.lower()

    def test_professional_personality(self):
        """Test professional personality traits."""
        prompt = PROFESSIONAL
        assert "professional" in prompt.lower()
        assert "concise" in prompt.lower() or "direct" in prompt.lower()

    def test_butler_personality(self):
        """Test butler personality traits."""
        prompt = BUTLER
        assert "butler" in prompt.lower() or "Jeeves" in prompt
        assert "Sir" in prompt or "Madam" in prompt

    def test_pirate_personality(self):
        """Test pirate personality traits."""
        prompt = PIRATE
        assert "pirate" in prompt.lower()
        assert "Arr" in prompt or "Ahoy" in prompt or "Matey" in prompt
