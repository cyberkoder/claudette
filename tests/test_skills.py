"""Tests for the skills module."""

import pytest
from datetime import datetime
from claudette.skills import Skill, SkillManager


class TestSkillBase:
    """Test the base Skill class."""

    def test_skill_matches_trigger(self):
        """Test that skills match their trigger phrases."""
        class TestSkill(Skill):
            name = "test"
            triggers = ["hello", "hi there"]
            def execute(self, command, claudette):
                return "Hello!"
        
        skill = TestSkill()
        assert skill.matches("hello world") is True
        assert skill.matches("say hi there please") is True
        assert skill.matches("goodbye") is False

    def test_skill_case_insensitive(self):
        """Test that trigger matching is case insensitive."""
        class TestSkill(Skill):
            name = "test"
            triggers = ["Hello World"]
            def execute(self, command, claudette):
                return "Hi!"
        
        skill = TestSkill()
        assert skill.matches("HELLO WORLD") is True
        assert skill.matches("hello world") is True
        assert skill.matches("HeLLo WoRLd") is True


class TestSkillManager:
    """Test the SkillManager class."""

    def test_builtin_skills_loaded(self):
        """Test that built-in skills are loaded."""
        manager = SkillManager(skills_dir=None)
        skill_names = [s.name for s in manager.skills]
        
        assert "time" in skill_names
        assert "date" in skill_names
        assert "status" in skill_names
        assert "clear_memory" in skill_names

    def test_find_skill(self):
        """Test finding a skill by command."""
        manager = SkillManager(skills_dir=None)
        
        skill = manager.find_skill("what time is it")
        assert skill is not None
        assert skill.name == "time"
        
        skill = manager.find_skill("random gibberish xyz")
        assert skill is None

    def test_list_skills(self):
        """Test listing all skills."""
        manager = SkillManager(skills_dir=None)
        skills_list = manager.list_skills()
        
        assert isinstance(skills_list, list)
        assert len(skills_list) > 0
        assert all("name" in s for s in skills_list)
        assert all("description" in s for s in skills_list)
        assert all("triggers" in s for s in skills_list)


class TestBuiltinSkills:
    """Test built-in skill implementations."""

    def test_time_skill(self, mock_claudette):
        """Test the time skill returns valid response."""
        manager = SkillManager(skills_dir=None)
        response = manager.execute("what time is it", mock_claudette)

        assert response is not None
        assert "o'clock" in response or ":" in response or "half past" in response
        assert "sir" in response.lower()

    def test_date_skill(self, mock_claudette):
        """Test the date skill returns valid response."""
        manager = SkillManager(skills_dir=None)
        response = manager.execute("what's the date today", mock_claudette)
        
        assert response is not None
        assert "sir" in response.lower()
        # Should contain day of week
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(day in response for day in days)

    def test_clear_memory_skill(self, mock_claudette):
        """Test the clear memory skill."""
        manager = SkillManager(skills_dir=None)
        response = manager.execute("clear memory", mock_claudette)
        
        assert response is not None
        mock_claudette.memory.clear.assert_called_once()

    def test_status_skill(self, mock_claudette):
        """Test the status skill."""
        manager = SkillManager(skills_dir=None)
        response = manager.execute("what's your status", mock_claudette)
        
        assert response is not None
        assert "operational" in response.lower() or "running" in response.lower()

    def test_list_skills_skill(self, mock_claudette):
        """Test the list skills skill."""
        manager = SkillManager(skills_dir=None)
        mock_claudette.skills = manager
        response = manager.execute("what can you do", mock_claudette)
        
        assert response is not None
        assert "skills" in response.lower()

    def test_voice_change_skill_list(self, mock_claudette):
        """Test listing available voices."""
        manager = SkillManager(skills_dir=None)
        response = manager.execute("change voice to what are available", mock_claudette)

        assert response is not None
        assert "voice" in response.lower()

    def test_voice_change_skill_switch(self, mock_claudette):
        """Test switching voice."""
        manager = SkillManager(skills_dir=None)
        response = manager.execute("change voice to libby", mock_claudette)
        
        assert response is not None
        assert mock_claudette.tts_voice == "en-GB-LibbyNeural"

    def test_personality_skill_list(self, mock_claudette):
        """Test listing personalities."""
        manager = SkillManager(skills_dir=None)
        response = manager.execute("change personality to what are available", mock_claudette)

        assert response is not None
        assert "personality" in response.lower()
