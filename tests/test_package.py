"""Tests for the main package."""

import pytest


class TestPackageImports:
    """Test that all package imports work correctly."""

    def test_import_main_package(self):
        """Test importing the main package."""
        import claudette
        assert hasattr(claudette, '__version__')

    def test_import_claudette_class(self):
        """Test importing the Claudette class."""
        from claudette import Claudette
        assert Claudette is not None

    def test_import_skill_classes(self):
        """Test importing skill classes."""
        from claudette import Skill, SkillManager
        assert Skill is not None
        assert SkillManager is not None

    def test_import_sound_effects(self):
        """Test importing SoundEffects."""
        from claudette import SoundEffects
        assert SoundEffects is not None

    def test_import_hotkey_manager(self):
        """Test importing HotkeyManager."""
        from claudette import HotkeyManager
        assert HotkeyManager is not None

    def test_import_tray_classes(self):
        """Test importing tray classes."""
        from claudette import TrayIcon, WaveformWindow
        assert TrayIcon is not None
        assert WaveformWindow is not None

    def test_import_notification_manager(self):
        """Test importing NotificationManager."""
        from claudette import NotificationManager
        assert NotificationManager is not None

    def test_import_personalities(self):
        """Test importing personality functions."""
        from claudette import PERSONALITIES, get_personality, list_personalities
        assert PERSONALITIES is not None
        assert get_personality is not None
        assert list_personalities is not None

    def test_import_audio_processor(self):
        """Test importing AudioProcessor."""
        from claudette import AudioProcessor
        assert AudioProcessor is not None

    def test_import_offline_classes(self):
        """Test importing offline classes."""
        from claudette import OfflineFallback, check_network
        assert OfflineFallback is not None
        assert check_network is not None

    def test_version_format(self):
        """Test that version follows semver format."""
        import claudette
        import re
        
        version = claudette.__version__
        # Basic semver pattern
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
        assert re.match(pattern, version), f"Version {version} doesn't match semver"

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import claudette
        
        expected = [
            'Claudette', 'main', 'Skill', 'SkillManager', 'SoundEffects',
            'HotkeyManager', 'TrayIcon', 'WaveformWindow', 'NotificationManager',
            'PERSONALITIES', 'get_personality', 'list_personalities', 'AudioProcessor',
            'OfflineFallback', 'check_network'
        ]
        
        for name in expected:
            assert name in claudette.__all__, f"{name} not in __all__"
