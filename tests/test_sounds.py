"""Tests for the sounds module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestSoundEffects:
    """Test the SoundEffects class."""

    @patch('pygame.mixer')
    def test_init_disabled(self, mock_mixer):
        """Test initialization with sounds disabled."""
        from claudette.sounds import SoundEffects
        
        effects = SoundEffects(enabled=False)
        assert effects.enabled is False
        assert len(effects._sounds) == 0

    @patch('pygame.mixer')
    def test_init_enabled(self, mock_mixer):
        """Test initialization with sounds enabled."""
        from claudette.sounds import SoundEffects
        
        # Mock the Sound class
        mock_mixer.Sound.return_value = MagicMock()
        
        effects = SoundEffects(enabled=True, volume=0.5)
        assert effects.enabled is True
        assert effects.volume == 0.5

    @patch('pygame.mixer')
    def test_volume_clamping(self, mock_mixer):
        """Test that volume is clamped to valid range."""
        from claudette.sounds import SoundEffects
        
        effects1 = SoundEffects(enabled=False, volume=-0.5)
        assert effects1.volume == 0.0
        
        effects2 = SoundEffects(enabled=False, volume=1.5)
        assert effects2.volume == 1.0

    @patch('pygame.mixer')
    def test_play_disabled(self, mock_mixer):
        """Test that play does nothing when disabled."""
        from claudette.sounds import SoundEffects
        
        effects = SoundEffects(enabled=False)
        # Should not raise any errors
        effects.play('wake')
        effects.play_wake()
        effects.play_done()

    def test_generate_tone(self):
        """Test tone generation."""
        from claudette.sounds import SoundEffects
        
        # Create instance without pygame init
        effects = SoundEffects.__new__(SoundEffects)
        effects.volume = 0.5
        
        tone = effects._generate_tone(440, 0.1)
        
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.int16
        assert len(tone) == int(44100 * 0.1)  # sample_rate * duration

    def test_generate_chime(self):
        """Test chime generation."""
        from claudette.sounds import SoundEffects
        
        # Create instance without pygame init
        effects = SoundEffects.__new__(SoundEffects)
        effects.volume = 0.5
        
        chime = effects._generate_chime([440, 550, 660], 0.1)
        
        assert isinstance(chime, np.ndarray)
        assert chime.dtype == np.int16
        assert len(chime) > 0
