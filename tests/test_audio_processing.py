"""Tests for the audio processing module."""

import pytest
import numpy as np
from claudette.audio_processing import AudioProcessor, estimate_noise_level


class TestAudioProcessor:
    """Test the AudioProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        processor = AudioProcessor()
        assert processor.sample_rate == 16000
        assert processor.normalize is True

    def test_init_custom(self):
        """Test custom initialization."""
        processor = AudioProcessor(
            sample_rate=44100,
            noise_reduce=False,
            high_pass_cutoff=100.0,
            normalize=False
        )
        assert processor.sample_rate == 44100
        assert processor.noise_reduce is False
        assert processor.high_pass_cutoff == 100.0
        assert processor.normalize is False

    def test_process_float32(self):
        """Test processing float32 audio."""
        processor = AudioProcessor(noise_reduce=False)
        
        # Create test audio (sine wave)
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        result = processor.process(audio)
        
        assert result.dtype == np.float32
        assert len(result) == len(audio)

    def test_process_int16(self):
        """Test processing int16 audio."""
        processor = AudioProcessor(noise_reduce=False)
        
        # Create test audio
        t = np.linspace(0, 1, 16000)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        result = processor.process(audio)
        
        assert result.dtype == np.int16
        assert len(result) == len(audio)

    def test_normalize_audio(self):
        """Test audio normalization."""
        processor = AudioProcessor(noise_reduce=False, normalize=True)
        
        # Create quiet audio
        audio = np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32)
        
        result = processor.process(audio)
        
        # Should be normalized to ~0.95 max
        assert np.max(np.abs(result)) > 0.9

    def test_process_empty_audio(self):
        """Test processing empty audio."""
        processor = AudioProcessor(noise_reduce=False, normalize=False)

        audio = np.array([], dtype=np.float32)
        result = processor.process(audio)

        assert len(result) == 0


class TestEstimateNoiseLevel:
    """Test the noise level estimation function."""

    def test_silent_audio(self):
        """Test noise level of silent audio."""
        audio = np.zeros(1000, dtype=np.float32)
        level = estimate_noise_level(audio)
        
        assert level == 0.0

    def test_loud_audio(self):
        """Test noise level of loud audio."""
        audio = np.ones(1000, dtype=np.float32) * 0.5
        level = estimate_noise_level(audio)
        
        assert level > 0.5

    def test_empty_audio(self):
        """Test noise level of empty audio."""
        audio = np.array([], dtype=np.float32)
        level = estimate_noise_level(audio)
        
        assert level == 0.0

    def test_int16_audio(self):
        """Test noise level with int16 audio."""
        audio = (np.random.randn(1000) * 10000).astype(np.int16)
        level = estimate_noise_level(audio)
        
        assert 0.0 <= level <= 1.0

    def test_noise_level_range(self):
        """Test that noise level is always in valid range."""
        for _ in range(10):
            audio = np.random.randn(1000).astype(np.float32)
            level = estimate_noise_level(audio)
            assert 0.0 <= level <= 1.0
