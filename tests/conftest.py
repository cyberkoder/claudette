"""Pytest configuration and fixtures for Claudette tests."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_claudette():
    """Create a mock Claudette instance for testing skills."""
    mock = MagicMock()
    mock.wake_word = "claudette"
    mock.wake_word_variants = []
    mock.tts_voice = "en-GB-SoniaNeural"
    mock.whisper_mode = "local"
    mock.memory = MagicMock()
    mock.memory.exchanges = []
    mock.skills = MagicMock()
    mock.skills.skills = []
    mock.system_prompt = "Test prompt"
    mock._audio_cache = {}
    return mock


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        "whisper": {
            "mode": "local",
            "model": "base",
            "language": "en"
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1
        },
        "wake_word": {
            "word": "claudette",
            "variants": []
        },
        "vad": {
            "threshold": 0.5,
            "min_speech_ms": 250,
            "silence_duration": 1.5,
            "device": "cpu"
        },
        "tts": {
            "voice": "en-GB-SoniaNeural",
            "rate": "+0%",
            "pitch": "+0Hz"
        },
        "memory": {
            "enabled": True,
            "max_exchanges": 20
        },
        "sounds": {
            "enabled": False,
            "volume": 0.3
        },
        "hotkey": {
            "enabled": False
        },
        "tray": {
            "enabled": False
        },
        "notifications": {
            "enabled": False
        },
        "personality": {
            "preset": "claudette"
        },
        "audio_processing": {
            "noise_reduce": False,
            "normalize": True
        },
        "offline": {
            "enabled": True
        }
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create a temporary config file."""
    import yaml
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path
