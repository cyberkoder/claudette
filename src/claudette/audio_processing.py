"""
Audio processing utilities for Claudette.

Provides noise reduction and audio enhancement features.
"""

import logging

import numpy as np

logger = logging.getLogger("claudette")

# Try to import noisereduce for spectral subtraction
try:
    import noisereduce as nr

    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    nr = None


class AudioProcessor:
    """Processes audio for improved speech recognition."""

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_reduce: bool = True,
        high_pass_cutoff: float = 80.0,
        normalize: bool = True,
    ):
        self.sample_rate = sample_rate
        self.noise_reduce = noise_reduce and NOISEREDUCE_AVAILABLE
        self.high_pass_cutoff = high_pass_cutoff
        self.normalize = normalize

        # Noise profile for adaptive noise reduction
        self._noise_profile = None
        self._noise_sample_count = 0

        if noise_reduce and not NOISEREDUCE_AVAILABLE:
            logger.warning(
                "Noise reduction disabled: noisereduce not installed. "
                "Install with: pip install noisereduce"
            )

    def _high_pass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply a simple high-pass filter to remove low-frequency noise."""
        if self.high_pass_cutoff <= 0:
            return audio

        try:
            from scipy.signal import butter, filtfilt

            # Design Butterworth high-pass filter
            nyquist = self.sample_rate / 2
            normalized_cutoff = self.high_pass_cutoff / nyquist
            b, a = butter(4, normalized_cutoff, btype="high")

            # Apply filter
            return filtfilt(b, a, audio).astype(audio.dtype)

        except ImportError:
            # Fall back to simple DC removal
            return audio - np.mean(audio)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to maximize dynamic range."""
        if len(audio) == 0:
            return audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Normalize to 95% of max to prevent clipping
            return (audio / max_val * 0.95).astype(audio.dtype)
        return audio

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral noise reduction."""
        if not self.noise_reduce or not NOISEREDUCE_AVAILABLE:
            return audio

        try:
            # Use noisereduce with automatic noise estimation
            reduced = nr.reduce_noise(
                y=audio, sr=self.sample_rate, stationary=True, prop_decrease=0.75
            )
            return reduced.astype(audio.dtype)

        except Exception as e:
            logger.debug(f"Noise reduction failed: {e}")
            return audio

    def update_noise_profile(self, audio_sample: np.ndarray):
        """Update noise profile from a sample of ambient noise.

        Call this with audio samples that contain only background noise
        (no speech) to improve noise reduction.
        """
        if not NOISEREDUCE_AVAILABLE:
            return

        if self._noise_profile is None:
            self._noise_profile = audio_sample.copy()
            self._noise_sample_count = 1
        else:
            # Running average of noise samples
            self._noise_sample_count += 1
            alpha = 1.0 / self._noise_sample_count
            self._noise_profile = (1 - alpha) * self._noise_profile + alpha * audio_sample

        logger.debug(f"Updated noise profile (samples: {self._noise_sample_count})")

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio with all enabled enhancements.

        Args:
            audio: Audio data as numpy array (float32 or int16)

        Returns:
            Processed audio data
        """
        # Convert to float32 for processing
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0
            was_int16 = True
        else:
            was_int16 = False

        # Apply high-pass filter
        audio = self._high_pass_filter(audio)

        # Apply noise reduction
        if self.noise_reduce:
            audio = self._reduce_noise(audio)

        # Normalize
        if self.normalize:
            audio = self._normalize_audio(audio)

        # Convert back to int16 if needed
        if was_int16:
            audio = (audio * 32767).astype(np.int16)

        return audio


def estimate_noise_level(audio: np.ndarray) -> float:
    """Estimate the noise level of an audio sample.

    Returns a value between 0 (silent) and 1 (very noisy).
    """
    if len(audio) == 0:
        return 0.0

    # RMS energy
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32767.0

    rms = np.sqrt(np.mean(audio**2))

    # Normalize to 0-1 range (assuming typical speech RMS is around 0.1-0.3)
    normalized = min(1.0, rms / 0.3)
    return normalized
