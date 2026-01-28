"""
Sound effects for Claudette state changes.

Uses synthesized tones - no external audio files needed.
"""

import io
import logging
import wave

import numpy as np
import pygame

logger = logging.getLogger("claudette")


class SoundEffects:
    """Manages audio feedback sounds for Claudette."""

    def __init__(self, enabled: bool = True, volume: float = 0.3):
        self.enabled = enabled
        self.volume = max(0.0, min(1.0, volume))
        self._sounds: dict[str, pygame.mixer.Sound] = {}

        if enabled:
            self._generate_sounds()
            logger.info("Sound effects initialized")

    def _generate_tone(
        self, frequency: float, duration: float, sample_rate: int = 44100, fade_ms: int = 20
    ) -> np.ndarray:
        """Generate a sine wave tone."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t)

        # Apply fade in/out to prevent clicks
        fade_samples = int(sample_rate * fade_ms / 1000)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

        # Apply volume
        tone *= self.volume

        # Convert to 16-bit
        return (tone * 32767).astype(np.int16)

    def _generate_chime(self, frequencies: list[float], duration: float = 0.15) -> np.ndarray:
        """Generate a multi-tone chime."""
        sample_rate = 44100
        total_samples = int(sample_rate * duration * len(frequencies))
        result = np.zeros(total_samples, dtype=np.float32)

        for i, freq in enumerate(frequencies):
            tone = self._generate_tone(freq, duration, sample_rate).astype(np.float32) / 32767
            start = int(i * sample_rate * duration * 0.7)  # Overlap slightly
            end = start + len(tone)
            if end <= len(result):
                result[start:end] += tone * (1 - i * 0.2)  # Fade each subsequent note

        # Normalize and convert
        result = np.clip(result, -1, 1)
        return (result * 32767).astype(np.int16)

    def _array_to_sound(self, audio: np.ndarray, sample_rate: int = 44100) -> pygame.mixer.Sound:
        """Convert numpy array to pygame Sound."""
        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio.tobytes())

        buffer.seek(0)
        return pygame.mixer.Sound(buffer)

    def _generate_sounds(self):
        """Generate all sound effects."""
        try:
            # Wake word detected - pleasant ascending chime
            wake_audio = self._generate_chime([523, 659, 784], 0.12)  # C5, E5, G5
            self._sounds["wake"] = self._array_to_sound(wake_audio)

            # Recording started - soft low tone
            record_audio = self._generate_tone(440, 0.1)  # A4
            self._sounds["record"] = self._array_to_sound(record_audio)

            # Processing/thinking - subtle double beep
            process_audio = np.concatenate(
                [
                    self._generate_tone(880, 0.05),  # A5
                    np.zeros(500, dtype=np.int16),  # Short gap
                    self._generate_tone(880, 0.05),  # A5
                ]
            )
            self._sounds["process"] = self._array_to_sound(process_audio)

            # Done/success - descending gentle chime
            done_audio = self._generate_chime([784, 659, 523], 0.1)  # G5, E5, C5
            self._sounds["done"] = self._array_to_sound(done_audio)

            # Error - low buzz
            error_audio = self._generate_tone(220, 0.2)  # A3
            self._sounds["error"] = self._array_to_sound(error_audio)

            logger.debug(f"Generated {len(self._sounds)} sound effects")

        except Exception as e:
            logger.error(f"Failed to generate sounds: {e}")
            self.enabled = False

    def play(self, sound_name: str):
        """Play a sound effect by name."""
        if not self.enabled:
            return

        sound = self._sounds.get(sound_name)
        if sound:
            try:
                sound.play()
            except Exception as e:
                logger.debug(f"Failed to play sound {sound_name}: {e}")

    def play_wake(self):
        """Play wake word detection sound."""
        self.play("wake")

    def play_record(self):
        """Play recording started sound."""
        self.play("record")

    def play_process(self):
        """Play processing/thinking sound."""
        self.play("process")

    def play_done(self):
        """Play completion sound."""
        self.play("done")

    def play_error(self):
        """Play error sound."""
        self.play("error")
