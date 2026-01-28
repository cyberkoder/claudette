#!/usr/bin/env python3
"""
Claudette - A sophisticated AI voice assistant

A 1940s British bombshell with wit, charm, and intelligence.
Wake word: "Claudette" -> responds "Yes, sir?"
"""

import asyncio
import concurrent.futures
import io
import logging
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import wave
from datetime import datetime
from pathlib import Path

import edge_tts
import numpy as np
import pygame
import requests
import sounddevice as sd
import torch
import yaml

# Optional: faster-whisper for local transcription
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None


# Set up logging - use current working directory for logs
log_dir = Path.cwd() / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"claudette_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("claudette")

# Reduce noise from other libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("pygame").setLevel(logging.WARNING)


def find_config_file(config_name: str = "config.yaml") -> Path:
    """Find config file in standard locations."""
    search_paths = [
        Path.cwd() / config_name,                          # Current directory
        Path.home() / ".config" / "claudette" / config_name,  # User config
        Path(__file__).parent.parent.parent / config_name,    # Package root
    ]

    for path in search_paths:
        if path.exists():
            logger.info(f"Found config at: {path}")
            return path

    # Return default path (will error if not found)
    return search_paths[0]

# Thread pool for parallel operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# Claudette's personality prompt for Claude CLI
CLAUDETTE_SYSTEM_PROMPT = """You are Claudette, a sophisticated AI assistant with the personality of a 1940s British bombshell - think Lauren Bacall meets British intelligence.

Your personality traits:
- Witty, sharp, and occasionally playful with dry British humor
- Confident and composed, never flustered
- Warm but professional - you call the user "sir" naturally
- Knowledgeable and helpful, delivering information with elegance
- Occasionally uses period-appropriate expressions subtly (not overdone)
- Your responses are concise and conversational - this is spoken dialogue, not text

Keep responses brief and natural for speech. You're having a conversation, not writing an essay.
Never use markdown, bullet points, or formatting - speak naturally.
If asked who you are, you're Claudette, a personal AI assistant."""


class VoiceState:
    """Visual state indicators for the terminal."""
    LISTENING = "🎤 Listening for 'Claudette'..."
    LISTENING_CONVO = "💬 Listening (conversation active)..."
    RECORDING = "🔴 Recording..."
    PROCESSING = "⏳ Transcribing..."
    THINKING = "🧠 Thinking..."
    SPEAKING = "🗣️  Speaking..."


class Claudette:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.audio_queue = queue.Queue()
        self.running = False
        self.current_state = VoiceState.LISTENING

        # Audio settings
        self.sample_rate = self.config["audio"]["sample_rate"]
        self.channels = self.config["audio"]["channels"]

        # VAD settings
        self.vad_threshold = self.config["vad"]["threshold"]
        self.min_speech_ms = self.config["vad"]["min_speech_ms"]
        self.silence_duration = self.config["vad"]["silence_duration"]

        # Whisper settings
        self.whisper_mode = self.config.get("whisper", {}).get("mode", "remote")
        self.whisper_url = self.config.get("whisper", {}).get("url", "http://localhost:9300/asr")
        self.whisper_language = self.config.get("whisper", {}).get("language", "en")
        self.whisper_model = None

        # Initialize local Whisper if configured
        if self.whisper_mode == "local":
            self._init_whisper()

        # Wake word
        self.wake_word = self.config.get("wake_word", "claudette").lower()

        # TTS settings - British English female voice
        self.tts_voice = self.config.get("tts", {}).get("voice", "en-GB-SoniaNeural")
        self.tts_rate = self.config.get("tts", {}).get("rate", "+0%")
        self.tts_pitch = self.config.get("tts", {}).get("pitch", "+0Hz")

        # Initialize pygame mixer for audio playback with proper settings
        pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=2048)
        pygame.mixer.init()
        logger.debug(f"Pygame mixer initialized: {pygame.mixer.get_init()}")

        # Cache for pre-generated audio
        self._audio_cache = {}

        # Conversation mode - stay active for follow-ups
        self.conversation_mode = False
        self.conversation_timeout = 10.0  # seconds to wait for follow-up
        self.last_command = None  # Track last command for follow-ups
        self.awaiting_confirmation = False  # Track if waiting for yes/no

        # Initialize VAD model
        self._init_vad()

        # Pre-cache common phrases
        self._precache_phrases()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = find_config_file(config_path)
        logger.info(f"Loading config from: {config_file}")
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def _init_vad(self):
        """Initialize Silero VAD model."""
        logger.info("Loading Silero VAD model...")
        self._print_status("Loading VAD model...")

        # Load Silero VAD
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True
        )

        logger.info("VAD model loaded successfully")
        self._print_status("VAD model loaded")

    def _init_whisper(self):
        """Initialize local Whisper model."""
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            raise ImportError("faster-whisper is required for local mode")

        model_name = self.config.get("whisper", {}).get("model", "base")
        device = self.config.get("whisper", {}).get("device", "auto")
        compute_type = self.config.get("whisper", {}).get("compute_type", "float16")

        # Handle device selection
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Adjust compute type for CPU
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"

        logger.info(f"Loading Whisper model '{model_name}' on {device} ({compute_type})...")
        self._print_status(f"Loading Whisper model ({model_name})...")

        self.whisper_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )

        logger.info("Whisper model loaded successfully")
        self._print_status("Whisper model loaded")

    def _precache_phrases(self):
        """Pre-generate audio for common phrases."""
        self._print_status("Caching common phrases...")

        common_phrases = [
            "Yes, sir?",
            "One moment, sir.",
            "Good day, sir. Claudette at your service.",
            "Goodbye, sir. It's been a pleasure.",
            "My pleasure, sir.",
            "Very well, sir. Proceeding.",
            "Understood, sir.",
        ]

        for phrase in common_phrases:
            try:
                audio_data = asyncio.run(self._synthesize_speech(phrase))
                self._audio_cache[phrase] = audio_data
            except Exception as e:
                print(f"Failed to cache '{phrase}': {e}", file=sys.stderr)

        self._print_status(f"Cached {len(self._audio_cache)} phrases")

    def _print_status(self, message: str, end: str = "\n"):
        """Print status message, clearing previous line."""
        sys.stdout.write(f"\r\033[K{message}{end}")
        sys.stdout.flush()

    def _update_state(self, state: str):
        """Update and display current state."""
        self.current_state = state
        self._print_status(state, end="")

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream - adds audio to queue."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def _audio_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        buffer.seek(0)
        return buffer.read()

    def _transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using local or remote Whisper."""
        audio_duration = len(audio_data) / self.sample_rate
        logger.info(f"Transcribing {audio_duration:.2f}s of audio ({self.whisper_mode} mode)...")

        if self.whisper_mode == "local":
            return self._transcribe_local(audio_data)
        else:
            return self._transcribe_remote(audio_data)

    def _transcribe_local(self, audio_data: np.ndarray) -> str:
        """Transcribe using local faster-whisper model."""
        try:
            start_time = datetime.now()

            # faster-whisper expects float32 audio
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Transcribe
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language=self.whisper_language,
                beam_size=5,
                vad_filter=True
            )

            # Collect all segments
            result = " ".join(segment.text for segment in segments).strip()

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Transcription ({elapsed:.2f}s): '{result}'")
            return result

        except Exception as e:
            logger.error(f"Local Whisper error: {e}")
            return ""

    def _transcribe_remote(self, audio_data: np.ndarray) -> str:
        """Transcribe using remote Whisper API server."""
        wav_bytes = self._audio_to_wav_bytes(audio_data)
        logger.debug(f"WAV size: {len(wav_bytes)} bytes")

        files = {"audio_file": ("audio.wav", wav_bytes, "audio/wav")}
        params = {"language": self.whisper_language, "output": "txt"}

        try:
            start_time = datetime.now()
            response = requests.post(
                self.whisper_url,
                files=files,
                params=params,
                timeout=30
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            response.raise_for_status()
            result = response.text.strip()
            logger.info(f"Transcription ({elapsed:.2f}s): '{result}'")
            return result
        except requests.RequestException as e:
            logger.error(f"Whisper API error: {e}")
            print(f"\nWhisper API error: {e}", file=sys.stderr)
            return ""

    async def _synthesize_speech(self, text: str) -> bytes:
        """Convert text to speech using edge-tts."""
        communicate = edge_tts.Communicate(
            text,
            self.tts_voice,
            rate=self.tts_rate,
            pitch=self.tts_pitch
        )

        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data

    def _speak(self, text: str, audio_data: bytes = None):
        """Speak text using TTS. If audio_data provided, use it directly."""
        logger.info(f"Speaking: '{text[:50]}...' ({len(text)} chars)")
        self._update_state(VoiceState.SPEAKING)
        print(f"\n💋 Claudette: {text}\n")

        try:
            # Use provided audio, cached audio, or generate new
            if audio_data is None:
                audio_data = self._audio_cache.get(text)
                if audio_data:
                    logger.debug("Using cached audio")

            if audio_data is None:
                logger.debug("Generating TTS audio...")
                start_time = datetime.now()
                audio_data = asyncio.run(self._synthesize_speech(text))
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.debug(f"TTS generated in {elapsed:.2f}s, {len(audio_data)} bytes")

            # Save to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            # Load and play audio with proper initialization
            pygame.mixer.music.load(temp_path)
            pygame.time.wait(100)  # Small delay to ensure audio is ready
            pygame.mixer.music.play()
            logger.debug("Audio playback started")

            # Wait for playback to actually start
            pygame.time.wait(50)

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)

            # Small delay after playback
            pygame.time.wait(100)
            logger.debug("Audio playback finished")

            # Cleanup
            os.unlink(temp_path)

        except Exception as e:
            logger.error(f"TTS error: {e}")
            print(f"TTS error: {e}", file=sys.stderr)

    def _generate_tts_async(self, text: str) -> bytes:
        """Generate TTS in a thread (for parallel execution)."""
        return asyncio.run(self._synthesize_speech(text))

    def _execute_claude(self, command: str) -> str:
        """Execute command with Claude CLI and return response."""
        logger.info(f"Executing Claude with command: '{command}'")
        self._update_state(VoiceState.THINKING)

        try:
            # Build the full prompt with personality
            full_prompt = f"{CLAUDETTE_SYSTEM_PROMPT}\n\nUser: {command}"
            logger.debug(f"Full prompt length: {len(full_prompt)} chars")

            start_time = datetime.now()
            result = subprocess.run(
                ["claude", "-p", full_prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            elapsed = (datetime.now() - start_time).total_seconds()

            response = result.stdout.strip()
            logger.info(f"Claude response ({elapsed:.2f}s): '{response[:100]}...' ({len(response)} chars)")

            if result.stderr:
                logger.warning(f"Claude stderr: {result.stderr}")

            return response
        except FileNotFoundError:
            logger.error("Claude CLI not found")
            return "I'm terribly sorry, sir, but it seems the Claude service isn't available at the moment."
        except subprocess.TimeoutExpired:
            logger.error("Claude CLI timed out")
            return "My apologies, sir. That request took rather longer than expected."
        except Exception as e:
            logger.error(f"Claude CLI error: {e}")
            return f"I'm afraid something went wrong, sir. Technical difficulties, you understand."

    def _detect_speech_segment(self) -> np.ndarray | None:
        """Listen for speech using VAD, return audio when speech ends."""
        chunk_samples = 512
        chunk_duration_sec = chunk_samples / self.sample_rate

        audio_buffer = []
        speech_detected = False
        silence_chunks = 0
        silence_chunks_threshold = int(self.silence_duration / chunk_duration_sec)
        min_speech_chunks = int((self.min_speech_ms / 1000) / chunk_duration_sec)
        speech_chunks = 0

        self.vad_model.reset_states()
        pending_audio = np.array([], dtype=np.float32)

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                chunk = chunk.flatten().astype(np.float32)
                pending_audio = np.concatenate([pending_audio, chunk])

                while len(pending_audio) >= chunk_samples:
                    vad_chunk = pending_audio[:chunk_samples]
                    pending_audio = pending_audio[chunk_samples:]

                    speech_prob = self.vad_model(
                        torch.from_numpy(vad_chunk),
                        self.sample_rate
                    ).item()

                    is_speech = speech_prob >= self.vad_threshold

                    if is_speech:
                        if not speech_detected:
                            self._update_state(VoiceState.RECORDING)
                        speech_detected = True
                        speech_chunks += 1
                        silence_chunks = 0
                        audio_buffer.append(vad_chunk)
                    elif speech_detected:
                        audio_buffer.append(vad_chunk)
                        silence_chunks += 1

                        if silence_chunks >= silence_chunks_threshold:
                            if speech_chunks >= min_speech_chunks:
                                return np.concatenate(audio_buffer)
                            else:
                                audio_buffer = []
                                speech_detected = False
                                speech_chunks = 0
                                silence_chunks = 0
                                self._update_state(VoiceState.LISTENING)

            except queue.Empty:
                continue

        return None

    def _process_audio(self, audio: np.ndarray):
        """Process recorded audio: transcribe and handle wake word."""
        logger.debug(f"Processing audio segment: {len(audio)} samples")
        self._update_state(VoiceState.PROCESSING)

        transcription = self._transcribe(audio)

        if not transcription:
            logger.warning("Empty transcription received")
            return

        transcription_lower = transcription.lower().strip()
        logger.info(f"Processing transcription: '{transcription}'")
        logger.debug(f"Lowercase: '{transcription_lower}'")
        logger.debug(f"Conversation mode: {self.conversation_mode}, Awaiting confirmation: {self.awaiting_confirmation}")
        print(f"\n👂 Heard: {transcription}")

        # Check for conversation-ending phrases
        end_phrases = ["thank you", "thanks", "that's all", "goodbye", "bye", "nevermind", "never mind"]
        for phrase in end_phrases:
            if phrase in transcription_lower:
                if self.conversation_mode:
                    self.conversation_mode = False
                    self.awaiting_confirmation = False
                    self._speak("My pleasure, sir.")
                    return

        # Check for confirmation/permission responses
        affirmative = ["yes", "yeah", "yep", "go ahead", "do it", "proceed", "please do", "approved", "confirmed", "affirmative"]
        negative = ["no", "nope", "don't", "stop", "cancel", "abort", "negative"]

        if self.conversation_mode and self.awaiting_confirmation:
            for phrase in affirmative:
                if phrase in transcription_lower:
                    print("   ✓ Confirmation received")
                    self.awaiting_confirmation = False
                    # Re-run the last command with explicit approval
                    if self.last_command:
                        self._speak("Very well, sir. Proceeding.")
                        approved_command = f"{self.last_command} - USER HAS CONFIRMED AND APPROVED THIS ACTION. Proceed with execution."
                        self._execute_and_respond(approved_command)
                    return

            for phrase in negative:
                if phrase in transcription_lower:
                    print("   ✓ Action declined")
                    self.awaiting_confirmation = False
                    self._speak("Understood, sir. Standing down.")
                    return

        # If in conversation mode, treat everything as a command
        if self.conversation_mode:
            print("   ✓ Conversation mode active")
            self._execute_and_respond(transcription)
            return

        # Check for wake word - include common Whisper transcription variations
        wake_word_variants = [
            self.wake_word,      # claudette
            "claudet",           # common mishearing
            "claudia",           # similar name
            "clodette",          # accent variation
            "cladette",          # mishearing
            "cloud",             # sometimes heard as
            "claud",             # partial
            "audit",             # stuffy nose / mishearing
            "audette",           # mishearing
            "kladette",          # accent
            "klodette",          # accent
            "plot it",           # mishearing
            "godette",           # mishearing
            "colette",           # similar name
            "clodet",            # mishearing
            "clawed",            # mishearing
            "plaudit",           # mishearing
            "hey claudette",     # with hey prefix
            "hey claudet",       # with hey prefix
            "okay claudette",    # with okay prefix
            "laudette",          # dropped 'c' mishearing
            "lodette",           # mishearing
            "la dette",          # split mishearing
        ]

        command = None
        wake_word_found = False
        matched_variant = None

        logger.debug(f"Checking for wake word variants: {wake_word_variants}")

        for variant in wake_word_variants:
            for suffix in [",", ".", "!", "?", " ", ""]:
                pattern = f"{variant}{suffix}"
                if transcription_lower.startswith(pattern):
                    wake_word_found = True
                    matched_variant = variant
                    # Extract command - find where the wake word ends
                    command = transcription[len(pattern):].strip()
                    command = command.lstrip(",.!? ")
                    logger.info(f"Wake word MATCH: variant='{variant}', pattern='{pattern}', command='{command}'")
                    break
            if wake_word_found:
                break

        if not wake_word_found:
            # Not for us - show it was ignored
            logger.info(f"No wake word found in: '{transcription_lower}'")
            print("   (No wake word detected)")
            return

        logger.info(f"Wake word detected: '{matched_variant}', extracted command: '{command}'")
        print(f"   ✓ Wake word detected: '{matched_variant}'")

        if not command or len(command) < 2:
            # Just the wake word - respond and listen for command
            self._speak("Yes, sir?")

            # Active listening mode - wait for the actual command
            print("   (Listening for command...)")
            self._update_state(VoiceState.RECORDING)

            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Listen for the follow-up command
            command_audio = self._detect_speech_segment()
            if command_audio is not None and len(command_audio) > 0:
                self._update_state(VoiceState.PROCESSING)
                command = self._transcribe(command_audio)
                if command:
                    print(f"\n👂 Command: {command}")
                    # Now execute the command
                    self._execute_and_respond(command)
        else:
            self._execute_and_respond(command)

    def _execute_and_respond(self, command: str):
        """Execute a command with Claude and speak the response."""
        # Store command for potential confirmation follow-up
        self.last_command = command

        # Run Claude and TTS acknowledgment in parallel
        # Start Claude in background
        claude_future = executor.submit(self._execute_claude, command)

        # Play acknowledgment from cache (instant) while Claude thinks
        self._speak("One moment, sir.")

        # Wait for Claude's response
        self._update_state(VoiceState.THINKING)
        response = claude_future.result()

        if response:
            # Generate TTS for response
            self._speak(response)

            # Check if response is asking for permission/confirmation
            response_lower = response.lower()
            permission_indicators = [
                "shall i", "should i", "would you like me to",
                "do you want me to", "may i", "can i proceed",
                "would you like", "do you approve", "is that okay",
                "confirm", "permission"
            ]
            for indicator in permission_indicators:
                if indicator in response_lower:
                    self.awaiting_confirmation = True
                    print("   (Awaiting confirmation: say 'yes' or 'no')")
                    break

        # Enter conversation mode for follow-ups
        self.conversation_mode = True
        if not self.awaiting_confirmation:
            print("   (Conversation mode: say follow-up or 'thank you' to end)")

    def run(self):
        """Main loop - listen, detect, transcribe, respond."""
        self.running = True

        print("\n" + "=" * 60)
        print("💋 Claudette - Your Sophisticated AI Assistant")
        print("=" * 60)
        print(f"Wake word: '{self.wake_word.capitalize()}'")
        print(f"Voice: {self.tts_voice}")
        print("Press Ctrl+C to exit")
        print("=" * 60 + "\n")

        # Greeting
        self._speak("Good day, sir. Claudette at your service.")

        def signal_handler(sig, frame):
            print("\n")
            self._speak("Goodbye, sir. It's been a pleasure.")
            self.running = False
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=512,
            callback=self._audio_callback
        ):
            self._update_state(VoiceState.LISTENING)

            while self.running:
                audio = self._detect_speech_segment()

                if audio is not None and len(audio) > 0:
                    self._process_audio(audio)
                    # Clear any audio that accumulated during processing
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break

                # Show appropriate listening state
                if self.conversation_mode:
                    self._update_state(VoiceState.LISTENING_CONVO)
                else:
                    self._update_state(VoiceState.LISTENING)


def main():
    """Entry point."""
    # Find config file
    config_path = find_config_file()
    if not config_path.exists():
        print(f"Error: Config file not found.")
        print(f"Please create a config.yaml in one of these locations:")
        print(f"  - {Path.cwd() / 'config.yaml'}")
        print(f"  - {Path.home() / '.config' / 'claudette' / 'config.yaml'}")
        print(f"\nYou can copy config.yaml.example as a starting point.")
        sys.exit(1)

    claudette = Claudette()
    claudette.run()


if __name__ == "__main__":
    main()
