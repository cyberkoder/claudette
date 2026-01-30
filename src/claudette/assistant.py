#!/usr/bin/env python3
"""
Claudette - A sophisticated AI voice assistant

A 1940s British bombshell with wit, charm, and intelligence.
Wake word: "Claudette" -> responds "Yes, sir?"
"""

import asyncio
import concurrent.futures
import io
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
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

# Import skills system
from .audio_processing import AudioProcessor
from .hotkey import HotkeyManager, get_default_hotkey
from .notifications import NotificationManager
from .offline import OfflineFallback
from .personalities import CLAUDETTE_DEFAULT, get_personality
from .skills import SkillManager
from .sounds import SoundEffects
from .tray import TrayIcon, WaveformWindow

# Set up logging - use current working directory for logs
log_dir = Path.cwd() / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"claudette_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("claudette")

# Reduce noise from other libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("pygame").setLevel(logging.WARNING)


def find_config_file(config_name: str = "config.yaml") -> Path:
    """Find config file in standard locations."""
    search_paths = [
        Path.cwd() / config_name,  # Current directory
        Path.home() / ".config" / "claudette" / config_name,  # User config
        Path(__file__).parent.parent.parent / config_name,  # Package root
    ]

    for path in search_paths:
        if path.exists():
            logger.info(f"Found config at: {path}")
            return path

    # Return default path (will error if not found)
    return search_paths[0]


# Thread pool for parallel operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# Default personality prompt (can be overridden in config)
CLAUDETTE_SYSTEM_PROMPT = CLAUDETTE_DEFAULT


class VoiceState:
    """Visual state indicators for the terminal."""

    LISTENING = "🎤 Listening for 'Claudette'..."
    LISTENING_CONVO = "💬 Listening (conversation active)..."
    RECORDING = "🔴 Recording..."
    PROCESSING = "⏳ Transcribing..."
    THINKING = "🧠 Thinking..."
    SPEAKING = "🗣️  Speaking..."


class ConversationMemory:
    """Persistent conversation memory for context across sessions."""

    def __init__(self, memory_file: Path | None = None, max_exchanges: int = 20):
        self.memory_file = memory_file or Path.cwd() / ".claudette_memory.json"
        self.max_exchanges = max_exchanges
        self.exchanges: list[dict] = []
        self._load()

    def _load(self):
        """Load conversation history from file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file) as f:
                    data = json.load(f)
                    self.exchanges = data.get("exchanges", [])[-self.max_exchanges :]
                    logger.debug(f"Loaded {len(self.exchanges)} exchanges from memory")
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load memory: {e}")
                self.exchanges = []

    def _save(self):
        """Save conversation history to file."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(
                    {
                        "exchanges": self.exchanges[-self.max_exchanges :],
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            logger.warning(f"Failed to save memory: {e}")

    def add_exchange(self, user_input: str, assistant_response: str):
        """Add a conversation exchange to memory."""
        self.exchanges.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "assistant": assistant_response,
            }
        )
        # Trim to max size
        self.exchanges = self.exchanges[-self.max_exchanges :]
        self._save()
        logger.debug(f"Saved exchange, total: {len(self.exchanges)}")

    def get_context(self, num_recent: int = 5) -> str:
        """Get recent conversation context for prompting."""
        recent = self.exchanges[-num_recent:]
        if not recent:
            return ""

        context_parts = ["Here is our recent conversation for context:"]
        for ex in recent:
            context_parts.append(f"User: {ex['user']}")
            context_parts.append(f"Claudette: {ex['assistant']}")

        return "\n".join(context_parts)

    def clear(self):
        """Clear conversation history."""
        self.exchanges = []
        self._save()
        logger.info("Conversation memory cleared")


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
        self.energy_threshold = self.config["vad"].get("energy_threshold", 0.005)  # Min audio energy

        # Whisper settings
        self.whisper_mode = self.config.get("whisper", {}).get("mode", "remote")
        self.whisper_url = self.config.get("whisper", {}).get("url", "http://localhost:9300/asr")
        self.whisper_language = self.config.get("whisper", {}).get("language", "en")
        self.whisper_model = None

        # Initialize local Whisper if configured
        if self.whisper_mode == "local":
            self._init_whisper()

        # Wake word settings
        wake_config = self.config.get("wake_word", {})
        if isinstance(wake_config, str):
            # Simple string format (backward compatible)
            self.wake_word = wake_config.lower()
            self.wake_word_variants = []
        else:
            # Dict format with variants
            self.wake_word = wake_config.get("word", "claudette").lower()
            self.wake_word_variants = [v.lower() for v in wake_config.get("variants", [])]

        # Default variants for common transcription errors
        self.default_wake_variants = [
            # Direct variations
            "claudet", "claudia", "clodette", "cladette", "claudete",
            "claudett", "clodett", "claudetta", "claudete",
            # Phonetic mishearings
            "cloud", "claud", "claude", "clowd", "klaud",
            "audit", "audette", "plaudit",
            "kladette", "klodette", "kludette",
            "plot it", "plot et", "clot it",
            "godette", "codette", "modette",
            "colette", "jolette", "polette",
            "clodet", "clawed", "clawdet",
            "laudette", "lodette", "la dette",
            "clue debt", "cloud debt", "claw debt",
            "clue det", "clo det", "glow det",
            # With prefixes
            "hey claudette", "hey claudet", "hey claude",
            "okay claudette", "ok claudette", "oh claudette",
            "hi claudette", "yo claudette",
            # Partial/fragmented
            "dett", "dette", "odette", "adette",
        ]

        # Fuzzy matching threshold (0-1, higher = stricter)
        self.wake_word_fuzzy_threshold = 0.7

        # TTS settings - British English female voice
        self.tts_voice = self.config.get("tts", {}).get("voice", "en-GB-SoniaNeural")
        self.tts_rate = self.config.get("tts", {}).get("rate", "+0%")
        self.tts_pitch = self.config.get("tts", {}).get("pitch", "+0Hz")

        # Personality settings
        personality_config = self.config.get("personality", {})
        personality_name = personality_config.get("preset", "claudette")
        custom_prompt = personality_config.get("custom_prompt")
        if custom_prompt:
            self.system_prompt = custom_prompt
            logger.info("Using custom personality prompt")
        else:
            self.system_prompt = get_personality(personality_name)
            logger.info(f"Using personality preset: {personality_name}")

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

        # Conversation memory for context across sessions
        memory_config = self.config.get("memory", {})
        if memory_config.get("enabled", True):
            memory_file = memory_config.get("file")
            if memory_file:
                memory_file = Path(memory_file).expanduser()
            max_exchanges = memory_config.get("max_exchanges", 20)
            self.memory = ConversationMemory(memory_file, max_exchanges)
            logger.info(
                f"Conversation memory enabled ({len(self.memory.exchanges)} exchanges loaded)"
            )
        else:
            self.memory = None
            logger.info("Conversation memory disabled")

        # Initialize skills/plugins system
        skills_dir = self.config.get("skills", {}).get("directory")
        if skills_dir:
            skills_dir = Path(skills_dir).expanduser()
        self.skills = SkillManager(skills_dir)
        logger.info(f"Loaded {len(self.skills.skills)} skills")

        # Initialize sound effects
        sounds_config = self.config.get("sounds", {})
        self.sounds = SoundEffects(
            enabled=sounds_config.get("enabled", True), volume=sounds_config.get("volume", 0.3)
        )

        # Initialize hotkey support
        hotkey_config = self.config.get("hotkey", {})
        self.hotkey_triggered = threading.Event()
        self.hotkey_manager = HotkeyManager(
            enabled=hotkey_config.get("enabled", True),
            hotkey=hotkey_config.get("key", get_default_hotkey()),
            callback=self._on_hotkey_pressed,
        )

        # Initialize system tray
        tray_config = self.config.get("tray", {})
        self.tray = TrayIcon(
            enabled=tray_config.get("enabled", True),
            on_activate=self._on_hotkey_pressed,
            on_quit=self._shutdown,
        )

        # Initialize waveform window (optional)
        self.waveform = WaveformWindow(enabled=tray_config.get("waveform", False))

        # Initialize desktop notifications
        notify_config = self.config.get("notifications", {})
        self.notifications = NotificationManager(
            enabled=notify_config.get("enabled", False)  # Off by default
        )

        # Initialize audio processor for noise reduction
        audio_processing_config = self.config.get("audio_processing", {})
        self.audio_processor = AudioProcessor(
            sample_rate=self.sample_rate,
            noise_reduce=audio_processing_config.get("noise_reduce", True),
            high_pass_cutoff=audio_processing_config.get("high_pass_cutoff", 80.0),
            normalize=audio_processing_config.get("normalize", True),
        )

        # Initialize offline fallback
        offline_config = self.config.get("offline", {})
        self.offline = OfflineFallback(enabled=offline_config.get("enabled", True))

        # Initialize web dashboard (optional)
        self.web_server = None
        self.state_manager = None
        web_config = self.config.get("web", {})
        if web_config.get("enabled", False):
            self._init_web_server(web_config)

        # Initialize VAD model
        self._init_vad()

        # Pre-cache common phrases
        self._precache_phrases()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = find_config_file(config_path)
        logger.info(f"Loading config from: {config_file}")
        with open(config_file) as f:
            return yaml.safe_load(f)

    def _init_web_server(self, web_config: dict):
        """Initialize web dashboard server."""
        try:
            from .web import ClaudetteStateManager, WebServer

            host = web_config.get("host", "127.0.0.1")
            port = web_config.get("port", 8420)

            # Create state manager and set claudette reference
            self.state_manager = ClaudetteStateManager()
            self.state_manager.set_claudette(self)

            # Create web server
            self.web_server = WebServer(
                state_manager=self.state_manager,
                host=host,
                port=port,
            )
            logger.info(f"Web dashboard configured at http://{host}:{port}")

        except ImportError as e:
            logger.warning(
                f"Web dashboard dependencies not installed: {e}. "
                "Install with: pip install claudette-voice[web]"
            )
            self.web_server = None
            self.state_manager = None

    def _init_vad(self):
        """Initialize Silero VAD model with optional GPU acceleration."""
        # Determine device for VAD
        vad_device = self.config.get("vad", {}).get("device", "auto")
        if vad_device == "auto":
            vad_device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Silero VAD model on {vad_device}...")
        self._print_status(f"Loading VAD model ({vad_device})...")

        # Load Silero VAD
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )

        # Move model to appropriate device
        self.vad_device = vad_device
        if vad_device == "cuda":
            self.vad_model = self.vad_model.to(vad_device)
            logger.info("VAD model moved to CUDA")

        logger.info(f"VAD model loaded successfully on {vad_device}")
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

        self.whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)

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
            "Still working on it, sir.",
            "Looking into that now, sir.",
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

        # Update tray icon state
        tray_states = {
            VoiceState.LISTENING: "listening",
            VoiceState.LISTENING_CONVO: "listening",
            VoiceState.RECORDING: "recording",
            VoiceState.PROCESSING: "processing",
            VoiceState.THINKING: "processing",
            VoiceState.SPEAKING: "speaking",
        }
        tray_state = tray_states.get(state, "idle")
        self.tray.set_state(tray_state)

        # Update web dashboard state manager
        if self.state_manager:
            self.state_manager.update_state(state)
            self.state_manager.update_conversation_mode(self.conversation_mode)
            self.state_manager.update_awaiting_confirmation(self.awaiting_confirmation)

    def _shutdown(self):
        """Graceful shutdown."""
        self.running = False

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream - adds audio to queue."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

        # Update audio level for web dashboard
        if self.state_manager:
            # Calculate RMS audio level
            audio_level = np.sqrt(np.mean(indata**2))
            self.state_manager.update_audio_level(float(audio_level))

    def _on_hotkey_pressed(self):
        """Called when activation hotkey is pressed."""
        logger.info("Hotkey activation triggered")
        self.hotkey_triggered.set()

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

        # Apply audio processing (noise reduction, filtering)
        audio_data = self.audio_processor.process(audio_data)

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

            # Use initial_prompt to bias Whisper toward recognizing the wake word
            # This helps with uncommon names like "Claudette"
            wake_word_prompt = f"{self.wake_word.capitalize()}, "

            # Transcribe
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language=self.whisper_language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=wake_word_prompt,
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
            response = requests.post(self.whisper_url, files=files, params=params, timeout=30)
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
            text, self.tts_voice, rate=self.tts_rate, pitch=self.tts_pitch
        )

        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for streaming TTS."""
        import re

        # Split on sentence-ending punctuation, keeping the punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Filter out empty strings and strip whitespace
        return [s.strip() for s in sentences if s.strip()]

    def _play_audio_file(self, file_path: str):
        """Play an audio file and wait for completion."""
        pygame.mixer.music.load(file_path)
        pygame.time.wait(50)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
        pygame.time.wait(50)

    def _speak(self, text: str, audio_data: bytes = None):
        """Speak text using TTS with streaming for long responses."""
        logger.info(f"Speaking: '{text[:50]}...' ({len(text)} chars)")
        self._update_state(VoiceState.SPEAKING)
        print(f"\n💋 Claudette: {text}\n")

        # Update web dashboard with response
        if self.state_manager:
            self.state_manager.update_last_response(text)

        try:
            # Check cache first
            if audio_data is None:
                audio_data = self._audio_cache.get(text)
                if audio_data:
                    logger.debug("Using cached audio")

            # If we have cached/provided audio, play it directly
            if audio_data is not None:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name
                self._play_audio_file(temp_path)
                os.unlink(temp_path)
                logger.debug("Audio playback finished")
                return

            # For long text, use streaming (sentence by sentence)
            sentences = self._split_into_sentences(text)

            if len(sentences) <= 1:
                # Short text - generate and play directly
                logger.debug("Generating TTS audio...")
                start_time = datetime.now()
                audio_data = asyncio.run(self._synthesize_speech(text))
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.debug(f"TTS generated in {elapsed:.2f}s, {len(audio_data)} bytes")

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name
                self._play_audio_file(temp_path)
                os.unlink(temp_path)
            else:
                # Streaming mode: generate and play sentence by sentence
                logger.debug(f"Streaming TTS for {len(sentences)} sentences")

                # Generate first sentence immediately
                first_audio = asyncio.run(self._synthesize_speech(sentences[0]))
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(first_audio)
                    first_file = f.name

                # Start generating remaining sentences in background
                remaining_futures = []
                for sentence in sentences[1:]:
                    future = executor.submit(
                        lambda s=sentence: asyncio.run(self._synthesize_speech(s))
                    )
                    remaining_futures.append((sentence, future))

                # Play first sentence
                logger.debug(f"Playing sentence 1/{len(sentences)}")
                self._play_audio_file(first_file)
                os.unlink(first_file)

                # Play remaining sentences as they complete
                for i, (_sentence, future) in enumerate(remaining_futures, 2):
                    audio_data = future.result()
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        f.write(audio_data)
                        temp_path = f.name
                    logger.debug(f"Playing sentence {i}/{len(sentences)}")
                    self._play_audio_file(temp_path)
                    os.unlink(temp_path)

            logger.debug("Audio playback finished")

        except Exception as e:
            logger.error(f"TTS error: {e}")
            print(f"TTS error: {e}", file=sys.stderr)

    def _generate_tts_async(self, text: str) -> bytes:
        """Generate TTS in a thread (for parallel execution)."""
        return asyncio.run(self._synthesize_speech(text))

    def _execute_claude(self, command: str) -> str:
        """Execute command with Claude CLI and return response with streaming progress."""
        logger.info(f"Executing Claude with command: '{command}'")
        self._update_state(VoiceState.THINKING)

        # Start tracking Claude activity
        if self.state_manager:
            self.state_manager.start_claude_activity(command)

        try:
            # Build the full prompt with personality and conversation context
            prompt_parts = [self.system_prompt]

            # Add conversation memory context if available
            if self.memory and self.memory.exchanges:
                context = self.memory.get_context(num_recent=5)
                if context:
                    prompt_parts.append(context)

            prompt_parts.append(f"User: {command}")
            full_prompt = "\n\n".join(prompt_parts)
            logger.debug(f"Full prompt length: {len(full_prompt)} chars")

            start_time = datetime.now()

            # Use Popen for streaming output
            process = subprocess.Popen(
                ["claude", "-p", full_prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            output_lines = []
            last_progress_speech = time.time()
            progress_spoken = False

            # Stream stdout
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    line = line.rstrip()
                    output_lines.append(line)

                    # Update state manager with progress
                    if self.state_manager:
                        # Detect status from output patterns
                        line_lower = line.lower()
                        if any(x in line_lower for x in ["searching", "looking", "finding"]):
                            self.state_manager.update_claude_status("searching")
                        elif any(x in line_lower for x in ["reading", "examining", "analyzing"]):
                            self.state_manager.update_claude_status("reading")
                        elif any(x in line_lower for x in ["writing", "creating", "generating"]):
                            self.state_manager.update_claude_status("writing")

                        self.state_manager.add_claude_progress(line)
                        self.state_manager.update_claude_output("\n".join(output_lines[-20:]))

                    # Speak brief progress update every 5 seconds
                    now = time.time()
                    if not progress_spoken and (now - last_progress_speech) > 5:
                        if len(output_lines) > 3:
                            # Quick progress speech
                            executor.submit(self._speak_progress, "Still working on it, sir.")
                            last_progress_speech = now
                            progress_spoken = True

            # Get any remaining stderr
            stderr = process.stderr.read()
            if stderr:
                logger.warning(f"Claude stderr: {stderr}")

            elapsed = (datetime.now() - start_time).total_seconds()
            response = "\n".join(output_lines).strip()

            logger.info(
                f"Claude response ({elapsed:.2f}s): '{response[:100]}...' ({len(response)} chars)"
            )

            # End Claude activity tracking
            if self.state_manager:
                self.state_manager.update_claude_output(response)
                self.state_manager.end_claude_activity()

            # Save to conversation memory
            if self.memory and response:
                self.memory.add_exchange(command, response)

            return response
        except FileNotFoundError:
            logger.error("Claude CLI not found")
            if self.state_manager:
                self.state_manager.end_claude_activity()
            self.sounds.play_error()
            return "I'm terribly sorry, sir, but it seems the Claude service isn't available at the moment."
        except subprocess.TimeoutExpired:
            logger.error("Claude CLI timed out")
            if self.state_manager:
                self.state_manager.end_claude_activity()
            self.sounds.play_error()
            return "My apologies, sir. That request took rather longer than expected."
        except Exception as e:
            logger.error(f"Claude CLI error: {e}")
            if self.state_manager:
                self.state_manager.end_claude_activity()
            self.sounds.play_error()
            return "I'm afraid something went wrong, sir. Technical difficulties, you understand."

    def _speak_progress(self, text: str):
        """Speak a brief progress update (runs in background thread)."""
        try:
            audio_data = self._audio_cache.get(text)
            if audio_data is None:
                audio_data = asyncio.run(self._synthesize_speech(text))
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            self._play_audio_file(temp_path)
            os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Progress speech error: {e}")

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

        # Pre-buffer to capture audio BEFORE VAD triggers (catches wake word)
        pre_buffer_seconds = 0.8  # Keep 0.8 seconds of audio before speech detected
        pre_buffer_chunks = int(pre_buffer_seconds / chunk_duration_sec)
        pre_buffer = []

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

                    # Always add to pre-buffer (rolling buffer)
                    pre_buffer.append(vad_chunk)
                    if len(pre_buffer) > pre_buffer_chunks:
                        pre_buffer.pop(0)

                    # Move tensor to VAD device (CPU or CUDA)
                    vad_tensor = torch.from_numpy(vad_chunk)
                    if self.vad_device == "cuda":
                        vad_tensor = vad_tensor.to(self.vad_device)

                    speech_prob = self.vad_model(vad_tensor, self.sample_rate).item()

                    is_speech = speech_prob >= self.vad_threshold

                    if is_speech:
                        if not speech_detected:
                            self._update_state(VoiceState.RECORDING)
                            self.sounds.play_record()
                            # Include pre-buffer when speech starts (catches wake word)
                            audio_buffer = list(pre_buffer)
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

    def _detect_wake_word(self, transcription: str, transcription_lower: str) -> tuple[str | None, str | None, bool]:
        """Detect wake word in transcription using multiple strategies.

        Returns:
            Tuple of (command, matched_variant, near_miss) where near_miss indicates
            if something close to the wake word was detected but not close enough.
        """
        from difflib import SequenceMatcher

        # Combine all variants
        wake_word_variants = [self.wake_word]
        wake_word_variants.extend(self.wake_word_variants)
        wake_word_variants.extend(self.default_wake_variants)
        wake_word_variants = list(dict.fromkeys(wake_word_variants))

        # Clean transcription - remove extra punctuation at start
        clean_trans = transcription_lower.lstrip(",.!?;:'\" ")

        # Track best near-miss for potential "did you call me?" response
        best_similarity = 0
        near_miss_word = None

        # Strategy 1: Exact match at start
        for variant in wake_word_variants:
            for suffix in [",", ".", "!", "?", " ", ""]:
                pattern = f"{variant}{suffix}"
                if clean_trans.startswith(pattern):
                    command = transcription[len(transcription) - len(clean_trans) + len(pattern):].strip()
                    command = command.lstrip(",.!? ")
                    logger.info(f"Wake word EXACT match: '{variant}'")
                    return command, variant, False

        # Strategy 2: Check first 5 words for exact variant match
        words = clean_trans.split()[:5]
        for i, word in enumerate(words):
            # Clean the word of punctuation
            clean_word = word.strip(",.!?;:'\"")
            if clean_word in wake_word_variants:
                # Found wake word - command is everything after
                command = " ".join(transcription.split()[i+1:]).strip()
                command = command.lstrip(",.!? ")
                logger.info(f"Wake word found at word {i}: '{clean_word}'")
                return command, clean_word, False

        # Strategy 3: Fuzzy match on first 3 words
        for i, word in enumerate(words[:3]):
            clean_word = word.strip(",.!?;:'\"")
            if len(clean_word) < 3:
                continue

            # Check similarity to main wake word
            similarity = SequenceMatcher(None, clean_word, self.wake_word).ratio()

            # Track near misses (50-70% similar)
            if similarity > best_similarity:
                best_similarity = similarity
                near_miss_word = clean_word

            if similarity >= self.wake_word_fuzzy_threshold:
                command = " ".join(transcription.split()[i+1:]).strip()
                command = command.lstrip(",.!? ")
                logger.info(f"Wake word FUZZY match: '{clean_word}' ~ '{self.wake_word}' ({similarity:.2f})")
                return command, clean_word, False

            # Also check against shorter variants for fuzzy
            for variant in ["claudet", "claude", "claud"]:
                similarity = SequenceMatcher(None, clean_word, variant).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    near_miss_word = clean_word
                if similarity >= 0.8:  # Higher threshold for short variants
                    command = " ".join(transcription.split()[i+1:]).strip()
                    command = command.lstrip(",.!? ")
                    logger.info(f"Wake word FUZZY match: '{clean_word}' ~ '{variant}' ({similarity:.2f})")
                    return command, clean_word, False

        # Strategy 4: Check if any word contains the wake word
        for i, word in enumerate(words[:4]):
            clean_word = word.strip(",.!?;:'\"")
            if self.wake_word in clean_word or "claudet" in clean_word or "claude" in clean_word:
                command = " ".join(transcription.split()[i+1:]).strip()
                command = command.lstrip(",.!? ")
                logger.info(f"Wake word SUBSTRING match in: '{clean_word}'")
                return command, clean_word, False

        # Check for near miss (sounded close but not close enough)
        # Only trigger if transcription is short (likely an attempt to call her)
        is_near_miss = (best_similarity >= 0.5 and len(words) <= 4) or \
                       (len(words) == 1 and len(words[0]) >= 4 and best_similarity >= 0.4)

        if is_near_miss:
            logger.info(f"Near miss detected: '{near_miss_word}' ({best_similarity:.2f} similar)")

        return None, None, is_near_miss

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

        # Update web dashboard with transcription
        if self.state_manager:
            self.state_manager.update_last_transcription(transcription)
        logger.debug(
            f"Conversation mode: {self.conversation_mode}, Awaiting confirmation: {self.awaiting_confirmation}"
        )
        print(f"\n👂 Heard: {transcription}")

        # Check for conversation-ending phrases
        end_phrases = [
            "thank you",
            "thanks",
            "that's all",
            "goodbye",
            "bye",
            "nevermind",
            "never mind",
        ]
        for phrase in end_phrases:
            if phrase in transcription_lower:
                if self.conversation_mode:
                    self.conversation_mode = False
                    self.awaiting_confirmation = False
                    self._speak("My pleasure, sir.")
                    return

        # Check for confirmation/permission responses
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

        # Check for wake word using improved detection
        command, matched_variant, near_miss = self._detect_wake_word(transcription, transcription_lower)

        if matched_variant is None:
            if near_miss:
                # Sounded close - ask for clarification
                import random
                clarifications = [
                    "I'm sorry, I didn't quite catch that, sir.",
                    "Did you call for me, sir?",
                    "Pardon me, sir?",
                    "I thought I heard my name, sir.",
                ]
                logger.info(f"Near miss - asking for clarification")
                print("   (Near miss - asking for clarification)")
                self._speak(random.choice(clarifications))
            else:
                # Not for us - show it was ignored
                logger.info(f"No wake word found in: '{transcription_lower}'")
                print("   (No wake word detected)")
            return

        logger.info(f"Wake word detected: '{matched_variant}', extracted command: '{command}'")
        print(f"   ✓ Wake word detected: '{matched_variant}'")
        self.sounds.play_wake()
        self.notifications.notify_wake()

        if not command or len(command) < 2:
            # Just the wake word - greet and listen for command
            import random
            greetings = ["Yes, sir?", "Yes, sir?", "At your service, sir.", "Sir?", "How may I assist you, sir?"]
            self._speak(random.choice(greetings))

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
            # Wake word + command - acknowledge and execute
            import random
            acknowledgments = ["Right away, sir.", "On it, sir.", "Certainly, sir.", "Very good, sir.", "Of course, sir."]
            self._speak(random.choice(acknowledgments))
            self._execute_and_respond(command)

    def _execute_and_respond(self, command: str):
        """Execute a command with Claude and speak the response."""
        # Store command for potential confirmation follow-up
        self.last_command = command

        # Try skills first (instant response for built-in commands)
        skill_response = self.skills.execute(command, self)
        if skill_response:
            logger.info(f"Skill handled command: '{command}'")
            self._speak(skill_response)
            # Still enter conversation mode
            self.conversation_mode = True
            print("   (Conversation mode: say follow-up or 'thank you' to end)")
            return

        # Check network connectivity
        if not self.offline.is_online():
            logger.warning("Offline - using fallback response")
            offline_response = self.offline.get_offline_response(command)
            if offline_response:
                self._speak(offline_response)
                self.conversation_mode = True
                print("   (Offline mode - limited functionality)")
                return

        # Run Claude and TTS acknowledgment in parallel
        # Start Claude in background
        claude_future = executor.submit(self._execute_claude, command)

        # Play acknowledgment from cache (instant) while Claude thinks
        self._speak("One moment, sir.")

        # Wait for Claude's response
        self._update_state(VoiceState.THINKING)
        self.sounds.play_process()
        response = claude_future.result()

        if response:
            # Generate TTS for response
            self._speak(response)
            self.sounds.play_done()

            # Check if response is asking for permission/confirmation
            response_lower = response.lower()
            permission_indicators = [
                "shall i",
                "should i",
                "would you like me to",
                "do you want me to",
                "may i",
                "can i proceed",
                "would you like",
                "do you approve",
                "is that okay",
                "confirm",
                "permission",
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
        if self.hotkey_manager.enabled:
            print(f"Hotkey: {self.hotkey_manager.hotkey}")
        print("Press Ctrl+C to exit")
        print("=" * 60 + "\n")

        # Start hotkey listener
        self.hotkey_manager.start()

        # Start system tray
        self.tray.start()
        self.waveform.start()

        # Start web dashboard server
        if self.web_server:
            self.web_server.start()
            print(f"Web dashboard: {self.web_server.url}")

        # Greeting
        self._speak("Good day, sir. Claudette at your service.")
        self.notifications.notify_started()

        def signal_handler(sig, frame):
            print("\n")
            self.hotkey_manager.stop()
            self.tray.stop()
            self.waveform.stop()
            if self.web_server:
                self.web_server.stop()
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
            callback=self._audio_callback,
        ):
            self._update_state(VoiceState.LISTENING)

            while self.running:
                # Check for hotkey activation
                if self.hotkey_triggered.is_set():
                    self.hotkey_triggered.clear()
                    logger.info("Hotkey triggered - listening for command")
                    print("\n🎹 Hotkey activated!")
                    self.sounds.play_wake()
                    self._speak("Yes, sir?")

                    # Clear audio queue
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break

                    # Listen for command (like active listening after wake word)
                    self._update_state(VoiceState.RECORDING)
                    command_audio = self._detect_speech_segment()
                    if command_audio is not None and len(command_audio) > 0:
                        self._update_state(VoiceState.PROCESSING)
                        command = self._transcribe(command_audio)
                        if command:
                            print(f"\n👂 Command: {command}")
                            self._execute_and_respond(command)
                    continue

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
        print("Error: Config file not found.")
        print("Please create a config.yaml in one of these locations:")
        print(f"  - {Path.cwd() / 'config.yaml'}")
        print(f"  - {Path.home() / '.config' / 'claudette' / 'config.yaml'}")
        print("\nYou can copy config.yaml.example as a starting point.")
        sys.exit(1)

    claudette = Claudette()
    claudette.run()


if __name__ == "__main__":
    main()
