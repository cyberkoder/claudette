<p align="center">
  <img src="logo.png" alt="Claudette" width="400">
</p>

<h1 align="center">Claudette</h1>

<p align="center">
  <em>A sophisticated AI voice assistant with 1940s British charm</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/claudette-voice/"><img src="https://img.shields.io/pypi/v/claudette-voice.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/claudette-voice/"><img src="https://img.shields.io/pypi/pyversions/claudette-voice.svg" alt="Python versions"></a>
  <a href="https://github.com/cyberkoder/claudette/blob/main/LICENSE"><img src="https://img.shields.io/github/license/cyberkoder/claudette" alt="License"></a>
  <a href="https://github.com/cyberkoder/claudette/actions"><img src="https://github.com/cyberkoder/claudette/workflows/CI/badge.svg" alt="CI Status"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#usage">Usage</a> •
  <a href="#skills">Skills</a> •
  <a href="#contributing">Contributing</a> •
  <a href="https://ko-fi.com/cyberkoder">Support</a>
</p>

<p align="center">
  <a href="https://ko-fi.com/cyberkoder">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi">
  </a>
</p>

---

**Claudette** is a voice-activated assistant that brings the elegance of a 1940s British bombshell to your command line. She listens for her wake word, transcribes your speech using Whisper, and responds with wit and charm through Claude CLI.

## Features

- **Wake Word Detection** - Say "Claudette" to activate (with fuzzy matching for accents)
- **Voice Activity Detection** - Uses Silero VAD with GPU acceleration
- **Speech-to-Text** - Local (faster-whisper) or remote Whisper API
- **Natural Conversation** - Maintains conversation memory across sessions
- **Text-to-Speech** - Streaming TTS with multiple voice options
- **Sound Effects** - Audio feedback for state changes
- **Hotkey Activation** - Ctrl+Shift+C (or Cmd+Shift+C on Mac)
- **System Tray** - Status icon with waveform visualization
- **Desktop Notifications** - Optional notification support
- **System Commands** - Volume, battery, screenshots, and more
- **Custom Skills** - Extensible plugin system
- **Multiple Personalities** - Switch between different AI personas
- **Noise Reduction** - Audio processing for cleaner recognition
- **Offline Mode** - Basic functionality when network unavailable

## Architecture

```
[Microphone] → [VAD Detection] → [Noise Filter] → [Whisper] → [Claude CLI]
                  (Silero)                         (local/remote)     ↓
                                                                   [TTS]
                                                                      ↓
                                                                [Speaker]
```

## Requirements

- Python 3.10+
- [Claude CLI](https://github.com/anthropics/claude-code) installed and configured
- PortAudio library (`portaudio19-dev` on Ubuntu/Debian)
- Working microphone

## Installation

### From PyPI

```bash
# Basic installation
pip install claudette-voice

# With local Whisper (no server needed)
pip install claudette-voice[local]

# With all optional features
pip install claudette-voice[all]

# Install system dependencies (Ubuntu/Debian)
sudo apt install portaudio19-dev
```

### Optional Extras

| Extra | Description | Install Command |
|-------|-------------|-----------------|
| `local` | Local Whisper transcription | `pip install claudette-voice[local]` |
| `hotkey` | Global hotkey support | `pip install claudette-voice[hotkey]` |
| `system` | System info commands | `pip install claudette-voice[system]` |
| `tray` | System tray with waveform | `pip install claudette-voice[tray]` |
| `notifications` | Desktop notifications | `pip install claudette-voice[notifications]` |
| `audio` | Noise reduction | `pip install claudette-voice[audio]` |
| `all` | Everything above + dev tools | `pip install claudette-voice[all]` |

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/cyberkoder/claudette.git
cd claudette

# Install system dependencies (Ubuntu/Debian)
sudo apt install portaudio19-dev

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with all extras
pip install -e ".[all]"

# Copy and configure settings
cp config.yaml.example config.yaml
```

## Configuration

Create a `config.yaml` file in your working directory:

```yaml
whisper:
  mode: "local"              # "local" or "remote"
  model: "base"              # tiny, base, small, medium, large-v3
  # url: "http://server:9300/asr"  # For remote mode

wake_word:
  word: "claudette"
  # variants: ["cloud it"]   # Add custom variants for your accent

vad:
  threshold: 0.5
  silence_duration: 1.5
  device: "auto"             # "auto", "cuda", or "cpu"

tts:
  voice: "en-GB-SoniaNeural"
  rate: "+0%"
  pitch: "+0Hz"

memory:
  enabled: true
  max_exchanges: 20

sounds:
  enabled: true
  volume: 0.3

hotkey:
  enabled: true
  # key: "<ctrl>+<shift>+c"  # Customize hotkey

tray:
  enabled: true
  waveform: false            # Floating waveform window

notifications:
  enabled: false

personality:
  preset: "claudette"        # claudette, professional, friendly, butler, pirate

audio_processing:
  noise_reduce: true
  high_pass_cutoff: 80.0
  normalize: true

offline:
  enabled: true
```

## Usage

```bash
# Run Claudette
claudette

# Or run as module
python -m claudette
```

### Voice Commands

1. **Activate**: Say "Claudette" or press Ctrl+Shift+C
2. **Command**: State your request after "Yes, sir?"
3. **Follow-up**: Continue without wake word
4. **Confirm**: Say "yes" or "go ahead" when asked
5. **End**: Say "thank you" or "goodbye"

### Example Interaction

```
You: "Claudette"
Claudette: "Yes, sir?"

You: "What time is it?"
Claudette: "It's half past 3 in the afternoon, sir."

You: "How's my battery?"
Claudette: "Battery is at 72% with about 3 hours remaining, sir."

You: "Thank you"
Claudette: "My pleasure, sir."
```

## Skills

Claudette includes many built-in skills:

| Skill | Trigger Examples |
|-------|------------------|
| Time | "what time is it", "current time" |
| Date | "what's the date", "what day is it" |
| Status | "system status", "how are you" |
| System Info | "how is my computer", "system info" |
| Battery | "battery level", "am I plugged in" |
| Volume | "volume up", "mute", "what's the volume" |
| Lock Screen | "lock screen", "lock computer" |
| Screenshot | "take a screenshot" |
| Voice Change | "list voices", "change voice to libby" |
| Personality | "list personalities", "change personality to butler" |
| Clear Memory | "clear memory", "forget everything" |
| List Skills | "what can you do", "list skills" |

### Custom Skills

Create custom skills in a `skills/` directory:

```python
from claudette import Skill

class WeatherSkill(Skill):
    name = "weather"
    description = "Check the weather"
    triggers = ["what's the weather", "weather forecast"]

    def execute(self, command: str, claudette) -> str:
        # Your implementation here
        return "It's a lovely day, sir."
```

## Project Structure

```
claudette/
├── src/claudette/
│   ├── __init__.py
│   ├── assistant.py        # Main assistant
│   ├── skills.py           # Skills system
│   ├── sounds.py           # Sound effects
│   ├── hotkey.py           # Hotkey support
│   ├── tray.py             # System tray
│   ├── notifications.py    # Desktop notifications
│   ├── personalities.py    # AI personalities
│   ├── audio_processing.py # Noise reduction
│   └── offline.py          # Offline fallback
├── skills/                  # Custom skills directory
├── config.yaml.example
├── pyproject.toml
└── README.md
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint code
ruff check src/
```

## Publishing

For maintainers - publishing to PyPI:

```bash
# Build the package
python -m build

# Check the package
twine check dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

Or use GitHub Releases - creating a release automatically publishes to PyPI.

## Troubleshooting

### "PortAudio library not found"
```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev

# macOS
brew install portaudio

# Fedora
sudo dnf install portaudio-devel
```

### Wake word not detected
- Speak clearly and pause slightly after "Claudette"
- Check logs in `logs/` directory
- Add custom variants: say "add wake word cloud" if being misheard

### Audio cutting off
- Adjust `silence_duration` in config.yaml
- Check microphone levels

## Support the Project

If you find Claudette useful, consider buying me a coffee!

<a href="https://ko-fi.com/cyberkoder">
  <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi">
</a>

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Anthropic](https://anthropic.com) for Claude
- [Silero](https://github.com/snakers4/silero-vad) for VAD model
- [Edge TTS](https://github.com/rany2/edge-tts) for text-to-speech
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local transcription
- The 1940s for the aesthetic inspiration

---

<p align="center">
  <em>"Good day, sir. Claudette at your service."</em>
</p>
