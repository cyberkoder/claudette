<p align="center">
  <img src="logo.png" alt="Claudette" width="400">
</p>

<h1 align="center">Claudette</h1>

<p align="center">
  <em>A sophisticated AI voice assistant with 1940s British charm</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#usage">Usage</a> •
  <a href="#contributing">Contributing</a> •
  <a href="https://ko-fi.com/cyberkoder">Support</a>
</p>

<p align="center">
  <a href="https://ko-fi.com/cyberkoder">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi">
  </a>
</p>

---

**Claudette** is a voice-activated assistant that brings the elegance of a 1940s British bombshell to your command line. She listens for her wake word, transcribes your speech using a Whisper API, and responds with wit and charm through Claude CLI.

## Features

- **Wake Word Detection** - Say "Claudette" to activate
- **Voice Activity Detection** - Uses Silero VAD for accurate speech detection
- **Speech-to-Text** - Integrates with Whisper API (local or remote)
- **Natural Conversation** - Maintains conversation context for follow-up questions
- **Text-to-Speech** - British female voice using Edge TTS
- **Permission Handling** - Asks for confirmation before sensitive actions
- **Personality** - Sophisticated 1940s British charm with wit and professionalism

## Architecture

```
[Microphone] → [VAD Detection] → [Record Speech] → [Whisper API] → [Claude CLI]
                  (Silero)        (until silence)    (your server)      ↓
                                                                    [Edge TTS]
                                                                        ↓
                                                                   [Speaker]
```

## Requirements

- Python 3.10+
- [Claude CLI](https://github.com/anthropics/claude-code) installed and configured
- Whisper API endpoint (e.g., [faster-whisper-server](https://github.com/fedirz/faster-whisper-server))
- PortAudio library (`portaudio19-dev` on Ubuntu/Debian)
- Working microphone

## Installation

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

# Install in development mode
pip install -e ".[dev]"

# Copy and configure settings
cp config.yaml.example config.yaml
# Edit config.yaml with your Whisper API URL
```

### From PyPI (Coming Soon)

```bash
pip install claudette-voice
```

## Configuration

Create a `config.yaml` file in your working directory:

```yaml
whisper:
  url: "http://your-whisper-server:9300/asr"
  language: "en"

audio:
  sample_rate: 16000
  channels: 1

wake_word: "claudette"

vad:
  threshold: 0.5           # Speech probability threshold (0.0-1.0)
  min_speech_ms: 250       # Minimum speech duration to trigger
  silence_duration: 1.5    # Seconds of silence to end recording

tts:
  voice: "en-GB-SoniaNeural"  # British female voice
  rate: "+0%"                  # Speech rate adjustment
  pitch: "+0Hz"                # Pitch adjustment
```

### Available TTS Voices

British English female voices:
- `en-GB-SoniaNeural` (default) - Professional, clear
- `en-GB-LibbyNeural` - Warm, friendly
- `en-GB-MaisieNeural` - Young, energetic

## Usage

```bash
# Activate your virtual environment
source venv/bin/activate

# Run Claudette
claudette

# Or run directly
python -m claudette
```

### Voice Commands

1. **Activate**: Say "Claudette" to wake her up
2. **Command**: After "Yes, sir?", state your request
3. **Follow-up**: Continue conversation without wake word
4. **Confirm**: Say "yes" or "go ahead" when asked for permission
5. **End**: Say "thank you" or "goodbye" to end conversation

### Example Interaction

```
You: "Claudette"
Claudette: "Yes, sir?"

You: "How's my computer running?"
Claudette: "One moment, sir."
Claudette: "Well sir, your machine is in splendid form..."

You: "Can you check disk space?"
Claudette: "One moment, sir."
Claudette: "You've got plenty of room, sir..."

You: "Thank you"
Claudette: "My pleasure, sir."
```

## Project Structure

```
claudette/
├── src/
│   └── claudette/
│       ├── __init__.py
│       └── assistant.py      # Main assistant logic
├── tests/
│   └── test_assistant.py
├── config.yaml               # Configuration file
├── logo.png                  # Project logo
├── pyproject.toml           # Package configuration
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

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

### Areas for Contribution

- [ ] Additional wake word variants for different accents
- [ ] Support for more TTS providers
- [ ] Local Whisper integration (no server needed)
- [ ] GUI/system tray integration
- [ ] macOS and Windows support improvements
- [ ] Conversation history persistence
- [ ] Custom personality prompts
- [ ] Plugin system for extending capabilities

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
- Check the logs in `logs/` directory for what Whisper heard
- Add custom variants to the wake word list if needed

### Audio cutting off
- Adjust the `silence_duration` in config.yaml
- Check your microphone levels

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
- The 1940s for the aesthetic inspiration

---

<p align="center">
  <em>"Good day, sir. Claudette at your service."</em>
</p>
