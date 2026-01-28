# Contributing to Claudette

First off, thank you for considering contributing to Claudette! It's people like you that make Claudette such a delightful assistant.

## Code of Conduct

By participating in this project, you agree to maintain a welcoming and respectful environment for everyone.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (config snippets, log output)
- **Describe the behavior you observed and what you expected**
- **Include logs** from the `logs/` directory
- **Include your environment** (OS, Python version, audio device)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed feature**
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Install dev dependencies**: `pip install -e ".[dev]"`
3. **Make your changes**
4. **Add tests** if applicable
5. **Run the test suite**: `pytest`
6. **Format your code**: `black src/`
7. **Lint your code**: `ruff check src/`
8. **Commit your changes** with a clear message
9. **Push to your fork** and submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/claudette.git
cd claudette

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

## Project Structure

```
claudette/
├── src/claudette/
│   ├── __init__.py      # Package exports
│   └── assistant.py     # Main Claudette class
├── tests/
│   └── test_*.py        # Test files
├── config.yaml          # User configuration
└── pyproject.toml       # Package configuration
```

## Style Guide

- **Python**: Follow PEP 8, enforced by `black` and `ruff`
- **Line length**: 100 characters max
- **Docstrings**: Use Google style docstrings
- **Type hints**: Encouraged for public APIs
- **Commits**: Use clear, descriptive commit messages

### Code Example

```python
def process_audio(self, audio: np.ndarray) -> str | None:
    """Process recorded audio and return transcription.

    Args:
        audio: NumPy array of audio samples (float32, 16kHz mono)

    Returns:
        Transcribed text or None if transcription failed

    Raises:
        ConnectionError: If Whisper API is unreachable
    """
    # Implementation here
    pass
```

## Areas We'd Love Help With

- **Wake word detection**: Better variants for different accents
- **TTS providers**: Support for additional text-to-speech services
- **Local Whisper**: Integration with local Whisper models
- **Platform support**: Testing and fixes for macOS/Windows
- **Documentation**: Tutorials, examples, and guides
- **Translations**: Internationalization support
- **Testing**: More comprehensive test coverage

## Questions?

Feel free to open an issue with the "question" label if you have any questions about contributing.

---

*"Every contribution, no matter how small, is appreciated, sir."* — Claudette
