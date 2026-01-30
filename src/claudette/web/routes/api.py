"""
REST API endpoints for Claudette web dashboard.
"""

import logging
import platform
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger("claudette.web")


# Pydantic models for request/response validation


class StatusResponse(BaseModel):
    """Response model for status endpoint."""

    state: str
    conversation_mode: bool
    awaiting_confirmation: bool
    audio_level: float
    last_transcription: str
    last_response: str
    uptime_seconds: float
    timestamp: str


class ConfigResponse(BaseModel):
    """Response model for config endpoint."""

    whisper: dict
    vad: dict
    tts: dict
    wake_word: dict
    memory: dict
    sounds: dict
    hotkey: dict


class ConfigUpdateRequest(BaseModel):
    """Request model for config updates."""

    updates: dict = Field(..., description="Dictionary of config keys and values to update")


class SkillInfo(BaseModel):
    """Model for skill information."""

    name: str
    description: str
    triggers: list[str]
    category: str = "builtin"
    icon: str = "K"
    example: str = ""


class MCPToolInfo(BaseModel):
    """Model for MCP tool information."""

    name: str
    description: str
    category: str
    icon: str
    examples: list[str]


class SkillsResponse(BaseModel):
    """Response model for skills endpoint."""

    skills: list[SkillInfo]
    mcp_tools: list[MCPToolInfo]


class HistoryEntry(BaseModel):
    """Model for conversation history entry."""

    timestamp: str
    user: str
    assistant: str


class HistoryResponse(BaseModel):
    """Response model for history endpoint."""

    entries: list[HistoryEntry]
    total: int
    limit: int
    offset: int


class VoiceInfo(BaseModel):
    """Model for voice information."""

    id: str
    name: str
    language: str
    description: str


class PersonalityInfo(BaseModel):
    """Model for personality information."""

    name: str
    description: str


class SystemInfo(BaseModel):
    """Response model for system info endpoint."""

    platform: str
    platform_version: str
    python_version: str
    cpu_percent: float | None
    memory_percent: float | None
    memory_total_gb: float | None
    battery_percent: float | None
    battery_plugged: bool | None
    gpu_name: str | None
    gpu_available: bool


# Available voices (Edge TTS)
AVAILABLE_VOICES = [
    VoiceInfo(
        id="en-GB-SoniaNeural",
        name="Sonia",
        language="British English",
        description="Professional, clear (default)",
    ),
    VoiceInfo(
        id="en-GB-LibbyNeural", name="Libby", language="British English", description="Warm, friendly"
    ),
    VoiceInfo(
        id="en-GB-MaisieNeural",
        name="Maisie",
        language="British English",
        description="Young, energetic",
    ),
    VoiceInfo(
        id="en-GB-RyanNeural", name="Ryan", language="British English", description="Male voice"
    ),
    VoiceInfo(
        id="en-US-AriaNeural",
        name="Aria",
        language="American English",
        description="Clear, professional",
    ),
    VoiceInfo(
        id="en-US-JennyNeural",
        name="Jenny",
        language="American English",
        description="Warm, conversational",
    ),
    VoiceInfo(
        id="en-AU-NatashaNeural",
        name="Natasha",
        language="Australian English",
        description="Australian accent",
    ),
]


def create_api_router(state_manager: Any) -> APIRouter:
    """Create API router with all endpoints.

    Args:
        state_manager: ClaudetteStateManager instance

    Returns:
        Configured APIRouter
    """
    router = APIRouter(tags=["API"])

    @router.get("/status", response_model=StatusResponse)
    async def get_status():
        """Get current Claudette status."""
        snapshot = state_manager.get_snapshot()
        return StatusResponse(
            state=snapshot.state,
            conversation_mode=snapshot.conversation_mode,
            awaiting_confirmation=snapshot.awaiting_confirmation,
            audio_level=snapshot.audio_level,
            last_transcription=snapshot.last_transcription,
            last_response=snapshot.last_response,
            uptime_seconds=snapshot.uptime_seconds,
            timestamp=snapshot.timestamp,
        )

    @router.get("/config", response_model=ConfigResponse)
    async def get_config():
        """Get current configuration."""
        config = state_manager.get_config()
        if not config:
            raise HTTPException(status_code=503, detail="Claudette not initialized")
        return config

    @router.patch("/config")
    async def update_config(request: ConfigUpdateRequest):
        """Update configuration at runtime.

        Supports updates like:
        - {"updates": {"tts.voice": "en-GB-LibbyNeural"}}
        - {"updates": {"vad.threshold": 0.6}}
        """
        if not request.updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        updated_config = state_manager.update_config(request.updates)
        if not updated_config:
            raise HTTPException(status_code=503, detail="Claudette not initialized")

        return {"success": True, "config": updated_config}

    @router.post("/config/save")
    async def save_config():
        """Save current configuration to YAML file.

        Note: This endpoint would persist changes to config.yaml.
        Currently returns a placeholder response.
        """
        # TODO: Implement config file saving
        return {
            "success": False,
            "message": "Config file saving not yet implemented. Changes are runtime-only.",
        }

    @router.get("/skills", response_model=SkillsResponse)
    async def get_skills():
        """Get list of available skills and MCP tools."""
        raw_skills = state_manager.get_skills()

        # Skill category and icon mappings
        skill_metadata = {
            "time": ("info", "T", "What time is it?"),
            "date": ("info", "D", "What's the date today?"),
            "status": ("system", "S", "What's your status?"),
            "system_info": ("system", "I", "Tell me about my computer"),
            "battery": ("system", "B", "What's my battery level?"),
            "volume": ("media", "V", "Volume up / Turn it down"),
            "screenshot": ("media", "C", "Take a screenshot"),
            "lock_screen": ("system", "L", "Lock the screen"),
            "clear_memory": ("memory", "M", "Clear memory / Start fresh"),
            "voice_change": ("settings", "O", "Change voice to Libby"),
            "personality": ("settings", "P", "Change personality to butler"),
            "wake_word": ("settings", "W", "Add wake word variant"),
            "list_skills": ("help", "?", "What can you do?"),
            "greeting": ("social", "H", "Good morning, Claudette"),
            "joke": ("social", "J", "Tell me a joke"),
        }

        skills = []
        for s in raw_skills:
            meta = skill_metadata.get(s["name"], ("builtin", "K", ""))
            skills.append(SkillInfo(
                name=s["name"],
                description=s["description"],
                triggers=s["triggers"],
                category=meta[0],
                icon=meta[1],
                example=meta[2]
            ))

        # MCP Tools
        mcp_tools = [
            MCPToolInfo(
                name="Unraid Server",
                description="Access your Unraid NAS server - check Docker containers, browse files, execute commands, and monitor system status",
                category="infrastructure",
                icon="U",
                examples=[
                    "Check my downloads",
                    "What containers are running?",
                    "How much disk space do I have?",
                    "Show me the Plex logs"
                ]
            ),
            MCPToolInfo(
                name="News Feed",
                description="Get the latest news headlines from BBC News across various categories",
                category="information",
                icon="N",
                examples=[
                    "What's in the news?",
                    "Give me technology news",
                    "Any business headlines?",
                    "What's happening in the world?"
                ]
            ),
            MCPToolInfo(
                name="Web Search",
                description="Search the web using DuckDuckGo for current information",
                category="information",
                icon="S",
                examples=[
                    "Search for Python FastAPI tutorial",
                    "Look up the weather in London",
                    "Find information about...",
                    "Search the web for..."
                ]
            ),
        ]

        return SkillsResponse(skills=skills, mcp_tools=mcp_tools)

    @router.get("/history", response_model=HistoryResponse)
    async def get_history(limit: int = 50, offset: int = 0):
        """Get conversation history with pagination.

        Args:
            limit: Maximum number of entries to return (default: 50)
            offset: Number of entries to skip from the end (default: 0)
        """
        entries = state_manager.get_history(limit=limit, offset=offset)

        # Get total count
        claudette = state_manager.claudette
        total = 0
        if claudette and claudette.memory:
            total = len(claudette.memory.exchanges)

        return HistoryResponse(
            entries=[
                HistoryEntry(timestamp=e["timestamp"], user=e["user"], assistant=e["assistant"])
                for e in entries
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    @router.post("/history/clear")
    async def clear_history():
        """Clear conversation history."""
        state_manager.clear_history()
        return {"success": True, "message": "Conversation history cleared"}

    @router.get("/claude/activity")
    async def get_claude_activity():
        """Get current Claude CLI activity status."""
        activity = state_manager.get_claude_activity()
        return {
            "active": activity.active,
            "query": activity.query,
            "status": activity.status,
            "current_output": activity.current_output,
            "progress_lines": activity.progress_lines,
            "started_at": activity.started_at,
            "elapsed_seconds": activity.elapsed_seconds,
        }

    @router.get("/voices", response_model=list[VoiceInfo])
    async def get_voices():
        """Get available TTS voices."""
        return AVAILABLE_VOICES

    @router.get("/personalities", response_model=list[PersonalityInfo])
    async def get_personalities():
        """Get available personality presets."""
        from ...personalities import list_personalities

        personalities = list_personalities()
        return [PersonalityInfo(name=name, description=desc) for name, desc in personalities.items()]

    @router.get("/system", response_model=SystemInfo)
    async def get_system_info():
        """Get system information."""
        import sys

        info = SystemInfo(
            platform=platform.system(),
            platform_version=platform.release(),
            python_version=sys.version.split()[0],
            cpu_percent=None,
            memory_percent=None,
            memory_total_gb=None,
            battery_percent=None,
            battery_plugged=None,
            gpu_name=None,
            gpu_available=False,
        )

        # Try to get psutil info
        try:
            import psutil

            info.cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            info.memory_percent = mem.percent
            info.memory_total_gb = round(mem.total / (1024**3), 2)

            battery = psutil.sensors_battery()
            if battery:
                info.battery_percent = battery.percent
                info.battery_plugged = battery.power_plugged
        except ImportError:
            pass

        # Try to get GPU info
        try:
            import torch

            info.gpu_available = torch.cuda.is_available()
            if info.gpu_available:
                info.gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        return info

    @router.get("/logs")
    async def get_logs(limit: int = 50):
        """Get recent log entries.

        Args:
            limit: Maximum number of entries to return (default: 50)
        """
        logs = state_manager.get_recent_logs(limit=limit)
        return {
            "entries": [
                {"timestamp": log.timestamp, "level": log.level, "message": log.message}
                for log in logs
            ]
        }

    @router.get("/voice/test")
    async def test_voice(voice: str = "en-GB-SoniaNeural", rate: str = "+0%", text: str = "Yes, sir? How may I assist you today?"):
        """Generate test audio for a voice.

        Args:
            voice: Voice ID (e.g., en-GB-SoniaNeural)
            rate: Speech rate (e.g., +0%, -25%, +50%)
            text: Text to speak (default: greeting)

        Returns:
            MP3 audio data
        """
        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, voice, rate=rate)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            return Response(content=audio_data, media_type="audio/mpeg")

        except Exception as e:
            logger.error(f"Voice test error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
