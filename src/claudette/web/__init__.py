"""
Claudette Web Dashboard

A web-based interface for managing Claudette settings, viewing status,
and browsing conversation history.
"""

from .server import WebServer
from .state import ClaudeActivity, ClaudetteStateManager

__all__ = ["WebServer", "ClaudetteStateManager", "ClaudeActivity"]
