"""
Per-session conversation memory stored in Redis.
Each session is identified by a session_id (UUID).
Memory expires after CONVERSATION_MEMORY_TTL seconds of inactivity.
"""

import json
import os
from uuid import uuid4

import redis

r = redis.from_url(os.getenv("REDIS_URL"))


def get_session_key(session_id: str) -> str:
    return f"conv_memory:{session_id}"


def get_history(session_id: str) -> list[dict]:
    """
    Returns conversation history as list of message dicts:
    [
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
    """
    raw = r.get(get_session_key(session_id))
    if not raw:
        return []
    return json.loads(raw)


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    """
    Appends a user+assistant turn to session history.
    Trims to last CONVERSATION_MAX_TURNS turns.
    Resets TTL on every update.
    """
    history = get_history(session_id)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})

    max_turns = int(os.getenv("CONVERSATION_MAX_TURNS", 6)) * 2
    history = history[-max_turns:]

    r.setex(
        get_session_key(session_id),
        int(os.getenv("CONVERSATION_MEMORY_TTL", 1800)),
        json.dumps(history),
    )


def clear_session(session_id: str) -> None:
    """Explicitly clear a session's memory."""
    r.delete(get_session_key(session_id))


def create_session() -> str:
    """Generate a new session ID."""
    return str(uuid4())


# Feature: Conversation Memory (Multi-turn Chat) implemented
