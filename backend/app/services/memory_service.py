"""
app/services/memory_service.py
────────────────────────────────
In-process conversation memory with sliding window.
Each conversation_id maps to an ordered list of Messages.

For production, swap _store for Redis or a DB-backed store.
"""

from __future__ import annotations

from collections import defaultdict, deque

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import Message, RoleType

logger = get_logger(__name__)


class MemoryService:
    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        # conv_id -> deque of Messages (each "turn" = user + assistant pair)
        self._store: dict[str, deque[Message]] = defaultdict(
            lambda: deque(maxlen=max_turns * 2)
        )

    def add(self, conversation_id: str, message: Message) -> None:
        self._store[conversation_id].append(message)

    def get_history(self, conversation_id: str) -> list[Message]:
        return list(self._store[conversation_id])

    def append_user(self, conversation_id: str, content: str) -> None:
        self.add(conversation_id, Message(role=RoleType.user, content=content))

    def append_assistant(self, conversation_id: str, content: str) -> None:
        self.add(conversation_id, Message(role=RoleType.assistant, content=content))

    def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)
        logger.info("memory_cleared", conv_id=conversation_id)

    def list_conversations(self) -> list[str]:
        return list(self._store.keys())


memory_service = MemoryService(max_turns=settings.max_history_turns)
