"""
app/services/memory_service.py
────────────────────────────────
Conversation memory with sliding window stored in Redis.
Each conversation_id maps to an ordered list of Messages.
"""

from __future__ import annotations

import json

from redis.asyncio import Redis, from_url

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import Message, RoleType

logger = get_logger(__name__)


class MemoryService:
    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self.max_len = max_turns * 2
        self.redis: Redis = from_url(settings.redis_url, decode_responses=True)

    def _key(self, conversation_id: str) -> str:
        return f"memory:{conversation_id}"

    async def add(self, conversation_id: str, message: Message) -> None:
        key = self._key(conversation_id)
        # Push to right end of list
        await self.redis.rpush(key, message.model_dump_json())
        # Trim to max length (sliding window)
        await self.redis.ltrim(key, -self.max_len, -1)

    async def get_history(self, conversation_id: str) -> list[Message]:
        key = self._key(conversation_id)
        raw_items = await self.redis.lrange(key, 0, -1)
        return [Message.model_validate_json(item) for item in raw_items]

    async def append_user(self, conversation_id: str, content: str) -> None:
        await self.add(conversation_id, Message(role=RoleType.user, content=content))

    async def append_assistant(self, conversation_id: str, content: str) -> None:
        await self.add(conversation_id, Message(role=RoleType.assistant, content=content))

    async def clear(self, conversation_id: str) -> None:
        await self.redis.delete(self._key(conversation_id))
        logger.info("memory_cleared", conv_id=conversation_id)

    async def list_conversations(self) -> list[str]:
        keys = await self.redis.keys("memory:*")
        return [k.replace("memory:", "") for k in keys]


memory_service = MemoryService(max_turns=settings.max_history_turns)
