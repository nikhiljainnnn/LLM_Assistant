"""
app/api/chat.py
────────────────
POST /api/v1/chat          — standard chat (with optional RAG)
POST /api/v1/chat/stream   — SSE streaming chat
DELETE /api/v1/chat/{id}   — clear conversation memory
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.security import get_current_user
from app.models.domain import User
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Message,
    RoleType,
)
from app.services.llm_service import DEFAULT_SYSTEM_PROMPT, llm_service
from app.services.memory_service import memory_service
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _build_llm_input(req: ChatRequest):
    """Return (augmented_user_msg, sources, history)."""
    sources = []

    if req.use_rag:
        chunks = await rag_service.retrieve(req.message, top_k=5)
        if chunks:
            augmented = rag_service.build_augmented_prompt(req.message, chunks)
            sources = chunks
        else:
            augmented = req.message
    else:
        augmented = req.message

    history = await memory_service.get_history(req.conversation_id)
    messages = history + [Message(role=RoleType.user, content=augmented)]
    return messages, sources


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    t0 = time.monotonic()
    messages, sources = await _build_llm_input(req)

    content, usage = await llm_service.chat(
        messages=messages,
        provider=req.provider,
        model=req.model,
        system_prompt=req.system_prompt or DEFAULT_SYSTEM_PROMPT,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    # Persist to memory
    await memory_service.append_user(req.conversation_id, req.message)
    await memory_service.append_assistant(req.conversation_id, content)

    latency_ms = (time.monotonic() - t0) * 1000

    return ChatResponse(
        conversation_id=req.conversation_id,
        message=Message(role=RoleType.assistant, content=content),
        sources=sources,
        usage=usage,
        latency_ms=round(latency_ms, 1),
    )


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """Server-Sent Events streaming endpoint."""
    messages, _sources = await _build_llm_input(req)

    async def event_generator():
        collected = []
        try:
            async for token in llm_service.stream(
                messages=messages,
                provider=req.provider,
                model=req.model,
                system_prompt=req.system_prompt or DEFAULT_SYSTEM_PROMPT,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                collected.append(token)
                # SSE format: "data: <payload>\n\n"
                yield f"data: {token}\n\n"
        except Exception:
            yield (
                "data: "
                + json.dumps({"error": "Streaming failed. Please retry."})
                + "\n\n"
            )
        finally:
            full_response = "".join(collected)
            await memory_service.append_user(req.conversation_id, req.message)
            if full_response:
                await memory_service.append_assistant(req.conversation_id, full_response)
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    await memory_service.clear(conversation_id)
    return {"message": "Conversation cleared", "conversation_id": conversation_id}
