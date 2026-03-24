"""
app/services/llm_service.py
────────────────────────────
Unified LLM interface supporting:
  • OpenAI Chat Completions (streaming + non-streaming)
  • HuggingFace Transformers inference (local)

Usage:
    from app.services.llm_service import llm_service
    response = await llm_service.chat(messages, provider="openai")
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import Message, ModelProvider, RoleType, TokenUsage

logger = get_logger(__name__)

# ── Default system prompt ────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """You are a helpful, accurate, and thoughtful AI assistant.
When answering questions:
- Be concise yet thorough
- Cite sources when context is provided
- Acknowledge uncertainty rather than hallucinating
- Format code in markdown code blocks
"""


# ── OpenAI Backend ───────────────────────────────────────────────────────────

class OpenAIBackend:
    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    def _build_messages(
        self,
        messages: list[Message],
        system_prompt: str,
    ) -> list[dict]:
        result = [{"role": "system", "content": system_prompt}]
        for m in messages:
            result.append({"role": m.role.value, "content": m.content})
        return result

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, TokenUsage]:
        model = model or settings.openai_default_model
        oai_messages = self._build_messages(messages, system_prompt)

        logger.info("openai_chat", model=model, messages=len(oai_messages))
        t0 = time.monotonic()

        resp = await self.client.chat.completions.create(
            model=model,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = (time.monotonic() - t0) * 1000
        usage = TokenUsage(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
        )
        content = resp.choices[0].message.content or ""
        logger.info("openai_chat_done", latency_ms=round(elapsed), tokens=usage.total_tokens)
        return content, usage

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        model = model or settings.openai_default_model
        oai_messages = self._build_messages(messages, system_prompt)

        async with await self.client.chat.completions.create(
            model=model,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ── HuggingFace Backend ──────────────────────────────────────────────────────

class HuggingFaceBackend:
    """
    Runs a HuggingFace causal-LM locally.
    Lazy-loaded on first call to avoid startup overhead.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_id: str | None = None

    def _load(self, model_id: str) -> None:
        if self._model_id == model_id and self._pipeline is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        logger.info("hf_model_loading", model=model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=settings.hf_token or None,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=settings.hf_token or None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        self._model_id = model_id
        logger.info("hf_model_ready", model=model_id)

    def _format_prompt(self, messages: list[Message], system_prompt: str) -> str:
        """Simple Mistral-style instruction format."""
        parts = [f"<s>[INST] {system_prompt}\n\n"]
        for m in messages:
            if m.role == RoleType.user:
                parts.append(f"{m.content} [/INST]")
            else:
                parts.append(f"{m.content} </s><s>[INST] ")
        return "".join(parts)

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, TokenUsage]:
        import asyncio

        model_id = model or settings.hf_default_model
        prompt = self._format_prompt(messages, system_prompt)

        # Run blocking inference in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_inference, model_id, prompt, temperature, max_tokens
        )
        generated = result[0]["generated_text"][len(prompt):]
        # Approximate token count
        prompt_tokens = len(prompt.split())
        completion_tokens = len(generated.split())
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return generated, usage

    def _run_inference(
        self, model_id: str, prompt: str, temperature: float, max_tokens: int
    ) -> Any:
        self._load(model_id)
        return self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
        )

    async def stream(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        """HF streaming via TextIteratorStreamer — simplified version."""
        content, _ = await self.chat(*args, **kwargs)
        # Yield word-by-word to simulate streaming
        for word in content.split(" "):
            yield word + " "


# ── Unified Service ──────────────────────────────────────────────────────────

class LLMService:
    def __init__(self) -> None:
        self._openai = OpenAIBackend()
        self._hf = HuggingFaceBackend()

    def _backend(self, provider: ModelProvider):
        if provider == ModelProvider.openai:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not configured")
            return self._openai
        return self._hf

    async def chat(
        self,
        messages: list[Message],
        provider: ModelProvider = ModelProvider.openai,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, TokenUsage]:
        backend = self._backend(provider)
        return await backend.chat(
            messages=messages,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def stream(
        self,
        messages: list[Message],
        provider: ModelProvider = ModelProvider.openai,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        backend = self._backend(provider)
        async for token in backend.stream(
            messages=messages,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token


# Singleton
llm_service = LLMService()
