"""
app/services/llm_service.py
────────────────────────────
Unified LLM interface powered by LiteLLM.

Supports ANY provider with zero code changes — just set the right API key
and pass the LiteLLM model string in requests:

  Provider      | Example model string
  ─────────────────────────────────────────────────────
  Anthropic     | "claude-haiku-3-5"  or  "claude-3-5-sonnet-20241022"
  OpenAI        | "gpt-4o-mini"       or  "gpt-4o"
  HuggingFace   | "huggingface/mistralai/Mistral-7B-Instruct-v0.2"
  Gemini        | "gemini/gemini-pro"
  vLLM          | "openai/mistral"  (with VLLM_BASE_URL set)
  Ollama        | "ollama/llama3"

LiteLLM docs: https://docs.litellm.ai/docs/
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator

import litellm

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import Message, ModelProvider, TokenUsage

logger = get_logger(__name__)

# ── Configure LiteLLM ────────────────────────────────────────────────────────
# Inject all provider API keys so LiteLLM can route automatically.
# Keys are read from settings (which reads from .env).

def _configure_litellm() -> None:
    """Push all provider credentials into LiteLLM's environment."""
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

    if settings.hf_token:
        os.environ["HUGGINGFACE_API_KEY"] = settings.hf_token

    # If a vLLM base URL is set, configure it as an OpenAI-compatible custom endpoint
    if settings.vllm_base_url:
        os.environ["OPENAI_API_BASE"] = settings.vllm_base_url

    # Reduce verbose LiteLLM logging in production
    litellm.set_verbose = settings.app_env == "development"


_configure_litellm()

# ── Default system prompt ────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """You are a helpful, accurate, and thoughtful AI assistant.
When answering questions:
- Be concise yet thorough
- Cite sources when context is provided
- Acknowledge uncertainty rather than hallucinating
- Format code in markdown code blocks
"""

# ── Default models per provider ──────────────────────────────────────────────
_DEFAULT_MODEL: dict[ModelProvider, str] = {
    ModelProvider.openai:      settings.openai_default_model,
    ModelProvider.anthropic:   settings.anthropic_default_model,
    ModelProvider.huggingface: settings.hf_default_model,
}


def _resolve_model(provider: ModelProvider, model: str | None) -> str:
    """
    Return the final model string LiteLLM should use.

    LiteLLM expects:
      - OpenAI:    "gpt-4o-mini"                (no prefix needed)
      - Anthropic: "claude-haiku-3-5"           (no prefix needed)
      - HuggingFace: "huggingface/<model-id>"   (prefix required)
      - vLLM:      "openai/<model-id>"           (openai-compat prefix)

    If the user already passes a prefixed model string we pass it through
    unchanged; otherwise we resolve the default from settings.
    """
    chosen = model or _DEFAULT_MODEL.get(provider, "gpt-4o-mini")

    # Auto-prefix HuggingFace models if not already prefixed
    if provider == ModelProvider.huggingface and not chosen.startswith(("huggingface/", "openai/")):
        if settings.vllm_base_url:
            # Route through vLLM using the OpenAI-compat adapter
            chosen = f"openai/{chosen}"
        else:
            chosen = f"huggingface/{chosen}"

    if provider == ModelProvider.anthropic and not chosen.startswith("anthropic/"):
        chosen = f"anthropic/{chosen}"

    return chosen


def _build_messages(messages: list[Message], system_prompt: str) -> list[dict]:
    """Convert internal Message objects to LiteLLM/OpenAI message dicts."""
    result: list[dict] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        result.append({"role": m.role.value, "content": m.content})
    return result


# ── LiteLLM Service ──────────────────────────────────────────────────────────

class LLMService:
    """
    Single, provider-agnostic service backed by LiteLLM.
    Supports streaming and non-streaming for all providers.
    """

    async def chat(
        self,
        messages: list[Message],
        provider: ModelProvider = ModelProvider.anthropic,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, TokenUsage]:
        resolved = _resolve_model(provider, model)
        litellm_messages = _build_messages(messages, system_prompt)

        logger.info("litellm_chat", provider=provider, model=resolved, n_messages=len(litellm_messages))
        t0 = time.monotonic()

        resp = await litellm.acompletion(
            model=resolved,
            messages=litellm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = (time.monotonic() - t0) * 1000
        usage_data = resp.usage or {}
        usage = TokenUsage(
            prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_data, "total_tokens", 0) or 0,
        )
        content = resp.choices[0].message.content or ""
        logger.info("litellm_chat_done", model=resolved, latency_ms=round(elapsed), tokens=usage.total_tokens)
        return content, usage

    async def stream(
        self,
        messages: list[Message],
        provider: ModelProvider = ModelProvider.anthropic,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        resolved = _resolve_model(provider, model)
        litellm_messages = _build_messages(messages, system_prompt)

        logger.info("litellm_stream", provider=provider, model=resolved)

        response = await litellm.acompletion(
            model=resolved,
            messages=litellm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Singleton ────────────────────────────────────────────────────────────────
llm_service = LLMService()
