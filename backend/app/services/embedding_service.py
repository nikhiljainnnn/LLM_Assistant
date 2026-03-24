"""
app/services/embedding_service.py
───────────────────────────────────
Generates embeddings using:
  • OpenAI text-embedding-3-small  (default, cloud)
  • sentence-transformers           (local fallback)

Returns numpy float32 arrays suitable for FAISS.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Dimension map for OpenAI models
_OAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast
_LOCAL_DIM = 384


class EmbeddingService:
    def __init__(self) -> None:
        self._st_model = None
        self._openai_client = None

    # ── Public API ──────────────────────────────────────────────────────────

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Returns shape (N, D) float32 array.
        Uses OpenAI if configured, else local sentence-transformers.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if settings.use_openai and settings.openai_api_key:
            return await self._openai_embed(list(texts))
        return await asyncio.get_event_loop().run_in_executor(
            None, self._local_embed, list(texts)
        )

    async def embed_query(self, query: str) -> np.ndarray:
        """Returns shape (1, D) float32."""
        return await self.embed([query])

    @property
    def dim(self) -> int:
        if settings.use_openai and settings.openai_api_key:
            return _OAI_DIMS.get(settings.openai_embedding_model, 1536)
        return _LOCAL_DIM

    # ── OpenAI ──────────────────────────────────────────────────────────────

    async def _openai_embed(self, texts: list[str]) -> np.ndarray:
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        resp = await self._openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        vectors = [d.embedding for d in resp.data]
        return np.array(vectors, dtype=np.float32)

    # ── Local sentence-transformers ─────────────────────────────────────────

    def _local_embed(self, texts: list[str]) -> np.ndarray:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading_local_embedder", model=_LOCAL_MODEL)
            self._st_model = SentenceTransformer(
                _LOCAL_MODEL,
                device=settings.embedding_device,
            )
        vecs = self._st_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


embedding_service = EmbeddingService()
