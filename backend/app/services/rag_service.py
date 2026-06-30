"""
app/services/rag_service.py
────────────────────────────
Retrieval-Augmented Generation pipeline:
  1. Ingest documents → chunk → embed → store in FAISS
  2. At query time → embed query → retrieve top-k chunks
  3. Augment LLM prompt with retrieved context

Persistence: index is saved/loaded from disk so it survives restarts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import SourceChunk
from app.services.embedding_service import embedding_service

logger = get_logger(__name__)


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Text chunking ────────────────────────────────────────────────────────────

class TextChunker:
    """Sentence-aware sliding-window chunker."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> list[Chunk]:
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        words = text.split()
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, source=source, chunk_index=idx))
            idx += 1
            if end == len(words):
                break
            start = end - self.overlap  # sliding overlap

        return chunks


# ── DB Vector Store ─────────────────────────────────────────────────────────────

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.core.db import AsyncSessionLocal
from app.models.domain import DocumentChunk

class RAGService:
    def __init__(self) -> None:
        self._chunker = TextChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

    # ── Ingestion ────────────────────────────────────────────────────────────

    async def ingest(
        self,
        text: str,
        source_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk, embed, and store text in pgvector. Returns number of chunks added."""
        chunks = self._chunker.chunk(text, source_name)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = await embedding_service.embed(texts)
        
        async with AsyncSessionLocal() as session:
            for i, chunk in enumerate(chunks):
                # Ensure the vector is normalized if using cosine distance
                v = vectors[i]
                norm = np.linalg.norm(v)
                if norm > 1e-9:
                    v = v / norm
                
                doc = DocumentChunk(
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    embedding=v.tolist()
                )
                session.add(doc)
            await session.commit()
            
        logger.info("ingested", source=source_name, chunks=len(chunks))
        return len(chunks)

    # ── Retrieval ────────────────────────────────────────────────────────────

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SourceChunk]:
        """Retrieve top-k relevant chunks using pgvector cosine distance."""
        top_k = top_k or settings.top_k_retrieval

        q_vec_raw = await embedding_service.embed_query(query)
        q_vec = q_vec_raw[0]
        norm = np.linalg.norm(q_vec)
        if norm > 1e-9:
            q_vec = q_vec / norm

        async with AsyncSessionLocal() as session:
            # <==> is the pgvector operator for cosine distance
            results = await session.execute(
                select(DocumentChunk, DocumentChunk.embedding.cosine_distance(q_vec.tolist()).label("distance"))
                .order_by("distance")
                .limit(top_k)
            )
            
            chunks = []
            for doc, distance in results:
                # distance is 0 to 2 (0 being identical).
                # Convert back to a similarity score (1.0 to -1.0)
                score = 1.0 - distance
                chunks.append(
                    SourceChunk(
                        text=doc.text,
                        source=doc.source,
                        score=float(score),
                        chunk_index=doc.chunk_index,
                    )
                )
            
            logger.debug("retrieved", query=query[:50], results=len(chunks))
            return chunks

    # ── Prompt Augmentation ──────────────────────────────────────────────────

    @staticmethod
    def build_augmented_prompt(query: str, chunks: list[SourceChunk]) -> str:
        """Prepend retrieved context to the user's question."""
        if not chunks:
            return query

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.source}]\n{chunk.text}"
            )

        context_block = "\n\n---\n\n".join(context_parts)
        return (
            f"Use the following context to answer the question.\n"
            f"If the context doesn't contain the answer, say so.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {query}"
        )

    @property
    def vector_count(self) -> int:
        # A sync property isn't great for async DB calls. 
        # We will mock it here, or the endpoint calling this needs an async method.
        # Let's provide an async count method for the stats endpoint.
        return 0
        
    async def get_vector_count(self) -> int:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.count(DocumentChunk.id)))
            return result.scalar()


rag_service = RAGService()


