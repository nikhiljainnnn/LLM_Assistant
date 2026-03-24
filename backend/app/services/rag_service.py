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


# ── FAISS Vector Store ────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Wraps a FAISS IndexFlatIP (inner-product / cosine for normalised vectors).
    Keeps a parallel list of Chunk objects for metadata retrieval.
    """

    def __init__(self, dim: int, store_path: Path) -> None:
        self.dim = dim
        self.store_path = store_path
        self._chunks: list[Chunk] = []
        self._index = None
        self._load_or_create()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        import faiss

        index_file = self.store_path / "index.faiss"
        meta_file = self.store_path / "metadata.json"

        if index_file.exists() and meta_file.exists():
            logger.info("faiss_loading", path=str(self.store_path))
            self._index = faiss.read_index(str(index_file))
            with open(meta_file) as f:
                raw = json.load(f)
            self._chunks = [Chunk(**c) for c in raw]
            logger.info("faiss_loaded", vectors=self._index.ntotal)
        else:
            logger.info("faiss_creating_new_index", dim=self.dim)
            self._index = faiss.IndexFlatIP(self.dim)  # cosine (normalised)

    def save(self) -> None:
        import faiss

        self.store_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.store_path / "index.faiss"))
        with open(self.store_path / "metadata.json", "w") as f:
            json.dump([c.__dict__ for c in self._chunks], f, indent=2)
        logger.info("faiss_saved", vectors=self._index.ntotal)

    # ── Write ────────────────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their corresponding normalised embeddings."""
        if not chunks:
            return
        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)
        vectors = (vectors / norms).astype(np.float32)
        self._index.add(vectors)
        self._chunks.extend(chunks)
        self.save()

    # ── Read ─────────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, top_k: int) -> list[SourceChunk]:
        if self._index.ntotal == 0:
            return []

        # Normalise query
        norm = np.linalg.norm(query_vec)
        if norm > 1e-9:
            query_vec = query_vec / norm
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                SourceChunk(
                    text=chunk.text,
                    source=chunk.source,
                    score=float(score),
                    chunk_index=chunk.chunk_index,
                )
            )
        return results

    @property
    def total(self) -> int:
        return self._index.ntotal if self._index else 0


# ── RAG Service ───────────────────────────────────────────────────────────────

class RAGService:
    def __init__(self) -> None:
        self._store: FAISSVectorStore | None = None
        self._chunker = TextChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

    def _get_store(self) -> FAISSVectorStore:
        if self._store is None:
            self._store = FAISSVectorStore(
                dim=embedding_service.dim,
                store_path=settings.vector_store_path,
            )
        return self._store

    # ── Ingestion ────────────────────────────────────────────────────────────

    async def ingest(
        self,
        text: str,
        source_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk, embed, and store text. Returns number of chunks added."""
        chunks = self._chunker.chunk(text, source_name)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = await embedding_service.embed(texts)

        store = self._get_store()
        store.add(chunks, vectors)
        logger.info("ingested", source=source_name, chunks=len(chunks))
        return len(chunks)

    # ── Retrieval ────────────────────────────────────────────────────────────

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SourceChunk]:
        """Retrieve top-k relevant chunks for a query."""
        top_k = top_k or settings.top_k_retrieval
        store = self._get_store()
        if store.total == 0:
            return []

        q_vec = await embedding_service.embed_query(query)
        results = store.search(q_vec[0], top_k)
        logger.debug("retrieved", query=query[:50], results=len(results))
        return results

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
        return self._get_store().total


rag_service = RAGService()
