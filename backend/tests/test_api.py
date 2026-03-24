"""
tests/test_api.py
──────────────────
Integration-level tests using FastAPI's TestClient.
Run: pytest tests/ -v --cov=app
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers():
    return {"X-API-Key": settings.api_key}


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_missing_api_key_returns_401(self, client):
        resp = client.post("/api/v1/chat", json={"message": "hello"})
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"message": "hello"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


# ── RAG ───────────────────────────────────────────────────────────────────────

class TestRAG:
    def test_ingest_text(self, client, auth_headers):
        resp = client.post(
            "/api/v1/rag/ingest",
            json={
                "text": "Retrieval-Augmented Generation (RAG) combines a retrieval "
                        "system with a language model. FAISS is a library for efficient "
                        "similarity search of dense vectors.",
                "source_name": "test-doc",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_added"] >= 1
        assert data["source_name"] == "test-doc"

    def test_search_returns_results(self, client, auth_headers):
        resp = client.post(
            "/api/v1/rag/search",
            json={"query": "What is FAISS?", "top_k": 3},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)

    def test_stats_endpoint(self, client, auth_headers):
        resp = client.get("/api/v1/rag/stats", headers=auth_headers)
        assert resp.status_code == 200
        assert "total_vectors" in resp.json()

    def test_ingest_file_unsupported_type(self, client, auth_headers):
        resp = client.post(
            "/api/v1/rag/ingest/file",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
            headers=auth_headers,
        )
        assert resp.status_code == 415

    def test_ingest_txt_file(self, client, auth_headers):
        txt = b"This is a test document about machine learning and neural networks."
        resp = client.post(
            "/api/v1/rag/ingest/file",
            files={"file": ("test.txt", txt, "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_added"] >= 1


# ── Chat ──────────────────────────────────────────────────────────────────────

class TestChat:
    def test_chat_schema_validation(self, client, auth_headers):
        resp = client.post(
            "/api/v1/chat",
            json={"message": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_clear_conversation(self, client, auth_headers):
        conv_id = "test-conv-001"
        resp = client.delete(f"/api/v1/chat/{conv_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["conversation_id"] == conv_id


# ── Fine-tune ─────────────────────────────────────────────────────────────────

class TestFineTune:
    def test_list_jobs_empty(self, client, auth_headers):
        resp = client.get("/api/v1/finetune", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_nonexistent_job(self, client, auth_headers):
        resp = client.get("/api/v1/finetune/nonexistent-id", headers=auth_headers)
        assert resp.status_code == 404


# ── Unit Tests ────────────────────────────────────────────────────────────────

class TestUnits:
    def test_chunker_basic(self):
        from app.services.rag_service import TextChunker
        chunker = TextChunker(chunk_size=5, overlap=1)
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = chunker.chunk(text, "test")
        assert len(chunks) >= 2
        assert all(c.source == "test" for c in chunks)

    def test_memory_service_basic(self):
        from app.services.memory_service import MemoryService
        from app.models.schemas import RoleType
        mem = MemoryService(max_turns=2)
        mem.append_user("c1", "hello")
        mem.append_assistant("c1", "hi there")
        history = mem.get_history("c1")
        assert len(history) == 2
        assert history[0].role == RoleType.user

    def test_memory_sliding_window(self):
        from app.services.memory_service import MemoryService
        mem = MemoryService(max_turns=2)
        for i in range(5):
            mem.append_user("c1", f"msg {i}")
            mem.append_assistant("c1", f"resp {i}")
        history = mem.get_history("c1")
        assert len(history) == 4
