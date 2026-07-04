"""
tests/test_api.py
──────────────────
Integration-level tests using FastAPI's TestClient (sync) and
pytest-anyio for async unit tests.

Run: pytest tests/ -v --cov=app

Design notes
────────────
• The `client` fixture overrides the SQLAlchemy DB dependency with an
  in-memory async SQLite engine so tests never need a live PostgreSQL.
• Redis calls in MemoryService are mocked via `unittest.mock`.
• JWT tokens are obtained by calling the real /auth/register → /auth/token
  routes inside the test session, mirroring the production flow.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.db import Base, get_db
from app.main import app

# ── In-memory async SQLite engine ─────────────────────────────────────────────
# aiosqlite is the async driver for SQLite; it ships with most Python installs.
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

_test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
_TestSessionLocal = async_sessionmaker(
    bind=_test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def _override_get_db():
    async with _TestSessionLocal() as session:
        yield session


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Provide a TestClient whose DB is a fresh in-memory SQLite instance and
    whose lifespan DB-ping is mocked out (SQLite doesn't need pgvector).
    """
    # Override DB dependency
    app.dependency_overrides[get_db] = _override_get_db

    # Patch the lifespan DB ping so the server starts without a real Postgres
    import app.main as main_module
    original_lifespan = main_module.lifespan

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _mock_lifespan(a):
        # Create all SQLite tables (no pgvector, so skip Vector columns)
        async with _test_engine.begin() as conn:
            # Only create tables that SQLite supports (no pgvector Vector type)
            from app.models.domain import User
            from sqlalchemy import Table
            await conn.run_sync(
                lambda sync_conn: User.__table__.create(sync_conn, checkfirst=True)
            )
        yield
        await _test_engine.dispose()

    main_module.app.router.lifespan_context = _mock_lifespan

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def auth_headers(client):
    """
    Register a fresh test user, log in, and return JWT Bearer headers.
    This mirrors the real production auth flow.
    """
    reg = client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "email": "testuser@example.com",
            "password": "Str0ng!Pass",
        },
    )
    assert reg.status_code == 200, f"Register failed: {reg.text}"

    token_resp = client.post(
        "/api/v1/auth/token",
        data={"username": "testuser", "password": "Str0ng!Pass"},
    )
    assert token_resp.status_code == 200, f"Login failed: {token_resp.text}"
    token = token_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200


# ── Auth endpoints ────────────────────────────────────────────────────────────

class TestAuth:
    def test_register_creates_user(self, client):
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "S3cur3!pw",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "newuser"
        assert "id" in data

    def test_register_duplicate_username_returns_400(self, client):
        payload = {
            "username": "dupeuser",
            "email": "dupe1@example.com",
            "password": "S3cur3!pw",
        }
        client.post("/api/v1/auth/register", json=payload)
        resp = client.post(
            "/api/v1/auth/register",
            json={**payload, "email": "dupe2@example.com"},
        )
        assert resp.status_code == 400

    def test_login_returns_jwt_token(self, client):
        # Register then login
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "logintest",
                "email": "logintest@example.com",
                "password": "P@ssword1",
            },
        )
        resp = client.post(
            "/api/v1/auth/token",
            data={"username": "logintest", "password": "P@ssword1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password_returns_401(self, client):
        resp = client.post(
            "/api/v1/auth/token",
            data={"username": "testuser", "password": "wrongpassword"},
        )
        assert resp.status_code == 401

    def test_protected_route_without_token_returns_401(self, client):
        resp = client.post("/api/v1/chat", json={"message": "hello"})
        assert resp.status_code == 401

    def test_protected_route_with_invalid_token_returns_401(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"message": "hello"},
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401


# ── Chat ──────────────────────────────────────────────────────────────────────

class TestChat:
    def test_chat_schema_validation_empty_message(self, client, auth_headers):
        """Empty message string should fail Pydantic validation → 422."""
        resp = client.post(
            "/api/v1/chat",
            json={"message": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_clear_conversation(self, client, auth_headers):
        conv_id = "test-conv-001"
        with patch(
            "app.services.memory_service.memory_service.clear",
            new_callable=AsyncMock,
        ):
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
        """TextChunker is a pure-sync class — no DB or Redis needed."""
        from app.services.rag_service import TextChunker
        chunker = TextChunker(chunk_size=5, overlap=1)
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = chunker.chunk(text, "test")
        assert len(chunks) >= 2
        assert all(c.source == "test" for c in chunks)

    def test_memory_service_append_and_get(self):
        """
        MemoryService now uses Redis. We mock the Redis client so the test
        runs in-process without a live Redis instance.
        """
        import asyncio
        from app.services.memory_service import MemoryService
        from app.models.schemas import RoleType

        async def _run():
            mem = MemoryService(max_turns=2)

            # Mock the underlying Redis instance
            mock_redis = AsyncMock()
            # lrange returns a list of JSON strings
            from app.models.schemas import Message
            user_msg = Message(role=RoleType.user, content="hello")
            asst_msg = Message(role=RoleType.assistant, content="hi there")
            mock_redis.lrange.return_value = [
                user_msg.model_dump_json(),
                asst_msg.model_dump_json(),
            ]
            mem.redis = mock_redis

            history = await mem.get_history("c1")
            assert len(history) == 2
            assert history[0].role == RoleType.user
            assert history[1].role == RoleType.assistant

        asyncio.run(_run())

    def test_memory_service_clear_calls_redis_delete(self):
        """Verify that clear() issues a Redis DELETE command."""
        import asyncio
        from app.services.memory_service import MemoryService

        async def _run():
            mem = MemoryService(max_turns=2)
            mock_redis = AsyncMock()
            mem.redis = mock_redis

            await mem.clear("conv-x")
            mock_redis.delete.assert_called_once_with("memory:conv-x")

        asyncio.run(_run())

    def test_jwt_create_and_decode(self):
        """Validate that create_access_token produces a decodable JWT."""
        import jwt as pyjwt
        from app.core.security import create_access_token
        from app.core.config import settings

        token = create_access_token({"sub": "alice"})
        payload = pyjwt.decode(token, settings.app_secret_key, algorithms=["HS256"])
        assert payload["sub"] == "alice"
        assert "exp" in payload
