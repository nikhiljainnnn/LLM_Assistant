"""
app/main.py
────────────
FastAPI application factory.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api import auth, chat, finetune, rag
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.core.middleware import (
    http_exception_handler,
    request_id_middleware,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.models.schemas import HealthResponse
from prometheus_fastapi_instrumentator import Instrumentator
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration

setup_logging()
logger = get_logger(__name__)

# ── Sentry ────────────────────────────────────────────────────────────────────
# Only activates when SENTRY_DSN is set — safe to leave unset in development.
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
            RedisIntegration(),
        ],
        # Capture 20% of transactions for performance monitoring in production.
        traces_sample_rate=0.2 if settings.is_production else 0.0,
        environment=settings.app_env,
        send_default_pii=False,   # Do NOT send PII (emails, usernames) to Sentry
    )
    logger.info("sentry.initialized", environment=settings.app_env)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.db import engine

    # ── Startup ───────────────────────────────────────────────
    logger.info(
        "startup",
        env=settings.app_env,
        openai=settings.use_openai,
        vector_store=str(settings.vector_store_path),
    )

    # Verify DB connectivity and warm up the connection pool.
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("database.connected", url=settings.database_url.split("@")[-1])
    except Exception as exc:
        logger.error("database.connection_failed", error=str(exc))
        # Re-raise so the server refuses to start with a broken DB
        raise

    yield

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("shutdown")
    await engine.dispose()
    logger.info("database.pool_disposed")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM-Powered AI Assistant",
        description="RAG + LoRA fine-tuning API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.middleware("http")(request_id_middleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)

    # Routers
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(rag.router)
    app.include_router(finetune.router)

    # Health
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        from app.services.rag_service import rag_service
        count = await rag_service.get_vector_count()
        return HealthResponse(
            status="ok",
            version="1.0.0",
            providers={
                "openai":    settings.use_openai,
                "anthropic": settings.use_anthropic,
                "huggingface": True,
            },
            vector_store_size=count,
        )

    # Instrument Prometheus
    Instrumentator().instrument(app).expose(app)

    return app


app = create_app()
