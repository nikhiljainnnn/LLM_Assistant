"""
app/main.py
────────────
FastAPI application factory.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api import chat, finetune, rag
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.core.middleware import (
    http_exception_handler,
    request_id_middleware,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.models.schemas import HealthResponse

setup_logging()
logger = get_logger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "startup",
        env=settings.app_env,
        openai=settings.use_openai,
        vector_store=str(settings.vector_store_path),
    )
    yield
    logger.info("shutdown")


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
    app.include_router(chat.router)
    app.include_router(rag.router)
    app.include_router(finetune.router)

    # Health
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        from app.services.rag_service import rag_service
        return HealthResponse(
            status="ok",
            version="1.0.0",
            providers={
                "openai": settings.use_openai,
                "huggingface": True,
            },
            vector_store_size=rag_service.vector_count,
        )

    return app


app = create_app()
