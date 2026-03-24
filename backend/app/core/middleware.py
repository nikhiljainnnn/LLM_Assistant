"""Request middleware and shared exception handlers."""

from __future__ import annotations

import uuid

import structlog
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _error_payload(
    *,
    code: str,
    message: str,
    request_id: str | None,
    details: object | None = None,
) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": details,
        }
    }


async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)

    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()

    response.headers["X-Request-ID"] = request_id
    return response


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(
            code="http_error",
            message=str(exc.detail),
            request_id=request_id,
        ),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            code="validation_error",
            message="Request validation failed",
            request_id=request_id,
            details=exc.errors(),
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            code="internal_error",
            message="An internal error occurred",
            request_id=request_id,
        ),
    )
