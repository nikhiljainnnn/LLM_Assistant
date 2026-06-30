"""
app/api/rag.py
───────────────
POST /api/v1/rag/ingest         — ingest raw text
POST /api/v1/rag/ingest/file    — upload a PDF / DOCX / TXT file
POST /api/v1/rag/search         — semantic search (without generation)
GET  /api/v1/rag/stats          — vector store statistics
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.security import get_current_user
from app.models.domain import User
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.rag_service import rag_service
from app.utils.document_parser import extract_text

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(
    req: IngestRequest,
    current_user: User = Depends(get_current_user),
):
    chunks_added = await rag_service.ingest(
        text=req.text,
        source_name=req.source_name,
        metadata=req.metadata,
    )
    return IngestResponse(
        source_name=req.source_name,
        chunks_added=chunks_added,
        total_vectors=rag_service.vector_count,
    )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source_name: str = Form(default=""),
    current_user: User = Depends(get_current_user),
):
    allowed = {".pdf", ".docx", ".txt", ".md"}
    suffix = "." + file.filename.split(".")[-1].lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {allowed}",
        )

    content = await file.read()
    try:
        text = extract_text(content, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Parse error: {exc}") from exc

    if not text.strip():
        raise HTTPException(status_code=422, detail="File appears to be empty")

    src = source_name or file.filename
    chunks_added = await rag_service.ingest(text=text, source_name=src)
    return IngestResponse(
        source_name=src,
        chunks_added=chunks_added,
        total_vectors=rag_service.vector_count,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    current_user: User = Depends(get_current_user),
):
    results = await rag_service.retrieve(req.query, top_k=req.top_k)
    return SearchResponse(query=req.query, results=results)


@router.get("/stats")
async def stats(current_user: User = Depends(get_current_user)):
    return {
        "total_vectors": rag_service.vector_count,
        "embedding_dim": rag_service._get_store().dim,
        "store_path": str(rag_service._get_store().store_path),
    }
