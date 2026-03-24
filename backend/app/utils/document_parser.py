"""
app/utils/document_parser.py
──────────────────────────────
Extracts plain text from:
  • .pdf    — pypdf
  • .docx   — python-docx
  • .txt / .md — raw read
"""

from __future__ import annotations

import io
from pathlib import Path


def extract_text(content: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(content)
    if suffix == ".docx":
        return _parse_docx(content)
    if suffix in {".txt", ".md", ".markdown"}:
        return content.decode("utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p for p in pages if p.strip())


def _parse_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
