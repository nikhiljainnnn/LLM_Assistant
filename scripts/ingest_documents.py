#!/usr/bin/env python3
"""
scripts/ingest_documents.py
────────────────────────────
Bulk-ingest a folder of documents into the FAISS vector store
by calling the running API server.

Usage:
    python scripts/ingest_documents.py \
        --dir ./docs \
        --api_url http://localhost:8000 \
        --api_key dev-api-key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx


SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk document ingestion")
    p.add_argument("--dir", required=True, help="Directory containing documents")
    p.add_argument("--api_url", default="http://localhost:8000")
    p.add_argument("--api_key", default="dev-api-key")
    p.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    return p.parse_args()


def collect_files(directory: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED]
    else:
        files = [f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED]
    return sorted(files)


def ingest_file(client: httpx.Client, filepath: Path, headers: dict) -> dict:
    with open(filepath, "rb") as f:
        mime = "application/octet-stream"
        if filepath.suffix == ".pdf":
            mime = "application/pdf"
        elif filepath.suffix == ".txt":
            mime = "text/plain"
        elif filepath.suffix == ".md":
            mime = "text/markdown"

        resp = client.post(
            "/api/v1/rag/ingest/file",
            files={"file": (filepath.name, f, mime)},
            data={"source_name": filepath.name},
            headers=headers,
            timeout=60.0,
        )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    args = parse_args()
    doc_dir = Path(args.dir)

    if not doc_dir.exists():
        print(f"❌ Directory not found: {doc_dir}", file=sys.stderr)
        sys.exit(1)

    files = collect_files(doc_dir, args.recursive)
    if not files:
        print(f"No supported documents found in {doc_dir}")
        print(f"Supported types: {', '.join(SUPPORTED)}")
        sys.exit(0)

    print(f"Found {len(files)} documents to ingest\n")

    headers = {"X-API-Key": args.api_key}
    total_chunks = 0
    errors = []

    with httpx.Client(base_url=args.api_url) as client:
        for i, filepath in enumerate(files, 1):
            try:
                result = ingest_file(client, filepath, headers)
                chunks = result.get("chunks_added", 0)
                total_chunks += chunks
                print(f"  [{i:>3}/{len(files)}] ✓  {filepath.name}  ({chunks} chunks)")
            except Exception as exc:
                errors.append((filepath.name, str(exc)))
                print(f"  [{i:>3}/{len(files)}] ✗  {filepath.name}  — {exc}")

    print(f"\n{'='*50}")
    print(f"  Ingested : {len(files) - len(errors)}/{len(files)} files")
    print(f"  Chunks   : {total_chunks:,} total vectors added")
    if errors:
        print(f"  Errors   : {len(errors)}")
        for name, err in errors:
            print(f"    • {name}: {err}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
