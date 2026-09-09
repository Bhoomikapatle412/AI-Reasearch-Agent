"""
PDF processor — extracts text from uploaded PDFs, chunks it, and stores
chunks in the vector database.
"""
from __future__ import annotations
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.vector_store import vector_store

logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    """MD5 hash of file contents (used as stable paper_id)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text_from_pdf(path: Path) -> list[dict[str, Any]]:
    """
    Extract text page-by-page from a PDF using pdfplumber.
    Returns a list of {"page": int, "text": str} dicts.
    """
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # Normalise whitespace / ligatures
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    pages.append({"page": i, "text": text})
        return pages
    except Exception as exc:
        logger.error("pdfplumber extraction failed: %s — trying pypdf", exc)
        return _extract_with_pypdf(path)


def _extract_with_pypdf(path: Path) -> list[dict[str, Any]]:
    """Fallback extraction using pypdf."""
    from pypdf import PdfReader

    pages = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = None,
    overlap: int = None,
) -> list[dict[str, Any]]:
    """
    Sliding-window character-level chunking across pages.
    Each chunk carries page metadata for source attribution.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    # Build a flat list of (char_position, page_number, text)
    flat_text = ""
    char_to_page: list[int] = []
    for p in pages:
        start = len(flat_text)
        flat_text += p["text"] + " "
        char_to_page.extend([p["page"]] * (len(flat_text) - start))

    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(flat_text):
        end = min(start + chunk_size, len(flat_text))
        chunk_text = flat_text[start:end].strip()
        if chunk_text:
            page_num = char_to_page[start] if start < len(char_to_page) else pages[-1]["page"]
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "chunk_index": idx,
                        "page": page_num,
                        "start_char": start,
                        "end_char": end,
                    },
                }
            )
            idx += 1
        start += chunk_size - overlap

    return chunks


def process_pdf(
    file_path: Path,
    paper_title: str = "",
    paper_id: str | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: extract → chunk → embed → store.

    Returns a dict with paper_id, title, num_pages, num_chunks, and
    a short text preview (first 500 chars of the extracted content).
    """
    paper_id = paper_id or _hash_file(file_path)

    # Skip re-processing if already in vector store
    if vector_store.collection_exists(paper_id):
        logger.info("Paper %s already indexed, skipping.", paper_id)
        return {
            "paper_id": paper_id,
            "title": paper_title,
            "already_indexed": True,
        }

    pages = extract_text_from_pdf(file_path)
    if not pages:
        raise ValueError("Could not extract text from PDF. Is it a scanned image?")

    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("PDF chunking produced no text chunks.")

    # Attach paper-level metadata to each chunk
    for chunk in chunks:
        chunk["metadata"]["paper_id"] = paper_id
        chunk["metadata"]["paper_title"] = paper_title or file_path.stem

    num_chunks = vector_store.add_chunks(paper_id, chunks)

    preview = pages[0]["text"][:500] if pages else ""

    return {
        "paper_id": paper_id,
        "title": paper_title or file_path.stem,
        "num_pages": len(pages),
        "num_chunks": num_chunks,
        "preview": preview,
        "already_indexed": False,
    }
