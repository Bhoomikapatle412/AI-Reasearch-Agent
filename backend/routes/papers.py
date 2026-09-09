"""
FastAPI routes — /api/papers  (upload, list, delete)
"""
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import settings
from backend.paper_store import paper_store
from backend.pdf_processor import process_pdf
from backend.vector_store import vector_store

router = APIRouter(prefix="/api/papers", tags=["papers"])

MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    title: str = Form(default=""),
):
    """Upload a PDF and index it into the vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb} MB.",
        )

    # Save to disk
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    file_path = upload_dir / safe_name
    with open(file_path, "wb") as f:
        f.write(content)

    # Extract + embed
    paper_title = title.strip() or Path(file.filename).stem
    try:
        result = process_pdf(file_path, paper_title=paper_title)
    except Exception as exc:
        os.remove(file_path)
        raise HTTPException(status_code=422, detail=str(exc))

    # Store metadata
    paper_store.add(
        {
            "paper_id": result["paper_id"],
            "title": result["title"],
            "abstract": "",
            "authors": [],
            "year": None,
            "citations": 0,
            "url": None,
            "pdf_url": None,
            "topics": [],
            "source": "upload",
            "file_path": str(file_path),
            "indexed": True,
            "num_pages": result.get("num_pages", 0),
            "num_chunks": result.get("num_chunks", 0),
        }
    )

    return {
        "paper_id": result["paper_id"],
        "title": result["title"],
        "num_pages": result.get("num_pages", 0),
        "num_chunks": result.get("num_chunks", 0),
        "preview": result.get("preview", ""),
        "already_indexed": result.get("already_indexed", False),
        "message": "Paper uploaded and indexed successfully.",
    }


@router.get("")
async def list_papers():
    """Return all papers in the store (uploaded + searched)."""
    return {"papers": paper_store.get_all(), "total": paper_store.count()}


@router.get("/indexed")
async def list_indexed_papers():
    """Return only papers that have been indexed in the vector store."""
    ids = paper_store.get_indexed_ids()
    papers = [p for pid in ids if (p := paper_store.get(pid))]
    return {"papers": papers, "total": len(papers)}


@router.get("/{paper_id}")
async def get_paper(paper_id: str):
    paper = paper_store.get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@router.delete("/{paper_id}")
async def delete_paper(paper_id: str):
    paper = paper_store.get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    # Remove from vector store
    vector_store.delete_collection(paper_id)

    # Remove file if uploaded
    file_path = paper.get("file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    paper_store.delete(paper_id)
    return {"message": f"Paper {paper_id} deleted."}
