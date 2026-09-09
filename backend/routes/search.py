"""
FastAPI routes — /api/search
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from backend.agents.search_agent import search_agent
from backend.paper_store import paper_store

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchResponse(BaseModel):
    papers: list[dict]
    total: int
    query: str
    sources: list[str]


@router.get("", response_model=SearchResponse)
async def search_papers(
    q: str = Query(..., min_length=1, description="Research query"),
    limit: int = Query(default=10, ge=1, le=50),
    source: str = Query(default="both", pattern="^(both|semantic_scholar|arxiv)$"),
):
    """Search academic papers from Semantic Scholar and/or arXiv."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = await search_agent.search(q.strip(), limit=limit, source=source)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}")

    # Persist search results to paper store for later use
    for paper in result["papers"]:
        try:
            paper_store.add(paper)
        except Exception:
            pass

    return result
