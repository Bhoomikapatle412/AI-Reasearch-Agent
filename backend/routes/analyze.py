"""
FastAPI routes — /api/analyze  (summarize, compare, gaps, trends)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.agents.analysis_agent import analysis_agent
from backend.agents.comparison_agent import comparison_agent
from backend.agents.gap_agent import gap_agent
from backend.agents.trend_agent import trend_agent
from backend.agents.rag_agent import rag_agent
from backend.paper_store import paper_store

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    paper_id: str
    title: Optional[str] = ""
    abstract: Optional[str] = ""


class QARequest(BaseModel):
    question: str
    paper_ids: list[str]
    k: int = 6


class CompareRequest(BaseModel):
    paper_ids: list[str]


class GapRequest(BaseModel):
    paper_ids: list[str]


class TrendRequest(BaseModel):
    paper_ids: list[str]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/summarize")
async def summarize_paper(req: SummarizeRequest):
    """Generate a structured summary of a single paper."""
    paper = paper_store.get(req.paper_id)
    title = req.title or (paper.get("title") if paper else "") or ""
    abstract = req.abstract or (paper.get("abstract") if paper else "") or ""

    result = await analysis_agent.summarize(req.paper_id, title=title, abstract=abstract)
    return {"paper_id": req.paper_id, "title": title, "summary": result}


@router.post("/ask")
async def ask_question(req: QARequest):
    """Answer a question using RAG over the specified papers."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Verify at least one paper exists
    valid_ids = [pid for pid in req.paper_ids if paper_store.get(pid)]
    if not valid_ids:
        raise HTTPException(status_code=404, detail="None of the specified papers were found.")

    result = await rag_agent.answer(req.question.strip(), valid_ids, k=req.k)
    return result


@router.post("/compare")
async def compare_papers(req: CompareRequest):
    """Generate a structured comparison of multiple papers."""
    if len(req.paper_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 paper IDs required.")

    papers = []
    for pid in req.paper_ids:
        paper = paper_store.get(pid)
        if paper:
            papers.append({
                "paper_id": pid,
                "title": paper.get("title", "Unknown"),
                "abstract": paper.get("abstract", ""),
            })

    if len(papers) < 2:
        raise HTTPException(status_code=404, detail="Could not find enough papers.")

    result = await comparison_agent.compare(papers)
    return result


@router.post("/gaps")
async def detect_gaps(req: GapRequest):
    """Identify research gaps across multiple papers."""
    if not req.paper_ids:
        raise HTTPException(status_code=400, detail="At least 1 paper ID required.")

    papers = []
    for pid in req.paper_ids:
        paper = paper_store.get(pid)
        if paper:
            papers.append({
                "paper_id": pid,
                "title": paper.get("title", "Unknown"),
                "abstract": paper.get("abstract", ""),
            })

    if not papers:
        raise HTTPException(status_code=404, detail="No papers found.")

    result = await gap_agent.analyze_gaps(papers)
    return result


@router.post("/trends")
async def analyze_trends(req: TrendRequest):
    """Analyze research trends across a set of papers."""
    if not req.paper_ids:
        # If no IDs provided, analyze all stored papers
        all_papers = paper_store.get_all()
        papers = all_papers
    else:
        papers = [
            paper_store.get(pid)
            for pid in req.paper_ids
            if paper_store.get(pid)
        ]

    if not papers:
        raise HTTPException(status_code=404, detail="No papers found for trend analysis.")

    result = await trend_agent.analyze(papers)
    return result
