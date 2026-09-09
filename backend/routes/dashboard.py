"""
FastAPI routes — /api/dashboard
"""
from fastapi import APIRouter

from backend.paper_store import paper_store
from backend.vector_store import vector_store

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("")
async def get_dashboard():
    """Return aggregated dashboard statistics."""
    all_papers = paper_store.get_all()
    indexed_ids = paper_store.get_indexed_ids()

    # Publication year distribution
    year_counter: dict[int, int] = {}
    for paper in all_papers:
        year = paper.get("year")
        if year and isinstance(year, int):
            year_counter[year] = year_counter.get(year, 0) + 1

    pub_trend = [
        {"year": y, "count": c}
        for y, c in sorted(year_counter.items())
    ]

    # Top topics
    topic_counter: dict[str, int] = {}
    for paper in all_papers:
        for t in paper.get("topics", []):
            if t:
                topic_counter[t] = topic_counter.get(t, 0) + 1
    top_topics = sorted(topic_counter.items(), key=lambda x: -x[1])[:10]

    # Most cited papers
    sorted_by_citations = sorted(
        all_papers, key=lambda p: p.get("citations", 0), reverse=True
    )[:5]

    # Recent papers (sorted by year)
    recent = sorted(
        [p for p in all_papers if p.get("year")],
        key=lambda p: p["year"],
        reverse=True,
    )[:5]

    return {
        "total_papers": len(all_papers),
        "indexed_papers": len(indexed_ids),
        "topics": paper_store.get_topics(),
        "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
        "publications_by_year": pub_trend,
        "most_cited": sorted_by_citations,
        "recent_papers": recent,
    }
