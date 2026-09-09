"""
Paper Search Agent — queries Semantic Scholar and arXiv APIs.

Both APIs are free and require no authentication for basic usage.
Semantic Scholar provides richer metadata (citations, topics).
arXiv provides full-text preprints.
"""
from __future__ import annotations
import asyncio
import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "https://export.arxiv.org/api/query"

# Fields to request from Semantic Scholar
SS_FIELDS = (
    "paperId,title,abstract,year,citationCount,"
    "authors,externalIds,url,fieldsOfStudy,tldr,"
    "publicationTypes,openAccessPdf"
)


class PaperSearchAgent:
    """Searches Semantic Scholar and arXiv and merges the results."""

    def __init__(self):
        headers = {"User-Agent": "AIResearchAgent/1.0"}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key
        self._client = httpx.AsyncClient(
            headers=headers, timeout=20.0, follow_redirects=True
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def search(
        self, query: str, limit: int = 10, source: str = "both"
    ) -> dict[str, Any]:
        """
        Search papers from Semantic Scholar, arXiv, or both.

        Args:
            query:  Free-text research query
            limit:  Max results (split between sources when source="both")
            source: "semantic_scholar" | "arxiv" | "both"
        Returns:
            {"papers": [...], "total": int, "query": str, "sources": [...]}
        """
        tasks = []
        if source in ("semantic_scholar", "both"):
            tasks.append(self._search_semantic_scholar(query, limit))
        if source in ("arxiv", "both"):
            tasks.append(self._search_arxiv(query, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        papers: list[dict] = []
        sources_used: list[str] = []

        for result in results:
            if isinstance(result, Exception):
                logger.warning("Search source error: %s", result)
                continue
            papers.extend(result["papers"])
            sources_used.extend(result["sources"])

        # Deduplicate by title similarity (simple lowercase match)
        seen_titles: set[str] = set()
        unique_papers: list[dict] = []
        for paper in papers:
            key = re.sub(r"\s+", " ", paper.get("title", "")).strip().lower()
            if key not in seen_titles:
                seen_titles.add(key)
                unique_papers.append(paper)

        # Sort by citation count descending
        unique_papers.sort(key=lambda p: p.get("citations", 0), reverse=True)

        return {
            "papers": unique_papers[:limit],
            "total": len(unique_papers),
            "query": query,
            "sources": list(set(sources_used)),
        }

    async def get_paper_details(self, paper_id: str) -> dict[str, Any] | None:
        """Fetch detailed metadata for a Semantic Scholar paper ID."""
        url = f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}"
        try:
            resp = await self._client.get(url, params={"fields": SS_FIELDS})
            resp.raise_for_status()
            return self._normalize_ss_paper(resp.json())
        except Exception as exc:
            logger.error("get_paper_details failed: %s", exc)
            return None

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Semantic Scholar
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _search_semantic_scholar(
        self, query: str, limit: int
    ) -> dict[str, Any]:
        url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": SS_FIELDS,
        }
        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            papers = [
                self._normalize_ss_paper(p) for p in data.get("data", [])
            ]
            return {"papers": papers, "sources": ["semantic_scholar"]}
        except httpx.HTTPStatusError as exc:
            logger.warning("Semantic Scholar HTTP %s: %s", exc.response.status_code, exc)
            return {"papers": [], "sources": []}

    def _normalize_ss_paper(self, p: dict) -> dict:
        authors = [a.get("name", "") for a in p.get("authors", [])]
        pdf_url = None
        if p.get("openAccessPdf"):
            pdf_url = p["openAccessPdf"].get("url")
        arxiv_id = (p.get("externalIds") or {}).get("ArXiv")
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        tldr = p.get("tldr")
        return {
            "id": p.get("paperId", ""),
            "title": p.get("title", "Untitled"),
            "abstract": p.get("abstract", ""),
            "authors": authors,
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "url": p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}",
            "pdf_url": pdf_url,
            "topics": p.get("fieldsOfStudy") or [],
            "tldr": tldr.get("text") if tldr else None,
            "source": "semantic_scholar",
            "arxiv_id": arxiv_id,
        }

    # ------------------------------------------------------------------
    # arXiv
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _search_arxiv(self, query: str, limit: int) -> dict[str, Any]:
        params = {
            "search_query": f"all:{quote_plus(query)}",
            "start": 0,
            "max_results": min(limit, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            resp = await self._client.get(ARXIV_BASE, params=params)
            resp.raise_for_status()
            papers = self._parse_arxiv_response(resp.text)
            return {"papers": papers, "sources": ["arxiv"]}
        except Exception as exc:
            logger.warning("arXiv search failed: %s", exc)
            return {"papers": [], "sources": []}

    def _parse_arxiv_response(self, xml_text: str) -> list[dict]:
        """Parse arXiv Atom XML response."""
        import xml.etree.ElementTree as ET

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        papers = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id_raw = (entry.findtext("atom:id", "", ns) or "").split("/abs/")[-1]
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            abstract = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            published = entry.findtext("atom:published", "", ns) or ""
            year = int(published[:4]) if published else None
            authors = [
                a.findtext("atom:name", "", ns)
                for a in entry.findall("atom:author", ns)
            ]
            # Extract categories as topics
            topics = [
                c.get("term", "") for c in entry.findall("atom:category", ns)
            ]
            papers.append(
                {
                    "id": f"arxiv:{arxiv_id_raw}",
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "citations": 0,
                    "url": f"https://arxiv.org/abs/{arxiv_id_raw}",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id_raw}",
                    "topics": topics,
                    "tldr": None,
                    "source": "arxiv",
                    "arxiv_id": arxiv_id_raw,
                }
            )
        return papers


# Module-level singleton used by the FastAPI router
search_agent = PaperSearchAgent()
