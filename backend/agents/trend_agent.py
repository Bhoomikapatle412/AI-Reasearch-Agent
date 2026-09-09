"""
Trend Analysis Agent — analyzes publication patterns, keyword frequencies,
and topic evolution across a set of papers.
"""
from __future__ import annotations
import logging
import re
from collections import Counter, defaultdict
from typing import Any

from backend.llm_client import get_llm, extract_text

logger = logging.getLogger(__name__)

TREND_PROMPT = """\
You are a research trend analyst. Based on the following collection of research papers, identify:
1. Emerging research topics and trends
2. Key themes and their evolution over time
3. Hot research areas that are rapidly growing

Paper collection:
{paper_list}

Keywords and topics found: {keywords}

Provide analysis in EXACTLY this JSON format (no markdown fences):
{{
  "emerging_topics": [
    {{"topic": "name", "description": "brief description", "growth": "high/medium/low", "representative_papers": ["id1"]}}
  ],
  "dominant_themes": [
    {{"theme": "name", "frequency": 5, "years_active": "2020-2024"}}
  ],
  "declining_topics": ["topic1", "topic2"],
  "timeline_insights": "How has the research focus shifted over the years?",
  "hottest_area": "The single most active research area right now",
  "recommendations": ["recommendation 1", "recommendation 2"]
}}
"""

# Common stop words to exclude from keyword analysis
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "is",
    "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "we", "our",
    "this", "that", "these", "those", "it", "its", "which", "who", "what",
    "how", "when", "where", "than", "as", "not", "no", "so", "if", "can",
    "using", "based", "paper", "proposed", "show", "shows", "shown", "also",
    "results", "result", "method", "methods", "approach", "system", "model",
    "data", "both", "each", "such", "used", "use", "new", "high", "low",
    "large", "small", "two", "three", "first", "second", "third",
}


class TrendAnalysisAgent:
    """Analyzes publication trends across a paper collection."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.3, max_tokens=1500)
        return self._llm

    async def analyze(self, papers: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Full trend analysis for a list of papers.

        Args:
            papers: list of {"paper_id", "title", "abstract", "year", "topics"}
        """
        if not papers:
            return {"error": "No papers provided for trend analysis."}

        # --- Statistical analysis (no LLM needed) ---
        stats = self._compute_stats(papers)

        # --- LLM-powered qualitative analysis ---
        paper_list_str = "\n".join(
            f"  [{p.get('year', '?')}] {p.get('title', 'Unknown')} | Topics: {', '.join(p.get('topics', []))}"
            for p in papers
        )
        keywords_str = ", ".join(
            f"{w}({c})" for w, c in stats["top_keywords"][:30]
        )
        prompt = TREND_PROMPT.format(
            paper_list=paper_list_str[:4000],
            keywords=keywords_str,
        )

        qualitative: dict = {}
        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            text = extract_text(response)
            qualitative = self._parse_trends(text)
        except Exception as exc:
            logger.warning("Trend LLM analysis failed: %s", exc)
            qualitative = {"error": str(exc)}

        return {**stats, **qualitative, "papers_analyzed": len(papers)}

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def _compute_stats(self, papers: list[dict]) -> dict[str, Any]:
        years: list[int] = []
        word_counter: Counter = Counter()
        topic_counter: Counter = Counter()
        yearly_papers: dict[int, int] = defaultdict(int)

        for paper in papers:
            year = paper.get("year")
            if year and isinstance(year, int) and 1990 <= year <= 2030:
                years.append(year)
                yearly_papers[year] += 1

            # Extract keywords from title + abstract
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            words = re.findall(r"\b[a-z]{4,}\b", text.lower())
            word_counter.update(w for w in words if w not in STOP_WORDS)

            # Topics from API metadata
            for topic in paper.get("topics", []):
                if topic:
                    topic_counter[topic.lower()] += 1

        # Publications per year (sorted)
        pub_by_year = [
            {"year": y, "count": yearly_papers[y]}
            for y in sorted(yearly_papers.keys())
        ]

        return {
            "total_papers": len(papers),
            "year_range": [min(years), max(years)] if years else [],
            "publications_by_year": pub_by_year,
            "top_keywords": word_counter.most_common(50),
            "top_topics": topic_counter.most_common(20),
            "avg_year": round(sum(years) / len(years), 1) if years else None,
        }

    def _parse_trends(self, text: str) -> dict[str, Any]:
        import json, re as _re

        json_match = _re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {
            "emerging_topics": [],
            "dominant_themes": [],
            "timeline_insights": text[:600],
            "hottest_area": "N/A",
            "recommendations": [],
        }


trend_agent = TrendAnalysisAgent()
