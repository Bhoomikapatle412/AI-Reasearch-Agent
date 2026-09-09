"""
Research Gap Agent — analyzes multiple papers to identify:
  - Common limitations
  - Contradictions between papers
  - Under-explored areas
  - Potential future research directions
"""
from __future__ import annotations
import logging
from typing import Any

from backend.llm_client import get_llm, extract_text
from backend.vector_store import vector_store

logger = logging.getLogger(__name__)

GAP_PROMPT = """\
You are an expert research analyst specialized in identifying research gaps and future research opportunities.

Analyze the following research papers and identify gaps, limitations, contradictions, and research opportunities.

Papers analyzed:
{paper_list}

Paper Contents:
{context}

Provide your analysis in EXACTLY this JSON format (no markdown fences):
{{
  "common_limitations": [
    {{"limitation": "description", "papers_affected": ["paper_id_1", "paper_id_2"]}}
  ],
  "research_gaps": [
    {{"gap": "description", "explanation": "why this is a gap", "potential_impact": "high/medium/low"}}
  ],
  "contradictions": [
    {{"topic": "what contradicts", "paper_a": "paper_id_1", "claim_a": "claim", "paper_b": "paper_id_2", "claim_b": "claim"}}
  ],
  "unexplored_areas": [
    "area 1",
    "area 2"
  ],
  "future_directions": [
    {{"direction": "research direction", "rationale": "why this is important", "difficulty": "easy/medium/hard"}}
  ],
  "summary": "2-3 sentence summary of the main gaps found"
}}

Be analytical, specific, and grounded in the provided paper content.
"""


class ResearchGapAgent:
    """Identifies research gaps by analyzing multiple papers."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.3, max_tokens=2000)
        return self._llm

    async def analyze_gaps(
        self, papers: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze research gaps across multiple papers.

        Args:
            papers: list of {"paper_id", "title", "abstract"}
        """
        if not papers:
            return {"error": "No papers provided for gap analysis."}

        paper_list_str = "\n".join(
            f"  - [{p['paper_id']}] {p.get('title', 'Unknown')}"
            for p in papers
        )

        context_parts: list[str] = []
        for paper in papers:
            pid = paper["paper_id"]
            title = paper.get("title", "Unknown")
            abstract = paper.get("abstract", "")

            if vector_store.collection_exists(pid):
                chunks = vector_store.similarity_search(
                    [pid],
                    "limitations future work challenges open problems",
                    k=4,
                )
                content = " ".join(c["text"] for c in chunks)[:1500]
            else:
                content = abstract[:800]

            context_parts.append(
                f"=== [{pid}] {title} ===\n{content}"
            )

        context = "\n\n".join(context_parts)
        prompt = GAP_PROMPT.format(
            paper_list=paper_list_str, context=context[:7000]
        )

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            text = extract_text(response)
            return self._parse_gaps(text, papers)
        except Exception as exc:
            logger.error("Gap analysis failed: %s", exc)
            return {"error": str(exc), "papers": papers}

    def _parse_gaps(self, text: str, papers: list[dict]) -> dict[str, Any]:
        import json, re

        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                data["papers_analyzed"] = papers
                return data
            except json.JSONDecodeError:
                pass

        return {
            "common_limitations": [],
            "research_gaps": [],
            "contradictions": [],
            "unexplored_areas": [],
            "future_directions": [],
            "summary": text[:600],
            "papers_analyzed": papers,
            "parse_error": True,
        }


gap_agent = ResearchGapAgent()
