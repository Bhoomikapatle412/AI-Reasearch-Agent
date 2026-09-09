"""
Paper Comparison Agent — generates structured side-by-side comparisons
of multiple research papers.
"""
from __future__ import annotations
import logging
from typing import Any

from backend.llm_client import get_llm, extract_text
from backend.vector_store import vector_store

logger = logging.getLogger(__name__)

COMPARISON_PROMPT = """\
You are an expert research analyst. Compare the following research papers in detail.

Papers to compare:
{paper_list}

Paper Contents:
{context}

Generate a detailed comparison in EXACTLY this JSON format (no markdown fences):
{{
  "overview": "1-2 sentence overview of what all these papers are about",
  "comparison_table": [
    {{
      "aspect": "Research Problem",
      "papers": {{
        "paper_id_1": "description",
        "paper_id_2": "description"
      }}
    }},
    {{
      "aspect": "Methodology",
      "papers": {{}}
    }},
    {{
      "aspect": "Dataset",
      "papers": {{}}
    }},
    {{
      "aspect": "Results",
      "papers": {{}}
    }},
    {{
      "aspect": "Strengths",
      "papers": {{}}
    }},
    {{
      "aspect": "Weaknesses",
      "papers": {{}}
    }},
    {{
      "aspect": "Limitations",
      "papers": {{}}
    }}
  ],
  "best_for": {{
    "paper_id_1": "This paper is best for..."
  }},
  "conclusion": "Which paper(s) make the most significant contribution and why?"
}}

Use the actual paper_ids as keys. Be specific and analytical.
"""


class ComparisonAgent:
    """Compares multiple papers using their stored vector-store content."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.2, max_tokens=2000)
        return self._llm

    async def compare(
        self, papers: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Compare a list of papers.

        Args:
            papers: list of {"paper_id": str, "title": str, "abstract": str}

        Returns structured comparison dict.
        """
        if len(papers) < 2:
            return {"error": "Please provide at least 2 papers to compare."}

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
                    f"methodology results dataset contribution {title}",
                    k=4,
                )
                texts = [c["text"] for c in chunks]
                content = " ".join(texts)[:1500]
            else:
                content = abstract[:800]

            context_parts.append(
                f"=== Paper [{pid}]: {title} ===\n{content}"
            )

        context = "\n\n".join(context_parts)
        prompt = COMPARISON_PROMPT.format(
            paper_list=paper_list_str, context=context[:7000]
        )

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            text = extract_text(response)
            return self._parse_comparison(text, papers)
        except Exception as exc:
            logger.error("Comparison failed: %s", exc)
            return {"error": str(exc), "papers": papers}

    def _parse_comparison(
        self, text: str, papers: list[dict]
    ) -> dict[str, Any]:
        import json, re

        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                data["papers"] = papers
                return data
            except json.JSONDecodeError:
                pass

        # Structured fallback
        return {
            "overview": text[:500],
            "comparison_table": [],
            "best_for": {},
            "conclusion": text[500:800],
            "papers": papers,
            "parse_error": True,
        }


comparison_agent = ComparisonAgent()
