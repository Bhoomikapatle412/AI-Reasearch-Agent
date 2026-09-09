"""
Paper Analysis Agent — generates structured summaries of research papers.

For each paper it produces:
  - Problem addressed
  - Methodology
  - Dataset used
  - Results / findings
  - Limitations
  - Key contribution / novelty
"""
from __future__ import annotations
import logging
from typing import Any

from backend.llm_client import get_llm, extract_text
from backend.vector_store import vector_store

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """\
You are an expert research assistant. Analyze the following research paper text and provide a comprehensive structured summary.

Paper Title: {title}

Paper Text (excerpts):
{context}

Provide a detailed analysis in EXACTLY this JSON format (no markdown fences, just pure JSON):
{{
  "problem": "What problem or research question does this paper address?",
  "methodology": "What methods, algorithms, or approaches are used?",
  "dataset": "What datasets, benchmarks, or experimental setup is used?",
  "results": "What are the main results, findings, or performance metrics?",
  "limitations": "What are the stated or implied limitations?",
  "contribution": "What is the key novelty or contribution of this paper?",
  "summary": "A 2-3 sentence executive summary of the paper."
}}

Be specific and concise. Use information only from the provided text.
"""


class PaperAnalysisAgent:
    """Summarizes papers using RAG-retrieved context + LLM."""

    def __init__(self):
        self._llm = None  # lazy

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.2, max_tokens=1500)
        return self._llm

    async def summarize(
        self, paper_id: str, title: str = "", abstract: str = ""
    ) -> dict[str, Any]:
        """
        Generate a structured summary for a paper.

        Uses vector-store context if the paper has been indexed,
        otherwise falls back to the abstract.
        """
        context = self._get_context(paper_id, title, abstract)

        prompt = SUMMARY_PROMPT.format(title=title or "Unknown", context=context)

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            # Handle both str response (WatsonxLLM) and AIMessage (ChatOpenAI)
            text = extract_text(response)
            return self._parse_summary(text)
        except Exception as exc:
            logger.error("summarize failed: %s", exc)
            return self._fallback_summary(abstract, exc)

    def _get_context(
        self, paper_id: str, title: str, abstract: str, max_chars: int = 6000
    ) -> str:
        """Retrieve relevant chunks from vector store or fall back to abstract."""
        if vector_store.collection_exists(paper_id):
            query = f"methodology results dataset findings contribution {title}"
            chunks = vector_store.similarity_search([paper_id], query, k=8)
            if chunks:
                parts = [c["text"] for c in chunks]
                combined = "\n\n---\n\n".join(parts)
                return combined[:max_chars]
        return abstract or "No text available."

    def _parse_summary(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, with a robust fallback."""
        import json, re

        # Try to extract JSON block
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                # Ensure all keys are present
                for key in ("problem", "methodology", "dataset", "results",
                            "limitations", "contribution", "summary"):
                    data.setdefault(key, "Not available")
                return data
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text as summary
        return {
            "problem": "See full summary",
            "methodology": "See full summary",
            "dataset": "See full summary",
            "results": "See full summary",
            "limitations": "See full summary",
            "contribution": "See full summary",
            "summary": text[:800],
        }

    def _fallback_summary(self, abstract: str, error: Exception) -> dict[str, Any]:
        logger.warning("Using fallback summary due to: %s", error)
        return {
            "problem": "Could not analyze — LLM unavailable",
            "methodology": abstract[:300] if abstract else "N/A",
            "dataset": "N/A",
            "results": "N/A",
            "limitations": "N/A",
            "contribution": "N/A",
            "summary": abstract[:500] if abstract else "No abstract available.",
            "error": str(error),
        }


analysis_agent = PaperAnalysisAgent()
