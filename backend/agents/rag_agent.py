"""
RAG Research Agent — answers user questions using uploaded papers.

Architecture:
  1. Embed the question
  2. Retrieve top-k chunks from specified paper collections
  3. Build a context-aware prompt
  4. Call LLM and return answer with source attribution
"""
from __future__ import annotations
import logging
from typing import Any

from backend.llm_client import get_llm, extract_text
from backend.vector_store import vector_store

logger = logging.getLogger(__name__)

QA_PROMPT = """\
You are an expert research assistant. Answer the user's question using ONLY the provided paper excerpts below.
If the answer is not found in the excerpts, say "I couldn't find a direct answer in the provided papers."

Question: {question}

Paper Excerpts:
{context}

Instructions:
- Cite the source paper and page number where relevant (e.g., [Paper: "Title", Page 3])
- Be precise and factual
- If multiple papers address the question, synthesize their findings
- Do not make up information not present in the excerpts

Answer:"""


class RAGResearchAgent:
    """Answers questions grounded in the uploaded paper vector store."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.1, max_tokens=1200)
        return self._llm

    async def answer(
        self,
        question: str,
        paper_ids: list[str],
        k: int = 6,
    ) -> dict[str, Any]:
        """
        Answer a question using RAG over the specified papers.

        Returns:
            {
              "answer": str,
              "sources": [{"paper_id", "page", "text_snippet"}],
              "context_used": int  (number of chunks used)
            }
        """
        if not paper_ids:
            return {
                "answer": "No papers selected. Please upload or select papers first.",
                "sources": [],
                "context_used": 0,
            }

        chunks = vector_store.similarity_search(paper_ids, question, k=k)
        if not chunks:
            return {
                "answer": "No relevant content found in the selected papers for this question.",
                "sources": [],
                "context_used": 0,
            }

        context_parts: list[str] = []
        sources: list[dict] = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            title = meta.get("paper_title", chunk.get("paper_id", "Unknown"))
            page = meta.get("page", "?")
            snippet = chunk["text"][:200]
            context_parts.append(
                f'[Paper: "{title}", Page {page}]\n{chunk["text"]}'
            )
            sources.append(
                {
                    "paper_id": chunk["paper_id"],
                    "paper_title": title,
                    "page": page,
                    "snippet": snippet,
                }
            )

        context = "\n\n---\n\n".join(context_parts)
        prompt = QA_PROMPT.format(question=question, context=context[:5000])

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            answer_text = extract_text(response)
            return {
                "answer": answer_text.strip(),
                "sources": sources,
                "context_used": len(chunks),
            }
        except Exception as exc:
            logger.error("RAG answer failed: %s", exc)
            return {
                "answer": f"Error generating answer: {exc}",
                "sources": sources,
                "context_used": len(chunks),
                "error": str(exc),
            }


rag_agent = RAGResearchAgent()
