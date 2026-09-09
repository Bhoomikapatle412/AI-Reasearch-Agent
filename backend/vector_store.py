"""
Vector store manager — ChromaDB-backed document store for uploaded papers.
Each paper gets its own collection keyed by paper_id.
Provides CRUD operations and similarity search used by the RAG agent.
"""
from __future__ import annotations
import logging
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.llm_client import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Singleton wrapper around a persistent ChromaDB instance."""

    def __init__(self):
        self._chroma = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embeddings = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Embedding helper (lazy, loaded once)
    # ------------------------------------------------------------------

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        emb = self._get_embeddings()
        return emb.embed_documents(texts)

    def _embed_query(self, text: str) -> list[float]:
        emb = self._get_embeddings()
        return emb.embed_query(text)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _collection_name(self, paper_id: str) -> str:
        # ChromaDB collection names must be 3–63 chars, alphanumeric + dash
        safe = "".join(c if c.isalnum() else "-" for c in paper_id)[:60]
        return f"p-{safe}"

    def get_or_create_collection(self, paper_id: str):
        name = self._collection_name(paper_id)
        return self._chroma.get_or_create_collection(name=name)

    def delete_collection(self, paper_id: str):
        name = self._collection_name(paper_id)
        try:
            self._chroma.delete_collection(name)
            logger.info("Deleted collection: %s", name)
        except Exception as exc:
            logger.warning("Could not delete collection %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, paper_id: str, chunks: list[dict[str, Any]]) -> int:
        """
        Add text chunks to the paper's collection.

        Each chunk dict must have:
          - text: str
          - metadata: dict  (page, chunk_index, paper_title, etc.)

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(paper_id)
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]

        try:
            embeddings = self._embed_texts(texts)
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("Added %d chunks for paper %s", len(chunks), paper_id)
            return len(chunks)
        except Exception as exc:
            logger.error("add_chunks failed for %s: %s", paper_id, exc)
            raise

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        paper_ids: list[str],
        query: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Query one or more paper collections and return the top-k chunks.
        Results are sorted by distance (closest first).
        """
        query_embedding = self._embed_query(query)
        all_results: list[dict] = []

        for pid in paper_ids:
            collection = self.get_or_create_collection(pid)
            try:
                count = collection.count()
                if count == 0:
                    continue
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k, count),
                    include=["documents", "metadatas", "distances"],
                )
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    all_results.append(
                        {
                            "text": doc,
                            "metadata": meta,
                            "distance": dist,
                            "paper_id": pid,
                        }
                    )
            except Exception as exc:
                logger.warning("Query failed for paper %s: %s", pid, exc)

        # Sort all results by similarity score (lower distance = more similar)
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:k]

    def collection_exists(self, paper_id: str) -> bool:
        """Return True if a non-empty collection exists for this paper."""
        try:
            col = self._chroma.get_collection(self._collection_name(paper_id))
            return col.count() > 0
        except Exception:
            return False

    def list_collections(self) -> list[str]:
        """Return all collection names (paper IDs) currently stored."""
        return [c.name for c in self._chroma.list_collections()]


# Module-level singleton
vector_store = VectorStoreManager()
