"""
Paper Store — in-memory paper registry with JSON persistence.
Tracks uploaded and searched papers, their metadata, and index status.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STORE_FILE = "papers_store.json"


class PaperStore:
    """
    Simple file-backed paper registry.
    Each entry: {"paper_id", "title", "abstract", "authors", "year",
                 "citations", "url", "pdf_url", "topics", "source",
                 "indexed", "file_path"}
    """

    def __init__(self, store_path: str = STORE_FILE):
        self._path = store_path
        self._papers: dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._papers = json.load(f)
                logger.info("Loaded %d papers from store", len(self._papers))
            except Exception as exc:
                logger.warning("Could not load paper store: %s", exc)
                self._papers = {}

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._papers, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not save paper store: %s", exc)

    # ------------------------------------------------------------------

    def add(self, paper: dict[str, Any]) -> str:
        """Add or update a paper. Returns paper_id."""
        pid = paper.get("paper_id") or paper.get("id")
        if not pid:
            raise ValueError("paper must have a paper_id or id field")
        paper["paper_id"] = pid
        self._papers[pid] = {**self._papers.get(pid, {}), **paper}
        self._save()
        return pid

    def get(self, paper_id: str) -> dict | None:
        return self._papers.get(paper_id)

    def get_all(self) -> list[dict]:
        return list(self._papers.values())

    def delete(self, paper_id: str) -> bool:
        if paper_id in self._papers:
            del self._papers[paper_id]
            self._save()
            return True
        return False

    def mark_indexed(self, paper_id: str, num_chunks: int = 0):
        if paper_id in self._papers:
            self._papers[paper_id]["indexed"] = True
            self._papers[paper_id]["num_chunks"] = num_chunks
            self._save()

    def count(self) -> int:
        return len(self._papers)

    def get_topics(self) -> list[str]:
        topics: set[str] = set()
        for paper in self._papers.values():
            for t in paper.get("topics", []):
                if t:
                    topics.add(t)
        return sorted(topics)

    def get_indexed_ids(self) -> list[str]:
        return [pid for pid, p in self._papers.items() if p.get("indexed")]


paper_store = PaperStore()
