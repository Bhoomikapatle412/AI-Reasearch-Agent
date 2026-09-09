# -*- coding: utf-8 -*-
"""
Quick smoke-test for all backend agents and routes.
Run:  python -m scripts.test_agents
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.search_agent import search_agent
from backend.agents.trend_agent import trend_agent
from backend.paper_store import paper_store


async def main():
    print("=" * 60)
    print("AI Research Agent — Backend Smoke Tests")
    print("=" * 60)

    # 1. Search
    print("\n[1] Testing paper search (arXiv)...")
    try:
        result = await search_agent.search("transformer language model", limit=3, source="arxiv")
        print(f"    OK Found {result['total']} papers")
        for p in result['papers'][:2]:
            print(f"      - {p['title'][:60]}")
    except Exception as e:
        print(f"    FAIL {e}")

    # 2. Paper store
    print("\n[2] Testing paper store...")
    from scripts.load_demo_data import load_demo_data
    load_demo_data()
    count = paper_store.count()
    print(f"    OK Store has {count} papers")

    # 3. Trend analysis (stats-only, no LLM)
    print("\n[3] Testing trend analysis (statistical)...")
    try:
        papers = paper_store.get_all()
        stats = trend_agent._compute_stats(papers)
        print(f"    OK Year range: {stats.get('year_range')}")
        print(f"    OK Top keywords: {[w for w, _ in stats['top_keywords'][:5]]}")
    except Exception as e:
        print(f"    FAIL {e}")

    print("\n" + "=" * 60)
    print("Smoke tests complete. Start the server:")
    print("  cd backend && uvicorn backend.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
