"""
AI Research Agent — FastAPI application entry point.
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.routes import analyze, dashboard, papers, search

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title="AI Research Agent",
    description="Agentic AI system for finding, understanding, and analyzing research papers.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow the React dev server on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# API Routers — register BEFORE static files
# ------------------------------------------------------------------
app.include_router(search.router)
app.include_router(papers.router)
app.include_router(analyze.router)
app.include_router(dashboard.router)


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "has_watsonx": settings.has_watsonx,
        "has_openai": settings.has_openai,
        "version": "1.0.0",
    }


# ------------------------------------------------------------------
# Serve React build in production ONLY (not during dev with --reload)
# Mount LAST so API routes take priority
# ------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(STATIC_DIR) and not settings.debug:
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    logger.info("Serving React frontend from %s", STATIC_DIR)


# ------------------------------------------------------------------
# Dev runner
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
    )
