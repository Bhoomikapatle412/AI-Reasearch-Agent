# 🔬 AI Research Agent

> An agentic AI system for students, researchers, and professionals to **find, understand, organize, and analyze academic research papers** — powered by **IBM watsonx / IBM Granite** and **RAG**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Paper Search** | Search millions of papers via Semantic Scholar + arXiv APIs |
| 📄 **PDF Upload & RAG** | Upload PDFs → extract, chunk, embed into ChromaDB → answer questions |
| 📋 **AI Summarizer** | Structured summaries: problem, methodology, dataset, results, limitations |
| 💬 **Research Q&A Chat** | Ask questions, get answers grounded in your papers with source citations |
| ⚖️ **Paper Comparison** | Side-by-side structured comparison of multiple papers |
| 🕵️ **Research Gap Detection** | Identify unexplored areas, contradictions, and future research directions |
| 📈 **Trend Analysis** | Publication timelines, keyword frequencies, emerging topics with charts |
| 🎛️ **Research Dashboard** | Stats overview: paper count, topics, trends, most-cited papers |

---

## 🏗️ Architecture

```
AI Research Agent
├── backend/                   # FastAPI Python backend
│   ├── agents/
│   │   ├── search_agent.py    # Paper Search Agent (Semantic Scholar + arXiv)
│   │   ├── analysis_agent.py  # Paper Analysis Agent (summarization)
│   │   ├── rag_agent.py       # RAG Research Agent (Q&A)
│   │   ├── comparison_agent.py # Comparison Agent
│   │   ├── gap_agent.py       # Research Gap Agent
│   │   └── trend_agent.py     # Trend Analysis Agent
│   ├── routes/
│   │   ├── search.py          # GET /api/search
│   │   ├── papers.py          # POST /api/papers/upload, GET/DELETE /api/papers
│   │   ├── analyze.py         # POST /api/analyze/{summarize,ask,compare,gaps,trends}
│   │   └── dashboard.py       # GET /api/dashboard
│   ├── main.py                # FastAPI app entry point
│   ├── config.py              # Settings from .env
│   ├── llm_client.py          # IBM watsonx / OpenAI LLM wrapper
│   ├── vector_store.py        # ChromaDB vector store manager
│   ├── pdf_processor.py       # PDF extraction + chunking pipeline
│   └── paper_store.py         # JSON-backed paper registry
├── frontend/                  # React (Vite) frontend
│   └── src/
│       ├── pages/             # Dashboard, Search, Upload, Summarize, QA, Compare, Gaps, Trends
│       ├── components/        # Layout, sidebar, navigation
│       └── api.js             # Axios API client
├── scripts/
│   ├── load_demo_data.py      # Loads 6 landmark demo papers
│   └── test_agents.py         # Backend smoke tests
└── .env.example               # Environment variable template
```

### Agent Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│                                                         │
│  Search Agent ──────► Semantic Scholar / arXiv API      │
│                                                         │
│  PDF Upload ──────► pdf_processor ──► ChromaDB          │
│                                          │              │
│  RAG Agent ────────────────────────────►┘ (similarity   │
│                                           search)       │
│  Analysis Agent ──► LLM (watsonx/Granite) ──► Summary   │
│  Comparison Agent ──────────────────────► Comparison    │
│  Gap Agent ──────────────────────────────► Gaps         │
│  Trend Agent (stats + LLM) ──────────────► Trends       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
React Frontend (Dashboard / Search / Upload / Chat / ...)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- IBM watsonx API key **or** OpenAI API key (at least one required for LLM features)

### 1. Clone & configure

```bash
git clone <repo-url>
cd ai-research-agent

# Copy env template and fill in your keys
cp .env.example .env
```

Edit `.env`:
```
WATSONX_API_KEY=your_ibm_watsonx_api_key
WATSONX_PROJECT_ID=your_project_id
# OR
OPENAI_API_KEY=sk-...
```

> **Note:** Paper Search and Trend Statistics work without any API key. LLM features (summarize, Q&A, compare, gaps) require watsonx or OpenAI.

### 2. Backend setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Load demo data (optional but recommended)

```bash
cd ..
python -m scripts.load_demo_data
```

This loads 6 landmark AI papers (Attention is All You Need, BERT, GPT-3, RAG, DDPM, Llama 2) so you can test the app immediately.

### 4. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/api/docs

### 5. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open: **http://localhost:5173**

---

## 🔧 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `WATSONX_API_KEY` | ✓ (or OpenAI) | IBM watsonx API key |
| `WATSONX_PROJECT_ID` | ✓ (if using watsonx) | IBM watsonx project ID |
| `WATSONX_URL` | No | watsonx endpoint (default: us-south) |
| `OPENAI_API_KEY` | ✓ (or watsonx) | OpenAI fallback key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Increases rate limits (free) |
| `UPLOAD_DIR` | No | PDF storage directory (default: `uploads/`) |
| `CHROMA_PERSIST_DIR` | No | Vector DB directory (default: `chroma_db/`) |
| `CHUNK_SIZE` | No | PDF chunk size in characters (default: 1000) |
| `CHUNK_OVERLAP` | No | Chunk overlap in characters (default: 200) |
| `MAX_UPLOAD_SIZE_MB` | No | Max PDF upload size (default: 50 MB) |

---

## 📚 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check + LLM status |
| `/api/search?q=query` | GET | Search papers |
| `/api/papers` | GET | List all papers |
| `/api/papers/upload` | POST | Upload a PDF |
| `/api/papers/indexed` | GET | List RAG-indexed papers |
| `/api/papers/{id}` | DELETE | Delete a paper |
| `/api/analyze/summarize` | POST | Generate paper summary |
| `/api/analyze/ask` | POST | RAG Q&A |
| `/api/analyze/compare` | POST | Compare papers |
| `/api/analyze/gaps` | POST | Detect research gaps |
| `/api/analyze/trends` | POST | Analyze trends |
| `/api/dashboard` | GET | Dashboard statistics |

Full interactive docs: http://localhost:8000/api/docs

---

## 🤖 IBM watsonx / Granite Integration

The LLM client ([`backend/llm_client.py`](backend/llm_client.py)) uses:

- **Primary**: `ibm/granite-13b-instruct-v2` via `langchain-ibm` + `ibm-watsonx-ai`
- **Fallback**: `gpt-4o-mini` via `langchain-openai`
- **Embeddings**: `all-MiniLM-L6-v2` (HuggingFace, runs locally — no API key required)

All agents are LLM-agnostic and work with either backend. The agentic architecture follows the pattern:

> **Retrieve context → Build structured prompt → Call Granite → Parse structured JSON response**

---

## 🧪 Running Tests

```bash
python -m scripts.test_agents
```

---

## 📁 Project Structure Details

```
backend/
├── agents/           Each agent is an independent, focused module
│   ├── __init__.py
│   ├── search_agent.py    ← async HTTP to S2 + arXiv, merges results
│   ├── analysis_agent.py  ← RAG context → structured JSON summary
│   ├── rag_agent.py       ← multi-paper RAG Q&A with source citations
│   ├── comparison_agent.py ← side-by-side table comparison via LLM
│   ├── gap_agent.py       ← identifies limitations, gaps, contradictions
│   └── trend_agent.py     ← stats (no LLM) + qualitative (LLM)
├── routes/           One file per API resource group
├── main.py           App assembly + CORS + router mounting
├── config.py         Pydantic Settings, reads .env
├── llm_client.py     LLM + embeddings factory
├── vector_store.py   ChromaDB singleton, one collection per paper
├── pdf_processor.py  extract → chunk → embed → store pipeline
└── paper_store.py    JSON-backed in-memory registry

frontend/src/
├── pages/            One page component per feature
├── components/       Shared Layout + CSS
├── api.js            All axios calls in one place
└── index.css         Global dark theme design system
```

---

## 🔮 Future Improvements

- [ ] Citation graph visualization
- [ ] Multi-user support with authentication
- [ ] Integration with Zotero / Mendeley for reference management
- [ ] Support for Semantic Scholar paper recommendations
- [ ] Automated weekly trend reports via email
- [ ] Export summaries/comparisons to PDF/Word
- [ ] IBM Langflow pipeline visualization
- [ ] Browser extension for one-click paper capture
- [ ] Support for arXiv paper categories (e.g., cs.AI, cs.LG)
- [ ] Fine-tuned Granite model for scientific literature

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| **LLM** | IBM watsonx / IBM Granite 13B Instruct |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector DB** | ChromaDB (persistent, local) |
| **RAG Framework** | LangChain |
| **Backend** | FastAPI + Python 3.10+ |
| **Frontend** | React 18 + Vite |
| **Charts** | Recharts |
| **PDF Parsing** | pdfplumber + pypdf |
| **Academic APIs** | Semantic Scholar, arXiv |

---

## 🎓 Learning Resources

- [IBM watsonx Documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/welcome-main.html)
- [IBM Granite Models](https://www.ibm.com/granite)
- [LangChain Docs](https://python.langchain.com)
- [ChromaDB Docs](https://docs.trychroma.com)
- [Semantic Scholar API](https://api.semanticscholar.org/api-docs/)
- [arXiv API](https://arxiv.org/help/api)

---

## 📄 License

MIT License — free to use, modify, and distribute for educational purposes.
