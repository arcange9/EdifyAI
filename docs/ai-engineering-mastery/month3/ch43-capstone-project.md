# Chapter 43 — CAPSTONE PROJECT: Production AI Application

## Learning Objectives

- Build a complete production AI application from scratch
- Combine Python + ML + LLM + RAG + APIs + Database + Backend + Deployment
- Apply everything learned in the 90-day journey

---

## Project: AI Study Companion

A production AI application that helps students study by:
1. Uploading documents (PDF, TXT)
2. Asking questions about the documents (RAG)
3. Generating study materials (notes, flashcards, quizzes)
4. Using an AI agent to research and explain topics
5. Tracking study progress

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  AI STUDY COMPANION — ARCHITECTURE                         │
│                                                              │
│  ┌─────────────────────────────────────────────┐           │
│  │  FRONTEND (React/Next.js or CLI)              │           │
│  │  • Upload documents                          │           │
│  │  • Chat interface                           │           │
│  │  • Study tools dashboard                     │           │
│  └──────────────────┬──────────────────────────┘           │
│                     ▼                                        │
│  ┌─────────────────────────────────────────────┐           │
│  │  API LAYER (FastAPI)                           │           │
│  │  • POST /documents/upload                     │           │
│  │  • POST /chat                                 │           │
│  │  • POST /generate/notes                       │           │
│  │  • POST /generate/flashcards                  │           │
│  │  • POST /generate/quiz                        │           │
│  │  • GET /study-plan                            │           │
│  └──────────────────┬──────────────────────────┘           │
│                     ▼                                        │
│  ┌─────────────────────────────────────────────┐           │
│  │  AI PIPELINE                                  │           │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │           │
│  │  │  RAG     │ │  Agent   │ │Generator │    │           │
│  │  │  Engine  │ │  Engine  │ │  Engine  │    │           │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘    │           │
│  │       │            │            │            │           │
│  │  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐     │           │
│  │  │Vector DB│ │Tools    │ │LLM API  │     │           │
│  │  │ChromaDB│ │Search   │ │OpenAI/  │     │           │
│  │  │         │ │Calc     │ │Ollama   │     │           │
│  │  └─────────┘ └─────────┘ └─────────┘     │           │
│  └─────────────────────────────────────────────┘           │
│                     ▼                                        │
│  ┌─────────────────────────────────────────────┐           │
│  │  DATA LAYER                                    │           │
│  │  • PostgreSQL (users, documents, history)   │           │
│  │  • ChromaDB (vector embeddings)              │           │
│  │  • File storage (uploaded PDFs)              │           │
│  └─────────────────────────────────────────────┘           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ai-study-companion/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Database models (SQLAlchemy)
│   ├── schemas.py           # Pydantic request/response models
│   ├── rag.py               # RAG engine (chunking, retrieval)
│   ├── agent.py             # Agent engine (tools, reasoning)
│   ├── generator.py         # Content generation (notes, quizzes)
│   └── config.py            # Configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Core Implementation

```python
"""
Chapter 43 — Capstone: AI Study Companion
A complete production AI application.

requirements.txt:
  fastapi
  uvicorn
  openai
  chromadb
  python-multipart
  PyPDF2
  pydantic
"""

# ─── app/main.py ───────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os

app = FastAPI(title="AI Study Companion", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─── Data Models ───────────────────────────────
class ChatRequest(BaseModel):
    question: str
    document_id: Optional[str] = None

class GenerationRequest(BaseModel):
    document_id: str
    content_type: str  # "notes", "flashcards", "quiz"

# ─── In-memory storage (replace with DB in production) ───
documents = {}  # id → {title, text, chunks}
vector_store = []  # Simple list-based vector store

# ─── Chunking ──────────────────────────────────
def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ─── Endpoints ────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "documents": len(documents)}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for RAG."""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    doc_id = f"doc_{len(documents) + 1}"
    chunks = chunk_text(text)

    documents[doc_id] = {
        "title": file.filename,
        "text": text,
        "chunks": chunks
    }

    return {"id": doc_id, "title": file.filename, "chunks": len(chunks)}

@app.post("/chat")
def chat(request: ChatRequest):
    """Ask a question about a document (RAG)."""
    if request.document_id and request.document_id in documents:
        doc = documents[request.document_id]
        # Simple retrieval: find most relevant chunk
        query_words = set(request.question.lower().split())
        scored = [(chunk, len(query_words & set(chunk.lower().split())))
                  for chunk in doc["chunks"]]
        scored.sort(key=lambda x: x[1], reverse=True)
        context = "\n\n".join([c for c, s in scored[:3] if s > 0])

        prompt = f"""Context: {context}

Question: {request.question}

Answer based on the context. If not in context, say 'I don't know'.
Answer:"""
    else:
        prompt = request.question

    # In production: call LLM API here
    answer = f"[RAG Response] Based on the document, here's the answer to: {request.question}"

    return {"answer": answer, "sources": [request.document_id] if request.document_id else []}

@app.post("/generate")
def generate_content(request: GenerationRequest):
    """Generate study materials from a document."""
    if request.document_id not in documents:
        raise HTTPException(404, "Document not found")

    doc = documents[request.document_id]

    if request.content_type == "notes":
        result = {"notes": f"Study notes for {doc['title']}:\n\nKey concepts..."}
    elif request.content_type == "flashcards":
        result = {"flashcards": [{"front": "What is...?", "back": "It is..."}]}
    elif request.content_type == "quiz":
        result = {"questions": [{"q": "Question?", "options": ["A","B","C"], "answer": 0}]}
    else:
        raise HTTPException(400, "Invalid content type")

    return result

# ─── Run ────────────────────────────────────────
# uvicorn app.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive API docs!

print("""
╔═══════════════════════════════════════════════════════╗
║         AI STUDY COMPANION — CAPSTONE                 ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  POST /documents/upload  → Upload PDF/TXT             ║
║  POST /chat              → Ask questions (RAG)        ║
║  POST /generate          → Create notes/flashcards     ║
║  GET  /health            → Check status               ║
║  GET  /docs              → Interactive API docs       ║
║                                                       ║
║  Run: uvicorn app.main:app --reload --port 8000       ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝

To deploy:
1. Build: docker build -t ai-study-companion .
2. Run: docker-compose up
3. Or deploy to Railway/Render/Fly.io

This combines: Python + ML + LLM + RAG + API + Database + Docker
""")

print("✓ Chapter 43 complete! THE 90-DAY JOURNEY IS COMPLETE!")
print("=" * 50)
```

---

## Capstone Evaluation Rubric

```
┌──────────────────────────────────────────────────────────────┐
│  CAPSTONE EVALUATION RUBRIC                                │
│                                                              │
│  CRITERIA               WEIGHT   EXCELLENT (10)  POOR (1)  │
│  ──────────────────────────────────────────────────────────  │
│  RAG Pipeline           20%     Accurate retrieval, no    │
│                                 hallucination, citations   │
│                                                              │
│  API Design             15%     RESTful, documented,      │
│                                 proper error handling      │
│                                                              │
│  Code Quality           15%     Clean, typed, tested,     │
│                                 documented, modular        │
│                                                              │
│  Deployment             15%     Dockerized, deployed,     │
│                                 health checks work          │
│                                                              │
│  AI Features            15%     RAG + generation + agent   │
│                                 all functional              │
│                                                              │
│  Documentation          10%     README, API docs, arch     │
│                                                              │
│  Testing                10%     Unit tests, integration    │
│                                 tests, RAG eval            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## What You've Built Over 90 Days

```
┌──────────────────────────────────────────────────────────────┐
│  YOUR AI ENGINEERING JOURNEY — COMPLETE                   │
│                                                              │
│  MONTH 1 (Days 1-30):  FOUNDATIONS                         │
│  ✓ Python, NumPy, Pandas, Matplotlib                       │
│  ✓ Linear algebra, probability, calculus                   │
│  ✓ Data preprocessing, feature engineering                 │
│  ✓ Machine learning (regression, classification, trees)    │
│  ✓ Model evaluation, cross-validation                     │
│                                                              │
│  MONTH 2 (Days 31-60):  DEEP LEARNING                      │
│  ✓ Neural networks from scratch                            │
│  ✓ Backpropagation, gradient descent                       │
│  ✓ PyTorch, CNNs, RNNs/LSTMs                               │
│  ✓ Transformers, attention, LLMs                           │
│  ✓ Hugging Face, fine-tuning, LoRA, quantization           │
│                                                              │
│  MONTH 3 (Days 61-90):  PRODUCTION AI                      │
│  ✓ LLM APIs, structured outputs, tool calling             │
│  ✓ Vector databases, semantic search                       │
│  ✓ RAG (basic + advanced)                                  │
│  ✓ AI agents, multi-agent systems                          │
│  ✓ FastAPI, Docker, deployment                             │
│  ✓ Monitoring, cost optimization, security                  │
│  ✓ Production architecture, AI products                    │
│  ✓ CAPSTONE: Full-stack AI application                     │
│                                                              │
│  You went from beginner to building a production AI app.  │
│  That's extraordinary. Now go build more.                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```
