# Chapter 38 — Backend Development for AI: FastAPI

## Learning Objectives

- Build REST APIs for AI applications with FastAPI
- Implement streaming responses for real-time AI output
- Handle async operations and WebSocket connections
- Structure an AI backend properly

---

## Code — AI API with FastAPI

```python
"""
Chapter 38 — FastAPI for AI backends
Builds a complete AI API server.
pip install fastapi uvicorn
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

app = FastAPI(title="Edify AI API", version="1.0.0")

# ─── Models ────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500

class ChatResponse(BaseModel):
    answer: str
    sources: list = []
    tokens_used: int = 0

# ─── Endpoints ────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Edify AI API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    # In production: call LLM here
    answer = f"Response to: {request.message}"
    return ChatResponse(answer=answer, sources=[], tokens_used=50)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint (token by token)."""
    async def generate():
        words = f"Here is a response to {request.message}".split()
        for word in words:
            await asyncio.sleep(0.05)
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

# ─── Run ───────────────────────────────────────
# uvicorn main:app --reload --port 8000
# Then visit http://localhost:8000/docs for interactive API docs!

print("""To run the API:
  uvicorn main:app --reload --port 8000

Endpoints:
  GET  /health     → health check
  POST /chat       → get AI response
  POST /chat/stream → streaming response (SSE)
  GET  /docs       → interactive Swagger UI
""")

print("✓ Chapter 38 complete!")
```

## Chapter Summary

```
• FastAPI: fast, async, auto-generated docs
• POST /chat for regular responses
• StreamingResponse for token-by-token output
• Pydantic models for request validation
• /docs endpoint provides interactive API testing
• Production: add CORS, rate limiting, authentication
```
