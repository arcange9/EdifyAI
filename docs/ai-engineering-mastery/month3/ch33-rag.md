# Chapter 33 — Retrieval-Augmented Generation (RAG)

## Learning Objectives

- Understand the complete RAG pipeline end-to-end
- Build a working RAG system from scratch
- Know chunking strategies and their tradeoffs
- Combine retrieval with LLM generation for document Q&A

---

## Why This Matters

RAG is the #1 LLM application pattern in production. It solves LLM hallucination by grounding answers in your documents. ChatGPT's "knowledge base" feature, enterprise search, customer support bots, and document Q&A all use RAG. Mastering RAG means you can build real AI products.

---

## The Complete RAG Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  COMPLETE RAG PIPELINE                                       │
│                                                              │
│  ┌─ INGESTION (one-time or batch) ──────────────────┐       │
│  │                                                   │       │
│  │  Documents → Chunk → Embed → Store               │       │
│  │              ─────     ──────  ─────              │       │
│  │  PDF, DOCX    Split     Create   Save in          │       │
│  │  TXT, URL     into      vectors  vector DB        │       │
│  │               chunks                              │       │
│  └───────────────────────────────────────────────────┘       │
│                                                              │
│  ┌─ QUERY (every user question) ────────────────────┐       │
│  │                                                   │       │
│  │  Question → Embed → Search → Retrieve → Augment  │       │
│  │             ─────   ──────  ────────   ────────   │       │
│  │  User's    Create   Vector   Top-K       Add       │       │
│  │  question  query    DB       chunks      context   │       │
│  │             vector  lookup              to prompt  │       │
│  │                                                   │       │
│  │  Augmented Prompt → LLM → Answer + Citations     │       │
│  │  ────────────────     ───   ──────   ─────────    │       │
│  │  Context + Question   Generate  Response with     │       │
│  │                       answer   source citations   │       │
│  │                                                   │       │
│  └───────────────────────────────────────────────────┘       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Chunking Strategies

```
┌─────────────────────────────────────────────────────────────┐
│  CHUNKING STRATEGIES                                       │
│                                                             │
│  FIXED SIZE:                                                │
│  Split every N characters (e.g., 1000)                     │
│  Pros: simple, predictable                                  │
│  Cons: may break mid-sentence, lose context                │
│                                                             │
│  SENTENCE-BASED:                                            │
│  Split at sentence boundaries                               │
│  Pros: preserves sentence meaning                          │
│  Cons: sentences vary in length                            │
│                                                             │
│  PARAGRAPH-BASED:                                          │
│  Split at paragraph boundaries                             │
│  Pros: preserves topic context                            │
│  Cons: paragraphs can be very long                         │
│                                                             │
│  OVERLAPPING:                                               │
│  Add overlap (e.g., 150 chars) between chunks             │
│  Pros: context doesn't get cut at boundaries               │
│  Cons: slight storage overhead                             │
│                                                             │
│  RECOMMENDED: 1000-2000 chars, 150-200 overlap            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — Complete RAG System

```python
"""
Chapter 33 — Complete RAG System
Builds a working RAG pipeline from scratch.
pip install numpy openai
"""
import numpy as np
import json
import os
from typing import List, Dict, Tuple

# ════════════════════════════════════════════════════════
# STEP 1: DOCUMENT PROCESSING & CHUNKING
# ════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to end at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.5:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap  # Overlap for context
    return chunks

# Demo documents
documents = [
    {
        "id": "doc1",
        "title": "Neural Networks Introduction",
        "text": """Neural networks are computational models inspired by the human brain.
They consist of layers of interconnected neurons that process information.
Each connection between neurons has a weight that determines its importance.
During training, these weights are adjusted to minimize prediction errors.
The simplest neural network is called a perceptron, which has a single layer.
Multi-layer networks can learn complex non-linear patterns.
The most common activation function is ReLU (Rectified Linear Unit).
Training uses backpropagation to compute gradients efficiently."""
    },
    {
        "id": "doc2",
        "title": "Transformer Architecture",
        "text": """Transformers are a neural network architecture introduced in 2017.
They use self-attention mechanisms to process sequential data in parallel.
Unlike RNNs, transformers don't process data sequentially.
This makes them much faster to train on GPUs.
The key components are multi-head attention and feed-forward networks.
Positional encoding adds word order information.
GPT, BERT, and all modern LLMs are based on transformers."""
    },
    {
        "id": "doc3",
        "title": "RAG Overview",
        "text": """RAG stands for Retrieval-Augmented Generation.
It combines information retrieval with text generation.
Documents are chunked and stored in a vector database.
When a user asks a question, relevant chunks are retrieved.
These chunks are added to the prompt as context.
The LLM generates an answer based on the retrieved context.
This reduces hallucinations because answers are grounded in real data."""
    }
]

# Chunk all documents
all_chunks = []
for doc in documents:
    chunks = chunk_text(doc["text"])
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "id": f"{doc['id']}_chunk_{i}",
            "text": chunk,
            "source": doc["title"],
            "doc_id": doc["id"]
        })

print(f"=== RAG Pipeline ===")
print(f"Documents: {len(documents)}")
print(f"Total chunks: {len(all_chunks)}")

# ════════════════════════════════════════════════════════
# STEP 2: EMBEDDING & VECTOR STORE
# ════════════════════════════════════════════════════════

class RAGVectorStore:
    """Vector store with simulated embeddings (use real model in production)."""
    def __init__(self, dim=128):
        self.dim = dim
        self.vectors = []
        self.chunks = []

    def embed(self, text: str) -> np.ndarray:
        """Simulate embedding (replace with real model in production)."""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(self.dim)

    def add(self, chunks: List[Dict]):
        for chunk in chunks:
            vec = self.embed(chunk["text"])
            self.vectors.append(vec)
            self.chunks.append(chunk)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Find most similar chunks to query."""
        query_vec = self.embed(query)
        similarities = []
        for vec, chunk in zip(self.vectors, self.chunks):
            sim = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            similarities.append((chunk, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

store = RAGVectorStore()
store.add(all_chunks)
print(f"Vector store: {len(store.chunks)} chunks indexed")

# ════════════════════════════════════════════════════════
# STEP 3: RAG PIPELINE
# ════════════════════════════════════════════════════════

class RAGSystem:
    """Complete RAG system: retrieve + generate."""
    def __init__(self, vector_store, llm_client=None):
        self.store = vector_store
        self.llm = llm_client

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant chunks for the question."""
        results = self.store.search(question, top_k)
        return [{"text": r[0]["text"], "source": r[0]["source"],
                 "score": float(r[1])} for r in results]

    def build_prompt(self, question: str, retrieved: List[Dict]) -> str:
        """Build the RAG prompt with context."""
        context_parts = []
        for i, r in enumerate(retrieved):
            context_parts.append(f"[{i+1}] (Source: {r['source']})\n{r['text']}")

        context = "\n\n".join(context_parts)
        prompt = f"""Answer the question based ONLY on the following context.

Context:
{context}

Question: {question}

Instructions:
1. Answer using only the information in the context above
2. If the answer is not in the context, say "I don't know based on the provided information"
3. Cite the source for your answer
4. Be concise and accurate

Answer:"""
        return prompt

    def answer(self, question: str, top_k: int = 3) -> Dict:
        """Complete RAG: retrieve → augment → generate."""
        # Step 1: Retrieve
        retrieved = self.retrieve(question, top_k)

        # Step 2: Build augmented prompt
        prompt = self.build_prompt(question, retrieved)

        # Step 3: Generate (simulated for demo)
        if self.llm:
            response = self.llm.chat([
                {"role": "system", "content": "You are a helpful assistant that answers based only on provided context."},
                {"role": "user", "content": prompt}
            ])
        else:
            # Demo: show what would be sent to LLM
            response = f"[Demo] Would send prompt to LLM:\n{prompt[:200]}..."

        return {
            "question": question,
            "answer": response,
            "sources": [{"source": r["source"], "score": r["score"]}
                       for r in retrieved],
            "retrieved_chunks": len(retrieved)
        }

# ════════════════════════════════════════════════════════
# STEP 4: DEMO
# ════════════════════════════════════════════════════════

rag = RAGSystem(store)

questions = [
    "What is a neural network?",
    "How do transformers differ from RNNs?",
    "What does RAG stand for and how does it work?",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"Q: {q}")
    result = rag.answer(q)
    print(f"Retrieved {result['retrieved_chunks']} chunks:")
    for s in result["sources"]:
        print(f"  - {s['source']} (score: {s['score']:.3f})")
    print(f"\nA: {result['answer']}")

print(f"\n{'='*50}")
print("✓ Chapter 33 complete!")
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Chunks too small.* Loses context. Aim for 500-2000 chars.
2. *No overlap.* Information cut at chunk boundaries. Use 100-200 char overlap.
3. *Not citing sources.* Users need to know where the answer came from.
4. *Retrieving too many chunks.* Top 3-5 is usually optimal. More = noise.
5. *Not handling "I don't know."* The system must be able to say it doesn't know.

---

## Mini-Project — Document Q&A System

Build a RAG system that:
1. Reads a text file or PDF
2. Chunks it (500 chars, 100 overlap)
3. Embeds and stores in a vector DB
4. Accepts user questions
5. Returns answers with citations

---

## Chapter Summary

```
• RAG = Retrieve relevant docs → Augment prompt → Generate answer
• Chunking: 500-2000 chars with 100-200 overlap
• Vector DB stores embeddings for fast similarity search
• Retrieve top 3-5 chunks per query
• Always cite sources and handle "I don't know"
• RAG reduces hallucination by grounding in real data
```

---

## Interview Questions

1. "Walk through the RAG pipeline step by step."
2. "How do you choose chunk size and overlap?"
3. "What are the main failure modes of RAG?"
4. "How would you evaluate RAG quality?"
