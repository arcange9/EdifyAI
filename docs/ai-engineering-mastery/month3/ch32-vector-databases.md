# Chapter 32 — Vector Databases & Semantic Search

## Learning Objectives

- Understand vector databases and why they power RAG
- Build a semantic search engine with ChromaDB
- Know FAISS, ChromaDB, and Pinecone
- Implement similarity search with embeddings

---

## Why This Matters

Semantic search is the foundation of RAG. Instead of matching keywords, you match *meaning*. "How to bake bread" should find "bread recipes" even without exact keyword match. Vector databases make this possible at scale.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  SEMANTIC SEARCH PIPELINE                                    │
│                                                              │
│  DOCUMENTS:                                                  │
│  "Neural networks learn patterns"                            │
│  "Deep learning uses many layers"                            │
│  "Python is great for AI"                                    │
│         │                                                    │
│         ▼ Embedding Model                                    │
│  [0.2, 0.5, ...]  [0.3, 0.6, ...]  [0.1, 0.4, ...]         │
│         │                                                    │
│         ▼ Store in Vector DB                                 │
│  ┌──────────────────────────────────────┐                   │
│  │  Vector Database (ChromaDB/FAISS)    │                   │
│  │  ┌─────┬──────────┬────────────┐     │                   │
│  │  │ ID  │ Embedding │  Text/Metadata│  │                   │
│  │  │  1  │ [0.2,...]│ "Neural..."   │  │                   │
│  │  │  2  │ [0.3,...]│ "Deep..."     │  │                   │
│  │  │  3  │ [0.1,...]│ "Python..."   │  │                   │
│  │  └─────┴──────────┴────────────┘     │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
│  QUERY: "machine learning"                                   │
│     │                                                        │
│     ▼ Embedding Model                                        │
│  [0.25, 0.55, ...]                                          │
│     │                                                        │
│     ▼ Similarity Search (cosine)                            │
│  ┌──────────────────────────────────────┐                   │
│  │  Top results:                        │                   │
│  │  1. "Neural networks learn patterns" │ (0.92)            │
│  │  2. "Deep learning uses many layers" │ (0.88)            │
│  │  3. "Python is great for AI"        │ (0.45)            │
│  └──────────────────────────────────────┘                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Semantic Search with ChromaDB

```python
"""
Chapter 32 — Vector Databases & Semantic Search
pip install chromadb numpy
"""
import numpy as np

# ─── 1. Simple vector store from scratch ──────
class SimpleVectorStore:
    """Minimal vector store using cosine similarity."""

    def __init__(self):
        self.vectors = []
        self.documents = []
        self.metadata = []

    def add(self, documents, vectors, metadatas=None):
        for i, (doc, vec) in enumerate(zip(documents, vectors)):
            self.documents.append(doc)
            self.vectors.append(np.array(vec))
            self.metadata.append(metadatas[i] if metadatas else {})

    def search(self, query_vector, top_k=3):
        """Return top_k most similar documents."""
        query_vec = np.array(query_vector)
        similarities = [
            np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v))
            for v in self.vectors
        ]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.documents[i], similarities[i], self.metadata[i])
                for i in top_indices]

# Demo with fake embeddings (in production, use a real embedding model)
np.random.seed(42)

documents = [
    "Neural networks learn patterns from data",
    "Deep learning uses many layers of neurons",
    "Python is the most popular language for AI",
    "Transformers revolutionized natural language processing",
    "RAG combines retrieval with generation",
    "CNNs are used for image recognition",
    "Gradient descent optimizes neural network weights",
    "Tokenization converts text to numbers for LLMs",
]

# Simulate embeddings (similar documents get similar vectors)
doc_embeddings = np.random.randn(len(documents), 384)
# Make similar docs closer
doc_embeddings[0] = doc_embeddings[1] + np.random.randn(384) * 0.1  # NN & DL similar
doc_embeddings[3] = doc_embeddings[0] + np.random.randn(384) * 0.2   # Transformers & NN

store = SimpleVectorStore()
store.add(documents, doc_embeddings)

# Query: "neural network architecture"
query_embedding = doc_embeddings[0] + np.random.randn(384) * 0.1
results = store.search(query_embedding, top_k=3)

print("=== Semantic Search Results ===")
print("Query: 'neural network architecture'\n")
for doc, score, meta in results:
    print(f"  [{score:.4f}] {doc}")

# ─── 2. Using ChromaDB (production) ────────────
print("\n=== ChromaDB (Production Vector DB) ===\n")

print("""# pip install chromadb
import chromadb

# Create client
client = chromadb.Client()
collection = client.create_collection("documents")

# Add documents with embeddings
collection.add(
    documents=["Neural networks learn patterns",
                "Python is great for AI",
                "RAG combines retrieval with generation"],
    metadatas=[{"source": "ch16"}, {"source": "ch3"}, {"source": "ch33"}],
    ids=["doc1", "doc2", "doc3"]
)

# Query (ChromaDB generates embeddings automatically)
results = collection.query(
    query_texts=["machine learning"],
    n_results=2
)
print(results['documents'][0])
""")

# ─── 3. Comparison of vector DBs ───────────────
print("=== Vector Database Comparison ===\n")
print(f"{'Database':<15} {'Type':<12} {'Self-hosted':<12} {'Best for'}")
print("─" * 60)
print(f"{'ChromaDB':<15} {'Embedded':<12} {'Yes':<12} {'Dev/small apps'}")
print(f"{'FAISS':<15} {'Library':<12} {'Yes':<12} {'High-speed search'}")
print(f"{'Pinecone':<15} {'Cloud':<12} {'No':<12} {'Production at scale'}")
print(f"{'Weaviate':<15} {'Server':<12} {'Yes':<12} {'Multi-modal + GraphQL'}")
print(f"{'Qdrant':<15} {'Server':<12} {'Yes':<12} {'Production + filtering'}")
print(f"{'pgvector':<15} {'Extension':<12} {'Yes':<12} {'PostgreSQL apps'}")

print("\n✓ Chapter 32 complete!")
```

---

## Chapter Summary

```
• Vector DB stores embeddings + metadata + text
• Search = cosine similarity between query and stored vectors
• ChromaDB: easy embedded vector DB for development
• FAISS: fast similarity search library (Meta)
• Pinecone/Qdrant: cloud/production vector databases
• pgvector: vector search inside PostgreSQL
```

---

## Interview Questions

1. "How does semantic search differ from keyword search?"
2. "What is a vector database and when would you use one?"
3. "How do you choose between FAISS, ChromaDB, and Pinecone?"
4. "What is cosine similarity and why is it used for embeddings?"
