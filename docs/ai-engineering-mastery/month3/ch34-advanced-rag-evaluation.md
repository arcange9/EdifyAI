# Chapter 34 — Advanced RAG & RAG Evaluation

## Learning Objectives

- Implement advanced RAG techniques (reranking, hybrid search, query expansion)
- Build a RAG evaluation framework
- Measure faithfulness, relevance, and answer quality
- Optimize RAG systems for production

---

## Advanced RAG Techniques

```
┌──────────────────────────────────────────────────────────────┐
│  ADVANCED RAG TECHNIQUES                                    │
│                                                              │
│  1. RERANKING                                               │
│     Retrieve top 20 → rerank to top 5 with a better model    │
│     Bi-encoder (fast, rough) → Cross-encoder (slow, precise)│
│                                                              │
│  2. HYBRID SEARCH                                           │
│     Combine keyword (BM25) + semantic (vector) search       │
│     Best for: technical docs with exact terms               │
│                                                              │
│  3. QUERY EXPANSION                                         │
│     Rewrite user query into multiple variants               │
│     Search with all variants, merge results                 │
│     Best for: ambiguous or short queries                    │
│                                                              │
│  4. MULTI-VECTOR RETRIEVAL                                  │
│     Store summary + full text as separate vectors           │
│     Match on summary, retrieve full text                    │
│                                                              │
│  5. PARENT-CHILD CHUNKING                                   │
│     Embed small chunks (precise)                            │
│     Return parent chunk (more context)                      │
│                                                              │
│  6. METADATA FILTERING                                      │
│     Filter by date, source, category before search         │
│     Speeds up retrieval and improves accuracy              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## RAG Evaluation Metrics

```
┌─────────────────────────────────────────────────────────────┐
│  RAG EVALUATION METRICS                                    │
│                                                             │
│  FAITHFULNESS: Does the answer use only retrieved info?    │
│  → Score 0-1 (no hallucination = 1.0)                     │
│                                                             │
│  RELEVANCE: Is the retrieved context relevant to question? │
│  → Score 0-1 (perfectly relevant = 1.0)                   │
│                                                             │
│  ANSWER RELEVANCE: Does the answer address the question?   │
│  → Score 0-1                                                │
│                                                             │
│  CONTEXT RECALL: What fraction of relevant info was found? │
│  → Score 0-1 (found all relevant = 1.0)                   │
│                                                             │
│  CITATION ACCURACY: Are citations correct?                 │
│  → Score 0-1 (all citations correct = 1.0)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Code — RAG Evaluation

```python
"""Chapter 34 — Advanced RAG & Evaluation"""
import numpy as np

def evaluate_rag(rag_system, test_cases):
    """Evaluate RAG system on test cases."""
    metrics = {"faithfulness": [], "relevance": [], "answer_quality": []}

    for case in test_cases:
        result = rag_system.answer(case["question"])
        # Score each metric (simplified — in production use LLM-as-judge)
        faithfulness = 1.0 if "I don't know" in result["answer"] or \
                        any(s["source"] == case.get("expected_source", s["source"])
                            for s in result["sources"]) else 0.5
        relevance = np.mean([s["score"] for s in result["sources"]])
        answer_quality = 1.0 if len(result["answer"]) > 10 else 0.0

        metrics["faithfulness"].append(faithfulness)
        metrics["relevance"].append(relevance)
        metrics["answer_quality"].append(answer_quality)

    return {k: np.mean(v) for k, v in metrics.items()}

print("=== RAG Evaluation Framework ===")
print("Metrics: faithfulness, relevance, answer_quality")
print("Use LLM-as-judge for production evaluation")
print("✓ Chapter 34 complete!")
```

---

## Chapter Summary

```
• Reranking: retrieve many, rerank with cross-encoder
• Hybrid search: keyword + semantic for better recall
• Query expansion: rewrite queries for ambiguous inputs
• Evaluate: faithfulness, relevance, context recall
• LLM-as-judge: use LLM to score RAG outputs
```
