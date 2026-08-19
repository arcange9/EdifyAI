# Chapter 22 — Attention Mechanisms & Transformers

## Learning Objectives

- Understand attention and why it revolutionized NLP
- Know the Transformer architecture block by block
- Understand self-attention, multi-head attention, and positional encoding
- See how Transformers replaced RNNs and became the foundation of LLMs

---

## Why This Matters

The Transformer architecture ("Attention Is All You Need", 2017) is the single most important breakthrough in modern AI. GPT, BERT, ChatGPT, Claude, Gemini — every major LLM is a Transformer. Understanding attention is understanding how modern AI *thinks*.

---

## Concept Explanation

### The Attention Idea

```
┌──────────────────────────────────────────────────────────────┐
│  ATTENTION: "PAY ATTENTION TO WHAT MATTERS"                  │
│                                                              │
│  Sentence: "The cat sat on the mat because IT was tired"    │
│                                                              │
│  What does "IT" refer to? → "the cat" (not "the mat")       │
│                                                              │
│  Attention allows the model to focus on relevant parts        │
│  of the input when processing each word.                     │
│                                                              │
│  For "IT":                                                   │
│    "The"     → 0.05 (low attention)                          │
│    "cat"     → 0.45 (HIGH — this is what "it" refers to!)    │
│    "sat"     → 0.10                                          │
│    "on"      → 0.02                                          │
│    "the"     → 0.05                                          │
│    "mat"     → 0.15                                          │
│    "because" → 0.08                                          │
│    "was"     → 0.05                                          │
│    "tired"   → 0.05                                          │
│                                                              │
│  These weights are LEARNED during training.                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Self-Attention (The Core Mechanism)

```
┌──────────────────────────────────────────────────────────────┐
│  SELF-ATTENTION STEP BY STEP                                │
│                                                              │
│  For each word (token), compute:                             │
│  1. Query (Q): "What am I looking for?"                     │
│  2. Key (K):   "What do I contain?"                         │
│  3. Value (V): "Here is my information"                      │
│                                                              │
│  Formula:                                                    │
│  Attention(Q,K,V) = softmax(Q × K^T / √d) × V              │
│                                                              │
│  In simple terms:                                            │
│  1. Score how relevant each word is to each other word       │
│     (dot product of Q and K)                                 │
│  2. Normalize scores (softmax) → sum to 1                   │
│  3. Multiply by values → weighted sum of information        │
│                                                              │
│  Visual:                                                     │
│                                                              │
│  Words:   ["The", "cat", "sat", "on", "mat"]               │
│                                                              │
│  Q = Word × Wq   (What am I looking for?)                   │
│  K = Word × Wk   (What do I contain?)                       │
│  V = Word × Wv   (What info do I provide?)                   │
│                                                              │
│  Scores = Q × K^T  → 5×5 matrix of relevance scores        │
│  Weights = softmax(Scores / √d)  → normalized (sum to 1)    │
│  Output = Weights × V  → each word gets a new representation │
│         that incorporates context from ALL other words       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Multi-Head Attention

```
┌──────────────────────────────────────────────────────────────┐
│  MULTI-HEAD ATTENTION                                       │
│                                                              │
│  Instead of one attention mechanism, use MANY:              │
│                                                              │
│  Head 1: focuses on grammar                                │
│  Head 2: focuses on entity relationships                    │
│  Head 3: focuses on sentiment                               │
│  Head 4: focuses on position                                │
│  ...                                                        │
│  Head 8: focuses on negation                               │
│                                                              │
│  Each head has its OWN Q, K, V weights.                      │
│  Results are concatenated and linearly combined.            │
│                                                              │
│  This allows the model to attend to DIFFERENT aspects       │
│  of the input simultaneously.                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Complete Transformer Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  TRANSFORMER ARCHITECTURE (GPT-style, decoder-only)        │
│                                                              │
│  ┌─────────────────────────────────────────┐               │
│  │  INPUT TOKENS                            │               │
│  │  ["The", "cat", "sat"]                   │               │
│  └─────────────────┬───────────────────────┘               │
│                    ▼                                         │
│  ┌─────────────────────────────────────────┐               │
│  │  TOKEN EMBEDDINGS + POSITIONAL ENCODING  │               │
│  │  Each token → vector + position info    │               │
│  └─────────────────┬───────────────────────┘               │
│                    ▼                                         │
│  ┌─────────────────────────────────────────┐               │
│  │  ┌─────────────────────────────────┐   │               │
│  │  │  MULTI-HEAD SELF-ATTENTION       │   │               │
│  │  │  (each word attends to others)  │   │               │
│  │  └─────────────────┬───────────────┘   │               │
│  │                    ▼                     │  TRANSFORMER │
│  │  ┌─────────────────────────────────┐   │    BLOCK     │
│  │  │  ADD & NORM (residual + layer    │   │  (repeated  │
│  │  │  normalization)                  │   │   Nx times) │
│  │  └─────────────────┬───────────────┘   │               │
│  │                    ▼                     │               │
│  │  ┌─────────────────────────────────┐   │               │
│  │  │  FEED FORWARD NETWORK (2 linear  │   │               │
│  │  │  layers + ReLU)                  │   │               │
│  │  └─────────────────┬───────────────┘   │               │
│  │                    ▼                     │               │
│  │  ┌─────────────────────────────────┐   │               │
│  │  │  ADD & NORM                     │   │               │
│  │  └─────────────────┬───────────────┘   │               │
│  └─────────────────────┼───────────────────┘               │
│                        ▼ (repeat N blocks)                   │
│  ┌─────────────────────────────────────────┐               │
│  │  LINEAR LAYER → LOGITS                  │               │
│  │  (project to vocabulary size)           │               │
│  └─────────────────┬───────────────────────┘               │
│                    ▼                                         │
│  ┌─────────────────────────────────────────┐               │
│  │  SOFTMAX → NEXT TOKEN PROBABILITY       │               │
│  │  P("on") = 0.75, P("down") = 0.15, ...  │               │
│  └─────────────────────────────────────────┘               │
│                                                              │
│  KEY: NO recurrent connections, NO LSTM gates               │
│  ALL words processed simultaneously (parallel!)              │
│  This is why Transformers can be trained on GPUs at scale  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Positional Encoding

```
┌──────────────────────────────────────────────────────────────┐
│  POSITIONAL ENCODING                                         │
│                                                              │
│  Self-attention has no notion of word order.                 │
│  "Dog bites man" and "Man bites dog" would be identical!   │
│                                                              │
│  Solution: add position information to each embedding.       │
│                                                              │
│  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))            │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))            │
│                                                              │
│  Each position gets a unique sinusoidal pattern.             │
│  This is added to the token embedding before attention.      │
│                                                              │
│  Result: model knows BOTH what the word is AND where it is   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Mini Transformer from Scratch

```python
"""
Chapter 22 — Mini Transformer (simplified)
Shows the core components: attention, multi-head, feed-forward.
pip install torch numpy matplotlib
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        scores = Q @ K.transpose(-2, -1) / self.scale
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.attentions = nn.ModuleList([
            SelfAttention(self.head_dim) for _ in range(n_heads)
        ])
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq, dim = x.shape
        heads = [att(x[..., i*self.head_dim:(i+1)*self.head_dim])
                 for i, att in enumerate(self.attentions)]
        out = torch.cat([h[0] for h in heads], dim=-1)
        return self.fc(out), [h[1] for h in heads]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))  # Residual + Norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))     # Residual + Norm
        return x, weights

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_heads=4, n_blocks=2, max_seq=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq, embed_dim) * 0.01)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, embed_dim*4)
            for _ in range(n_blocks)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_encoding[:, :seq_len]
        all_weights = []
        for block in self.blocks:
            emb, weights = block(emb)
            all_weights.append(weights)
        return self.fc(emb), all_weights

# ─── Demo ──────────────────────────────────────
vocab_size = 100
model = MiniTransformer(vocab_size, embed_dim=64, n_heads=4, n_blocks=2)
x = torch.randint(0, vocab_size, (4, 10))  # Batch of 4, seq len 10
output, attention_weights = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of attention layers: {len(attention_weights)}")
print(f"Attention weights shape: {attention_weights[0][0].shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Visualize attention weights
fig, ax = plt.subplots(figsize=(8, 8))
weights = attention_weights[0][0][0].detach().numpy()  # First sample, first head
im = ax.imshow(weights, cmap='hot')
ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title('Self-Attention Weights (Head 1, Sample 1)')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('ch22_attention_weights.png', dpi=150)
plt.show()

print("\n✓ Chapter 22 complete!")
```

---

## Chapter Summary

```
• Attention = weighted focus on relevant parts of input
• Self-attention: Q×K→scores→softmax→weighted V
• Multi-head: multiple attention mechanisms in parallel
• Transformer block: attention → add+norm → feed-forward → add+norm
• Positional encoding: adds word order info
• No recurrence → fully parallel → GPU friendly → scalable
• This architecture powers GPT, BERT, Claude, and all modern LLMs
```

---

## Interview Questions

1. "Explain self-attention in your own words."
2. "Why do Transformers outperform RNNs?"
3. "What is multi-head attention and why use multiple heads?"
4. "What problem does positional encoding solve?"

---

## Knowledge Check

1. What are Q, K, and V in attention?
2. Why can't self-attention work without positional encoding?
3. What are the components of a Transformer block?
4. Why is the scale factor (√d) important in attention?
5. How does attention solve the vanishing gradient problem of RNNs?
