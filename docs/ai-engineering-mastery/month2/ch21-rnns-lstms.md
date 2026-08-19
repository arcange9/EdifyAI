# Chapter 21 — Recurrent Neural Networks & LSTMs

## Learning Objectives

- Understand sequential data and why standard NNs can't handle it
- Know how RNNs process sequences and maintain memory
- Understand LSTMs and the gate mechanism
- Build an RNN/LSTM for text classification

---

## Why This Matters

Text, speech, stock prices, weather — all are *sequential* data. Transformers (Chapter 22) are the modern approach, but RNNs/LSTMs are still used in production and are essential for understanding how sequence modeling evolved.

---

## Concept Explanation

### Why Sequences Need Special Networks

```
Standard NN:     Each input processed independently
RNN:            Each input depends on PREVIOUS inputs

Text example:
"the cat sat on the ___" → "mat"

A standard NN can't use the context "the cat sat on the"
An RNN passes information from previous words to the next step
```

### How RNNs Work

```
┌──────────────────────────────────────────────────────────────┐
│  RNN UNROLLED OVER TIME                                     │
│                                                              │
│  Time t-2    Time t-1    Time t      Time t+1                │
│                                                              │
│  x₁ ●──→ h₁ ●──→ h₂ ●──→ h₃ ●──→ ...                      │
│         ↑          ↑          ↑                              │
│    RNN cell   RNN cell   RNN cell                            │
│                                                              │
│  At each time step:                                          │
│    h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)            │
│                                                              │
│  h_t = hidden state (memory) at time t                      │
│  h_{t-1} = hidden state from previous step                  │
│  x_t = input at current step                                │
│  W_hh = weights for hidden state                            │
│  W_xh = weights for input                                   │
│                                                              │
│  The hidden state carries information from the past.        │
│  This is the network's "memory."                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### The Vanishing Gradient Problem

```
┌──────────────────────────────────────────────────────────────┐
│  WHY VANILLA RNNs FAIL                                       │
│                                                              │
│  In a long sequence (100+ words), the gradient signal        │
│  from the end gets multiplied by small numbers repeatedly    │
│  going backward through the chain rule.                     │
│                                                              │
│  Result: gradient ≈ 0 → network can't learn long-range      │
│  dependencies.                                               │
│                                                              │
│  "The cat, which was black and had green eyes and lived      │
│   in the house for many years, ___ on the mat"               │
│                                                              │
│  A vanilla RNN forgets "cat" by the time it reaches "___"    │
│                                                              │
│  SOLUTION: LSTM (Long Short-Term Memory)                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### LSTM — The Solution

```
┌──────────────────────────────────────────────────────────────┐
│  LSTM CELL — THREE GATES                                    │
│                                                              │
│  ┌─────────────────────────────────────────┐                │
│  │           LSTM CELL                     │                │
│  │                                         │                │
│  │  Cell State (long-term memory)          │                │
│  │  ────────────────────────────────────── │                │
│  │  │   FORGET GATE: what to erase        │ │                │
│  │  │   INPUT GATE: what to add           │ │                │
│  │  │   OUTPUT GATE: what to output       │ │                │
│  │  ────────────────────────────────────── │                │
│  │                                         │                │
│  │  Forget gate: decides what to remove from memory         │
│  │  Input gate:  decides what new info to store             │
│  │  Output gate: decides what part of memory to output      │
│  │                                         │                │
│  │  Each gate is a small neural network (sigmoid: 0 or 1)   │
│  │                                         │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
│  LSTM can remember information for HUNDREDS of time steps.  │
│  The gates learn WHEN to remember and WHEN to forget.       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Sentiment Analysis with LSTM

```python
"""
Chapter 21 — RNNs & LSTMs for Text Classification
pip install torch numpy
"""
import torch
import torch.nn as nn
import numpy as np

# ─── Synthetic sentiment data ──────────────────
positive_reviews = [
    "amazing great wonderful fantastic excellent",
    "love this product best ever",
    "perfect outstanding brilliant superb",
    "happy satisfied delighted pleased",
    "incredible awesome magnificent terrific",
    "good nice lovely enjoyable fun",
] * 20

negative_reviews = [
    "terrible awful horrible disgusting bad",
    "hate this worst product ever",
    "disappointing frustrating annoying terrible",
    "sad angry upset disappointed",
    "horrible bad terrible waste money",
    "broken damaged useless poor quality",
] * 20

texts = positive_reviews + negative_reviews
labels = [1]*len(positive_reviews) + [0]*len(negative_reviews)

# Build vocabulary
vocab = set()
for text in texts:
    vocab.update(text.split())
vocab = sorted(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Encode text to sequences of integers
def encode(text, max_len=10):
    tokens = text.split()[:max_len]
    ids = [word2idx[w] for w in tokens if w in word2idx]
    while len(ids) < max_len:
        ids.append(0)  # Pad
    return ids

X = torch.LongTensor([encode(t) for t in texts])
y = torch.LongTensor(labels)

# ─── LSTM Model ────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)        # (batch, seq, embed)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        out = self.dropout(hidden[-1])      # Last hidden state
        return self.fc(out)                 # Classification

model = SentimentLSTM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ─── Train ─────────────────────────────────────
for epoch in range(30):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        acc = (output.max(1)[1] == y).float().mean().item()
        print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f}")

# Test with new sentences
test_sentences = ["amazing wonderful love it", "terrible bad worst hate"]
test_encoded = torch.LongTensor([encode(t) for t in test_sentences])
with torch.no_grad():
    preds = model(test_encoded)
    for sent, pred in zip(test_sentences, preds):
        sentiment = "Positive" if pred.argmax() == 1 else "Negative"
        print(f"  '{sent}' → {sentiment}")

print("\n✓ Chapter 21 complete!")
```

---

## Chapter Summary

```
• RNNs process sequences, maintaining hidden state (memory)
• Vanishing gradient problem: vanilla RNNs forget long-range info
• LSTMs use gates (forget, input, output) to control memory
• LSTMs can remember hundreds of time steps
• Modern alternative: Transformers (Chapter 22)
```
