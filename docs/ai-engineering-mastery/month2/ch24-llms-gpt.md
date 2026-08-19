# Chapter 24 — Large Language Models: How GPT Works

## Learning Objectives

- Understand the full GPT architecture from tokens to text generation
- Know how LLMs are trained (pre-training + fine-tuning)
- Understand decoding strategies (temperature, top-k, top-p)
- See why LLMs "hallucinate" and how to reduce it

---

## Why This Matters

LLMs are the most impactful AI technology of our time. ChatGPT, Claude, Gemini, LLaMA — they all share the same fundamental architecture. As an AI Engineer, you'll work with LLMs daily. Understanding how they work makes you a better practitioner.

---

## Concept Explanation

### How GPT Generates Text

```
┌──────────────────────────────────────────────────────────────┐
│  GPT TEXT GENERATION (AUTOREGRESSIVE)                       │
│                                                              │
│  Step 1: "The cat sat on the"                                │
│         ↓ Transformer                                        │
│         ↓ Predict next token                                 │
│  Step 2: "The cat sat on the mat"                           │
│         ↓ Transformer                                        │
│         ↓ Predict next token                                 │
│  Step 3: "The cat sat on the mat because"                   │
│         ↓ Transformer                                        │
│         ↓ Predict next token                                 │
│  Step 4: "The cat sat on the mat because it"               │
│         ...continues...                                       │
│                                                              │
│  KEY INSIGHT:                                                │
│  GPT doesn't "understand" — it predicts the most likely     │
│  next token given the previous tokens.                      │
│  It does this ONE TOKEN AT A TIME.                          │
│                                                              │
│  The "intelligence" comes from:                              │
│  1. Billions of parameters (learned patterns)               │
│  2. Trained on trillions of tokens (massive data)           │
│  3. Many Transformer layers (deep processing)               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Training Stages

```
┌──────────────────────────────────────────────────────────────┐
│  LLM TRAINING PIPELINE                                       │
│                                                              │
│  STAGE 1: PRE-TRAINING (unsupervised)                       │
│  ┌─────────────────────────────────────────┐               │
│  │  Input: trillions of tokens of text      │               │
│  │  Task: predict next token                 │               │
│  │  Cost: millions of GPU-hours             │               │
│  │  Result: base model (can generate text)  │               │
│  └─────────────────────────────────────────┘               │
│  The model learns: grammar, facts, reasoning patterns       │
│  from massive internet text data.                            │
│                                                              │
│  STAGE 2: FINE-TUNING (supervised)                          │
│  ┌─────────────────────────────────────────┐               │
│  │  Input: high-quality Q&A pairs            │               │
│  │  Task: respond helpfully and safely       │               │
│  │  Cost: much less than pre-training         │               │
│  │  Result: instruction-following model      │               │
│  └─────────────────────────────────────────┘               │
│  The model learns to follow instructions and format          │
│  responses properly.                                        │
│                                                              │
│  STAGE 3: RLHF (Reinforcement Learning from Human Feedback) │
│  ┌─────────────────────────────────────────┐               │
│  │  Input: human preference rankings         │               │
│  │  Task: generate responses humans prefer  │               │
│  │  Result: aligned model (ChatGPT-like)    │               │
│  └─────────────────────────────────────────┘               │
│  The model learns to be helpful, harmless, and honest.     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Decoding Strategies

```
┌─────────────────────────────────────────────────────────────┐
│  DECODING STRATEGIES                                       │
│                                                             │
│  GREEDY: always pick the most likely token                 │
│  "The cat sat on the ___" → always "mat" (deterministic)   │
│  Problem: boring, repetitive, no creativity               │
│                                                             │
│  TEMPERATURE (T): controls randomness                      │
│  T = 0.0: always pick best → deterministic, safe          │
│  T = 0.7: balanced → some variety, mostly coherent         │
│  T = 1.0: full distribution → creative but may drift       │
│  T = 2.0: very random → often nonsensical                  │
│                                                             │
│  Formula: logits = logits / temperature                    │
│  Then softmax → sample from distribution                   │
│                                                             │
│  TOP-K: only sample from top K tokens                       │
│  K = 1: same as greedy                                    │
│  K = 50: consider 50 most likely tokens                   │
│  Removes the long tail of unlikely tokens                  │
│                                                             │
│  TOP-P (nucleus sampling): consider tokens until their     │
│  cumulative probability reaches P                          │
│  P = 0.9: consider tokens covering 90% of probability     │
│  Adaptive: more tokens when distribution is flat,          │
│  fewer when one token dominates                            │
│                                                             │
│  BEST PRACTICE: temperature=0.7, top_p=0.9                 │
│  For FACTUAL: temperature=0.0 (deterministic)             │
│  For CREATIVE: temperature=0.9, top_p=0.95                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — Simulated LLM Generation

```python
"""
Chapter 24 — How GPT Works
Simulates LLM text generation with different decoding strategies.
pip install numpy matplotlib
"""
import numpy as np

def softmax(x, temperature=1.0):
    """Temperature-scaled softmax."""
    x = x / temperature
    e = np.exp(x - np.max(x))
    return e / e.sum()

def generate_step(logits, temperature=1.0, top_k=None, top_p=None):
    """Generate next token from logits with decoding strategy."""
    probs = softmax(logits, temperature)

    if top_k is not None:
        # Keep only top k probabilities
        if top_k < len(probs):
            threshold = np.sort(probs)[-top_k]
            probs = np.where(probs >= threshold, probs, 0)
            probs = probs / probs.sum()

    if top_p is not None:
        # Nucleus sampling
        sorted_idx = np.argsort(probs)[::-1]
        cumulative = 0
        for idx in sorted_idx:
            cumulative += probs[idx]
            if cumulative >= top_p:
                break
        keep = set(sorted_idx[:np.where(sorted_idx == idx)[0][0] + 1])
        probs = np.where(np.isin(np.arange(len(probs)), list(keep)), probs, 0)
        probs = probs / probs.sum()

    return np.random.choice(len(probs), p=probs)

# Simulate token probabilities for "The cat sat on the ___"
vocab = ["mat", "floor", "ground", "couch", "bed", "chair", "table", "roof"]
logits = np.array([5.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.5, 0.1])

print("=== Decoding Strategies ===\n")

# Greedy
print("Greedy (T=0):", vocab[np.argmax(softmax(logits, 0.01))])

# Different temperatures
for temp in [0.0, 0.5, 0.7, 1.0, 1.5]:
    if temp == 0:
        token = vocab[np.argmax(logits)]
    else:
        token = vocab[generate_step(logits, temperature=temp)]
    print(f"Temperature {temp}: {token}")

# Top-K
print("\nTop-K sampling:")
for k in [1, 3, 5, 8]:
    samples = [vocab[generate_step(logits, top_k=k)] for _ in range(20)]
    print(f"  K={k}: {samples[:5]}")

# Show probability distributions
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, temp in zip(axes, [0.5, 0.7, 1.0, 2.0]):
    probs = softmax(logits, temp)
    ax.bar(vocab, probs, color='steelblue')
    ax.set_title(f'Temperature = {temp}')
    ax.set_ylabel('Probability')
    ax.set_xticklabels(vocab, rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ch24_decoding.png', dpi=150)
plt.show()

print("\n✓ Chapter 24 complete!")
```

---

## Chapter Summary

```
• GPT = autoregressive: predict next token, one at a time
• Pre-training: learn from massive text (predict next token)
• Fine-tuning: learn to follow instructions
• RLHF: align with human preferences
• Temperature: 0 = deterministic, 1 = creative
• Top-K: limit to K most likely tokens
• Top-P: limit to tokens covering P% of probability
• Hallucination: model generates plausible but false text
  because it's predicting, not reasoning
```

---

## Interview Questions

1. "How does GPT generate text?"
2. "What's the difference between pre-training and fine-tuning?"
3. "How does temperature affect text generation?"
4. "Why do LLMs hallucinate and how can you reduce it?"

---

## Knowledge Check

1. What does "autoregressive" mean in the context of LLMs?
2. What are the 3 stages of LLM training?
3. When would you set temperature to 0?
4. What's the difference between top-k and top-p?
5. Why can't LLMs be trusted for factual accuracy without RAG?
