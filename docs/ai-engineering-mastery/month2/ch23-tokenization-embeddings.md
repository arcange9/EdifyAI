# Chapter 23 — Tokenization & Embeddings

## Learning Objectives

- Understand tokenization: how text becomes numbers
- Know BPE, WordPiece, and SentencePiece tokenizers
- Understand word embeddings and contextual embeddings
- Visualize embeddings to see semantic relationships

---

## Why This Matters

Before an LLM can process text, the text must be converted to numbers. Tokenization is the bridge between human language and neural networks. The quality of tokenization directly affects model quality. Embeddings are how models *understand meaning* — similar concepts get similar vectors.

---

## Concept Explanation

### Tokenization — Words to Numbers

```
┌──────────────────────────────────────────────────────────────┐
│  TOKENIZATION METHODS                                       │
│                                                              │
│  WORD-LEVEL:                                                 │
│  "I love AI" → ["I", "love", "AI"] → [45, 892, 1234]       │
│  Problem: can't handle new words, huge vocabulary           │
│                                                              │
│  CHARACTER-LEVEL:                                            │
│  "I love AI" → ["I", " ", "l", "o", "v", "e", ...]         │
│  Problem: very long sequences, no word meaning              │
│                                                              │
│  SUBWORD (BPE — Byte Pair Encoding):                        │
│  "unbelievable" → ["un", "believ", "able"] → [34, 567, 89]│
│  Best of both: handles new words + manageable vocab         │
│  Used by: GPT, GPT-2, GPT-3, GPT-4                         │
│                                                              │
│  WORDPIECE:                                                  │
│  Similar to BPE but uses ## prefix for continuations       │
│  "playing" → ["play", "##ing"]                              │
│  Used by: BERT, DistilBERT                                   │
│                                                              │
│  SENTENCEPIECE:                                              │
│  Language-agnostic, treats text as raw stream               │
│  Used by: T5, LLaMA, Mistral                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Embeddings — Numbers to Meaning

```
┌──────────────────────────────────────────────────────────────┐
│  WORD EMBEDDINGS                                             │
│                                                              │
│  Each token gets a vector (e.g., 768 dimensions)             │
│  Similar words → similar vectors                             │
│                                                              │
│  king  = [0.9,  0.8,  0.1,  ...]                            │
│  queen = [0.85, 0.9,  0.1,  ...]                            │
│  apple = [0.1,  0.05, 0.9,  ...]                            │
│                                                              │
│  The classic example:                                        │
│  king - man + woman ≈ queen                                 │
│  paris - france + germany ≈ berlin                          │
│                                                              │
│  This works because the embedding space captures            │
│  semantic relationships. The model learns that             │
│  "king" and "queen" differ mainly in gender.                │
│                                                              │
│  Contextual Embeddings (Transformer era):                   │
│  "bank" in "river bank" and "bank account"                  │
│  get DIFFERENT embeddings because the surrounding          │
│  words change the meaning.                                   │
│                                                              │
│  Static (word2vec): same vector for "bank" always            │
│  Contextual (BERT/GPT): different vector based on context   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Tokenization & Embeddings

```python
"""
Chapter 23 — Tokenization & Embeddings
pip install torch transformers matplotlib scikit-learn
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ─── 1. Simple BPE-like tokenizer ──────────────
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def build_vocab(self, texts, min_freq=1):
        word_counts = {}
        for text in texts:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        self.vocab = {"<pad>": 0, "<unk>": 1}
        for word, count in sorted(word_counts.items()):
            if count >= min_freq:
                self.vocab[word] = len(self.vocab)
        self.vocab_size = len(self.vocab)

    def encode(self, text, max_len=20):
        tokens = text.lower().split()[:max_len]
        ids = [self.vocab.get(t, 1) for t in tokens]
        while len(ids) < max_len:
            ids.append(0)
        return ids

    def decode(self, ids):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return " ".join(inv_vocab.get(i, "<unk>") for i in ids if i > 0)

tokenizer = SimpleTokenizer()
texts = ["the cat sat on the mat", "the dog ran in the park",
         "cats and dogs are pets", "I love machine learning"]
tokenizer.build_vocab(texts)
print(f"Vocabulary: {tokenizer.vocab_size} words")

sample = "the cat sat on the mat"
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)
print(f"Text: '{sample}'")
print(f"Tokens: {sample.split()}")
print(f"IDs: {encoded}")
print(f"Decoded: '{decoded}'")

# ─── 2. Embedding Layer ────────────────────────
embed_dim = 32
embedding = nn.Embedding(tokenizer.vocab_size, embed_dim)

# Get embeddings for a sentence
token_ids = torch.LongTensor([tokenizer.encode("the cat sat on the mat")])
embedded = embedding(token_ids)
print(f"\nToken IDs shape: {token_ids.shape}")
print(f"Embedded shape: {embedded.shape}")  # (1, 20, 32)

# ─── 3. Word similarity via embeddings ──────────
# Get embeddings for specific words
words = ["cat", "dog", "the", "mat", "park", "learning"]
word_ids = [tokenizer.vocab.get(w, 1) for w in words]
word_embeddings = embedding(torch.LongTensor(word_ids)).detach().numpy()

# Compute cosine similarity
sim_matrix = cosine_similarity(word_embeddings)

print("\n=== Word Similarity Matrix ===")
print(f"{'':12s}", end="")
for w in words:
    print(f"{w:12s}", end="")
print()
for i, w in enumerate(words):
    print(f"{w:12s}", end="")
    for j in range(len(words)):
        print(f"{sim_matrix[i,j]:12.3f}", end="")
    print()

# ─── 4. Visualize embeddings with PCA ──────────
pca = PCA(n_components=2)
reduced = pca.fit_transform(word_embeddings)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(reduced[:, 0], reduced[:, 1], c='steelblue', s=100)
for i, word in enumerate(words):
    ax.annotate(word, (reduced[i, 0], reduced[i, 1]),
                fontsize=12, fontweight='bold',
                xytext=(5, 5), textcoords='offset points')
ax.set_title('Word Embeddings (PCA 2D Projection)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch23_embeddings.png', dpi=150)
plt.show()

# ─── 5. Using Hugging Face tokenizers ───────────
print("\n=== Hugging Face Tokenizers ===")
try:
    from transformers import AutoTokenizer
    tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2")
    text = "Artificial intelligence is amazing!"

    tokens = tokenizer_gpt.encode(text)
    decoded = tokenizer_gpt.decode(tokens)
    token_strs = tokenizer_gpt.convert_ids_to_tokens(tokens)

    print(f"Text: '{text}'")
    print(f"Tokens: {token_strs}")
    print(f"IDs: {tokens}")
    print(f"Decoded: '{decoded}'")
except Exception as e:
    print(f"  (Install transformers to run: {e})")

print("\n✓ Chapter 23 complete!")
```

---

## Chapter Summary

```
• Tokenization: text → tokens → integer IDs
• BPE (subword): handles new words, used by GPT
• WordPiece: used by BERT
• Embeddings: token IDs → dense vectors (768+ dimensions)
• Similar words → similar vectors (cosine similarity)
• Contextual embeddings (Transformers): same word, different
  embedding based on surrounding context
• king - man + woman ≈ queen (semantic arithmetic)
```

---

## Interview Questions

1. "What's the difference between word-level and subword tokenization?"
2. "How do embeddings capture semantic meaning?"
3. "What's the difference between static and contextual embeddings?"
4. "Why is BPE better than word-level tokenization?"
