# Chapter 26 — Hugging Face & Open-Source Models

## Learning Objectives

- Use the Hugging Face ecosystem (transformers, datasets, models)
- Run open-source models locally and via API
- Set up Ollama for free local model inference
- Understand model sizes and their tradeoffs

---

## Why This Matters

Not every AI project needs a paid API. Hugging Face hosts 500,000+ open-source models. Ollama lets you run LLMs on your laptop for free. For a student or a startup, open-source models are the way to build AI without API costs.

---

## Concept Explanation

### Hugging Face Ecosystem

```
┌──────────────────────────────────────────────────────────────┐
│  HUGGING FACE ECOSYSTEM                                      │
│                                                              │
│  transformers ── Run models (BERT, GPT, LLaMA, etc.)       │
│  datasets ────── Access 100,000+ datasets                   │
│  tokenizers ──── Fast tokenization                          │
│  accelerate ──── Distributed training                        │
│  Hub ──────────── Model repository (500k+ models)            │
│  Spaces ───────── Host demos for free                       │
│  PEFT ────────── Parameter-efficient fine-tuning           │
│                                                              │
│  Key model families:                                         │
│  • BERT/RoBERTa — encoder models (understanding)            │
│  • GPT/Llama — decoder models (generation)                  │
│  • T5/BART — encoder-decoder (translation, summarization)   │
│  • Mistral/Mixtral — efficient open models                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Open-Source Model Sizes

```
┌─────────────────────────────────────────────────────────────┐
│  MODEL SIZES AND REQUIREMENTS                              │
│                                                             │
│  Model          Params  RAM needed  Speed    Quality      │
│  ─────────────────────────────────────────────────────────  │
│  TinyLlama      1.1B    ~2 GB        Fast     Basic        │
│  Phi-2          2.7B    ~4 GB        Fast     Good         │
│  Llama 3.2      3B     ~6 GB        Fast     Good         │
│  Mistral 7B     7B     ~8 GB        Medium   Very good    │
│  Llama 3.1 8B   8B     ~10 GB       Medium   Very good    │
│  Mixtral 8x7B   47B    ~24 GB       Slow     Excellent    │
│  Llama 3.1 70B  70B    ~40 GB       Slow     Excellent    │
│                                                             │
│  Quantized (4-bit): divide RAM by ~2.5x                     │
│  7B model at 4-bit → ~3 GB RAM → runs on most laptops!     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — Using Hugging Face + Ollama

```python
"""
Chapter 26 — Hugging Face & Open-Source Models
pip install transformers torch
"""

# ─── 1. Using Hugging Face transformers ────────
print("=== Hugging Face Transformers ===\n")

from transformers import pipeline

# Sentiment analysis (small model, runs on CPU)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier([
    "I love this course!",
    "This is terrible and confusing."
])
print("Sentiment analysis:")
for text, result in zip(["I love this course!", "This is terrible."], results):
    print(f"  '{text}' → {result['label']} ({result['score']:.3f})")

# Text generation (small GPT-2 model)
generator = pipeline("text-generation", model="gpt2", max_new_tokens=30)
output = generator("Artificial intelligence is", do_sample=True, temperature=0.7)
print(f"\nText generation:\n  {output[0]['generated_text']}")

# ─── 2. Using Ollama for local models ───────────
print("\n=== Ollama (Local Models) ===\n")
print("""To use Ollama:
1. Install: https://ollama.com
2. Run: ollama pull llama3.2
3. Run: ollama run llama3.2

In Python (pip install ollama):
  from ollama import chat
  response = chat(model='llama3.2', messages=[
      {'role': 'user', 'content': 'What is AI engineering?'}
  ])
  print(response['message']['content'])

Free, private, no API key needed.
""")

# ─── 3. Comparing models ────────────────────────
print("=== Model Comparison Framework ===\n")

models_comparison = {
    "distilbert-base-uncased": {"size": "268MB", "task": "classification", "quality": "Good"},
    "gpt2": {"size": "548MB", "task": "generation", "quality": "Basic"},
    "llama-3.2-3b": {"size": "6GB", "task": "generation", "quality": "Very good"},
    "mistral-7b": {"size": "8GB", "task": "generation", "quality": "Excellent"},
}

for name, info in models_comparison.items():
    print(f"  {name}: {info['size']} | {info['task']} | {info['quality']}")

print("\n✓ Chapter 26 complete!")
```

---

## Chapter Summary

```
• Hugging Face: 500k+ open models, datasets, tokenizers
• Pipeline API: simplest way to run models
• Ollama: run LLMs locally for free, no API key
• Model size tradeoff: bigger = better but slower + more RAM
• Quantization (4-bit): run large models on small machines
• Open-source alternatives to paid APIs for production
```

---

## Interview Questions

1. "When would you choose an open-source model over a paid API?"
2. "What is quantization and why is it useful?"
3. "How would you run an LLM on a laptop with 8GB RAM?"
4. "What's the Hugging Face ecosystem and why is it important?"
