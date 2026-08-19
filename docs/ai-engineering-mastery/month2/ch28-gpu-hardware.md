# Chapter 28 — GPUs and AI Hardware

## Learning Objectives

- Understand why GPUs are essential for deep learning
- Know the AI hardware landscape (GPU, TPU, NPU)
- Calculate GPU memory requirements for models
- Choose the right hardware for AI projects

---

## Why This Matters

Training and running AI models requires computational power. Understanding hardware helps you make smart decisions: when to use a GPU, when CPU is enough, how much memory you need, and how to optimize costs in production.

---

## Concept Explanation

### CPU vs GPU for AI

```
┌──────────────────────────────────────────────────────────────┐
│  CPU vs GPU                                                  │
│                                                              │
│  CPU: 4-16 cores, each very fast, handles complex logic       │
│  GPU: thousands of cores, each simple, handles parallel math │
│                                                              │
│  Analogy:                                                    │
│  CPU = 8 professors solving 8 complex problems                │
│  GPU = 5000 students solving 5000 simple additions            │
│                                                              │
│  Neural networks = massive matrix multiplications             │
│  = thousands of simple parallel calculations                 │
│  = PERFECT for GPUs                                          │
│                                                              │
│  Training speed:                                             │
│  CPU:  1 epoch of MNIST = minutes                            │
│  GPU:  1 epoch of MNIST = seconds (100x faster)             │
│                                                              │
│  Inference speed:                                            │
│  CPU:  OK for small models (distilbert, tinyllama)          │
│  GPU:  required for large models (7B+, image generation)    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### GPU Memory Calculation

```
┌──────────────────────────────────────────────────────────────┐
│  HOW MUCH GPU MEMORY DO YOU NEED?                           │
│                                                              │
│  For INFERENCE (running a model):                            │
│  Memory = model_params × bytes_per_param                     │
│                                                              │
│  7B model, FP16:  7B × 2 = 14 GB                           │
│  7B model, INT4:   7B × 0.5 = 3.5 GB                       │
│  70B model, INT4:  70B × 0.5 = 35 GB                      │
│                                                              │
│  Plus overhead for context window and activations:           │
│  KV cache ≈ 2 × n_layers × seq_len × hidden_dim × 2 bytes   │
│                                                              │
│  For TRAINING (fine-tuning):                                 │
│  Memory = model + gradients + optimizer states              │
│  ≈ 3-4× the inference memory                                │
│                                                              │
│  7B model training, FP16: ~42-56 GB                         │
│  7B model QLoRA training: ~10-14 GB                        │
│                                                              │
│  Common GPUs:                                                │
│  GPU Model          VRAM    Can run (inference)              │
│  ──────────────────────────────────────────────              │
│  RTX 3060          12 GB   7B INT4, 3B FP16                 │
│  RTX 4070          12 GB   7B INT4, 3B FP16                 │
│  RTX 4090          24 GB   7B FP16, 13B INT4                │
│  A100              40-80GB  70B INT4, 13B FP16               │
│  H100              80 GB   70B FP16                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Hardware Decision Framework

```
┌──────────────────────────────────────────────────────────────┐
│  CHOOSING HARDWARE                                          │
│                                                              │
│  FOR LEARNING (this book):                                  │
│  → Your laptop CPU is fine for Chapters 1-18                │
│  → Google Colab free GPU for Chapters 19-22                  │
│  → No purchase needed                                       │
│                                                              │
│  FOR SMALL MODELS (<3B params):                             │
│  → CPU works (slow but functional)                          │
│  → Any GPU speeds it up 10x                                 │
│  → 4 GB VRAM minimum                                        │
│                                                              │
│  FOR MEDIUM MODELS (7B):                                    │
│  → 8 GB VRAM with INT4 quantization                         │
│  → 16 GB VRAM for comfortable inference                     │
│  → RTX 3060/4070 or better                                   │
│                                                              │
│  FOR TRAINING/FINE-TUNING:                                  │
│  → 16+ GB VRAM for LoRA on 7B                               │
│  → 40+ GB for full fine-tuning                              │
│  → Cloud GPU (A100, H100) for large models                  │
│                                                              │
│  FOR PRODUCTION:                                            │
│  → Cloud GPUs (AWS, GCP, RunPod) — pay per use              │
│  → CPU inference for small models (cheaper)                  │
│  → Edge devices (Jetson, Coral) for embedded AI            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — GPU Usage and Memory Estimation

```python
"""
Chapter 28 — GPUs and AI Hardware
pip install torch
"""
import torch

# ─── 1. Check available hardware ───────────────
print("=== Hardware Check ===\n")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_memory / 1e9:.1f} GB")
else:
    print("GPU: Not available (using CPU)")
    print("Tip: Google Colab provides free GPUs!")

# ─── 2. Memory estimation tool ─────────────────
print("\n=== Memory Estimation Tool ===\n")

def estimate_memory(n_params_billion, precision="fp16", task="inference"):
    """Estimate GPU memory needed."""
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}
    base = n_params_billion * bytes_per_param[precision]

    if task == "training":
        # Training needs: model + gradients + optimizer states
        multiplier = {"fp32": 12, "fp16": 6, "int8": 3, "int4": 2}
        total = n_params_billion * multiplier[precision]
    elif task == "lora":
        # LoRA: model (frozen) + small trainable + optimizer for trainable
        total = base * 1.3  # ~30% overhead
    else:
        # Inference: model + KV cache overhead
        total = base * 1.15  # ~15% overhead

    return total

models = [
    ("TinyLlama 1.1B", 1.1),
    ("Llama 3.2 3B", 3),
    ("Mistral 7B", 7),
    ("Llama 3.1 70B", 70),
]

print(f"{'Model':<20} {'FP32':>8} {'FP16':>8} {'INT8':>8} {'INT4':>8}")
print("─" * 56)
for name, size in models:
    fp32 = estimate_memory(size, "fp32")
    fp16 = estimate_memory(size, "fp16")
    int8 = estimate_memory(size, "int8")
    int4 = estimate_memory(size, "int4")
    print(f"{name:<20} {fp32:>7.1f}GB {fp16:>7.1f}GB {int8:>7.1f}GB {int4:>7.1f}GB")

# ─── 3. Moving tensors to GPU ──────────────────
print("\n=== GPU Operations ===\n")

if torch.cuda.is_available():
    # Create on CPU
    x = torch.randn(1000, 1000)
    print(f"  CPU tensor: {x.device}")

    # Move to GPU
    x_gpu = x.cuda()
    print(f"  GPU tensor: {x_gpu.device}")

    # GPU operations are much faster
    import time
    start = time.time()
    for _ in range(1000):
        _ = x @ x  # CPU
    cpu_time = time.time() - start

    start = time.time()
    for _ in range(1000):
        _ = x_gpu @ x_gpu  # GPU
    gpu_time = time.time() - start

    print(f"  CPU time: {cpu_time:.3f}s")
    print(f"  GPU time: {gpu_time:.3f}s")
    print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
else:
    print("  (GPU not available — using CPU)")
    print("  Tip: Use Google Colab for free GPU access")

# ─── 4. Google Colab ────────────────────────────
print("""
=== Google Colab (Free GPU) ===

1. Go to https://colab.research.google.com
2. Runtime → Change runtime type → T4 GPU
3. Install: !pip install torch transformers
4. Run your code — get free GPU acceleration!

Colab provides:
  T4 GPU (16 GB VRAM) — free
  A100 GPU (40 GB VRAM) — paid (Colab Pro)
""")

print("✓ Chapter 28 complete! Month 2 finished!")
print("=" * 50)
```

---

## Chapter Summary

```
• GPU: thousands of parallel cores for matrix math
• CPU: fine for learning, slow for training
• GPU: essential for training and large model inference
• Memory = params × bytes_per_param + overhead
• INT4 quantization: 8x memory savings
• Training needs 3-4x more memory than inference
• LoRA reduces training memory significantly
• Google Colab: free GPU for learning
• Cloud GPUs for production (pay per use)
```

---

## Interview Questions

1. "Why are GPUs better than CPUs for deep learning?"
2. "How much memory do you need to run a 7B model?"
3. "What's the difference between inference and training memory?"
4. "How would you run a large model on limited hardware?"

---

## Knowledge Check

1. Why are GPUs better for matrix multiplication than CPUs?
2. How much VRAM does a 7B model need at INT4? At FP16?
3. Why does training need more memory than inference?
4. What is the KV cache and why does it grow with context length?
5. Name two free ways to access GPU compute.
