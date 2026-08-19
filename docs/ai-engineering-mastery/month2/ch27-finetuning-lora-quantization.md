# Chapter 27 — Fine-Tuning, LoRA & Quantization

## Learning Objectives

- Understand fine-tuning concepts (full, PEFT, LoRA)
- Know when to fine-tune vs use prompt engineering
- Understand quantization (4-bit, 8-bit) and memory savings
- See how to make models smaller and faster for production

---

## Why This Matters

Pre-trained models are general-purpose. Fine-tuning adapts them to your specific domain — medical text, legal documents, code. LoRA makes fine-tuning affordable. Quantization makes inference possible on smaller hardware. These are essential AI Engineering skills.

---

## Concept Explanation

### Fine-Tuning Approaches

```
┌──────────────────────────────────────────────────────────────┐
│  FINE-TUNING APPROACHES                                     │
│                                                              │
│  FULL FINE-TUNING:                                           │
│  Update ALL parameters of the model                          │
│  Pros: best quality                                          │
│  Cons: expensive (needs lots of GPU memory)                 │
│  Use when: you have resources and need maximum quality       │
│                                                              │
│  PEFT (Parameter-Efficient Fine-Tuning):                    │
│  Freeze most weights, only update a small set               │
│  Pros: cheap, fast, works on limited hardware                │
│  Cons: slightly lower quality than full fine-tuning         │
│  Use when: limited GPU, want to save storage                 │
│                                                              │
│  LoRA (Low-Rank Adaptation):                                 │
│  Add small trainable matrices beside frozen weights          │
│  Pros: VERY efficient, only ~1% of parameters               │
│  Cons: not as good as full fine-tuning for complex tasks    │
│  Use when: most common PEFT method in 2026                   │
│                                                              │
│  WHEN TO FINE-TUNE:                                          │
│  • Domain-specific language (medical, legal, code)           │
│  • Specific output format needed                              │
│  • Performance gap that prompt engineering can't close      │
│                                                              │
│  WHEN NOT TO FINE-TUNE:                                      │
│  • General-purpose tasks (prompt engineering is enough)     │
│  • Small dataset (<100 examples)                             │
│  • Limited compute resources                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### LoRA Explained

```
┌──────────────────────────────────────────────────────────────┐
│  LoRA (LOW-RANK ADAPTATION)                                  │
│                                                              │
│  Original weight matrix W (frozen):                          │
│  ┌──────────────────────┐                                   │
│  │                      │  ← e.g., 4096 × 4096 = 16M params │
│  │    W (frozen)        │                                   │
│  │                      │                                   │
│  └──────────────────────┘                                   │
│                                                              │
│  LoRA adds: ΔW = A × B                                       │
│  ┌──────┐   ┌──────────┐                                    │
│  │  A   │ × │    B     │  ← A: 4096×8, B: 8×4096           │
│  └──────┘   └──────────┘  ← Only 65K params (0.4% of W!)    │
│                                                              │
│  New weight = W + A×B                                        │
│  W is frozen (not updated during training)                   │
│  Only A and B are trained (tiny!)                             │
│                                                              │
│  Why it works: the "important" changes in weights             │
│  have low rank — they can be captured by small matrices      │
│                                                              │
│  Storage: save only A and B (65K) instead of W (16M)        │
│  → One base model + many tiny LoRA adapters                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Quantization

```
┌──────────────────────────────────────────────────────────────┐
│  QUANTIZATION: SMALLER MODELS                               │
│                                                              │
│  FP32 (float32): 32 bits per parameter                       │
│  → 7B model = 28 GB                                         │
│                                                              │
│  FP16 (float16): 16 bits per parameter                       │
│  → 7B model = 14 GB                                         │
│                                                              │
│  INT8 (8-bit): 8 bits per parameter                          │
│  → 7B model = 7 GB                                          │
│  → Minimal quality loss                                      │
│                                                              │
│  INT4 (4-bit): 4 bits per parameter                          │
│  → 7B model = 3.5 GB                                        │
│  → Small quality loss, runs on laptops!                     │
│                                                              │
│  The math:                                                   │
│  Original range: [-1.5, 2.3] (float)                        │
│  4-bit: map to 16 levels (0-15)                             │
│  Store: integer 0-15 + scale + zero-point                   │
│  Reconstruct: original_value ≈ int × scale + zero_point     │
│                                                              │
│  Memory savings:                                             │
│  7B FP32 → 7B INT4 = 28GB → 3.5GB (8x smaller!)           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Fine-Tuning Concept Demo

```python
"""
Chapter 27 — Fine-Tuning, LoRA, Quantization
Conceptual demo (actual fine-tuning requires GPU + large downloads)
pip install torch numpy
"""
import numpy as np

# ─── 1. LoRA Concept Demo ───────────────────────
print("=== LoRA Concept Demo ===\n")

# Simulate a large weight matrix
W = np.random.randn(4096, 4096)  # 16M params (frozen)

# LoRA: two small matrices
rank = 8  # Very small rank
A = np.random.randn(4096, rank) * 0.01  # 32K params (trainable)
B = np.zeros((rank, 4096))                # 32K params (trainable)

# Original parameter count
original_params = W.size  # 16,777,216
lora_params = A.size + B.size  # 65,536
ratio = lora_params / original_params * 100

print(f"  Original W: {W.shape} = {original_params:,} params")
print(f"  LoRA A: {A.shape} = {A.size:,} params")
print(f"  LoRA B: {B.shape} = {B.size:,} params")
print(f"  LoRA total: {lora_params:,} params ({ratio:.2f}% of original)")
print(f"  Savings: {100-ratio:.2f}%")

# Simulate fine-tuning: update only A and B
# ΔW = A @ B
delta_W = A @ B
new_W = W + delta_W  # Effective weight after fine-tuning
print(f"  ΔW shape: {delta_W.shape}")
print(f"  Effective W = frozen W + trained A×B")

# ─── 2. Quantization Concept Demo ──────────────
print("\n=== Quantization Concept Demo ===\n")

# Simulate weight quantization
weights_fp32 = np.random.randn(1000).astype(np.float32)

# 8-bit quantization
w_min, w_max = weights_fp32.min(), weights_fp32.max()
scale_8bit = (w_max - w_min) / 255
zero_point_8bit = -w_min / scale_8bit
weights_int8 = np.round(weights_fp32 / scale_8bit + zero_point_8bit).astype(np.int8)

# 4-bit quantization
scale_4bit = (w_max - w_min) / 15
zero_point_4bit = -w_min / scale_4bit
weights_int4 = np.round(weights_fp32 / scale_4bit + zero_point_4bit).clip(0, 15).astype(np.int8)

# Dequantize (reconstruct)
reconstructed_8bit = (weights_int8.astype(np.float32) - zero_point_8bit) * scale_8bit
reconstructed_4bit = (weights_int4.astype(np.float32) - zero_point_4bit) * scale_4bit

# Error analysis
mse_8bit = np.mean((weights_fp32 - reconstructed_8bit) ** 2)
mse_4bit = np.mean((weights_fp32 - reconstructed_4bit) ** 2)

print(f"  FP32 size: {weights_fp32.nbytes} bytes")
print(f"  INT8 size: {weights_int8.nbytes} bytes (4x smaller)")
print(f"  INT4 effective: {weights_fp32.nbytes / 8} bytes (8x smaller)")
print(f"  8-bit MSE: {mse_8bit:.8f}")
print(f"  4-bit MSE: {mse_4bit:.8f}")

# ─── 3. Memory estimation ───────────────────────
print("\n=== Memory Estimation ===\n")
model_sizes = [1, 3, 7, 13, 70]  # billions of params

for size in model_sizes:
    fp32 = size * 4  # GB
    fp16 = size * 2
    int8 = size * 1
    int4 = size * 0.5
    print(f"  {size}B model: FP32={fp32:.0f}GB, FP16={fp16:.0f}GB, "
          f"INT8={int8:.0f}GB, INT4={int4:.1f}GB")

print("\n  A 7B model at INT4 = 3.5GB → fits on most laptops!")
print("  A 7B model at FP32 = 28GB → needs a powerful GPU!")

print("\n✓ Chapter 27 complete!")
```

---

## Chapter Summary

```
• Full fine-tuning: best quality, expensive
• LoRA: train 0.4% of params, save 99.6% storage
• Quantization: FP32→INT4 = 8x memory savings
• When to fine-tune: domain-specific, specific format
• When NOT to fine-tune: general tasks, small data
• LoRA + Quantization (QLoRA) = fine-tune large models on single GPU
```

---

## Interview Questions

1. "What is LoRA and why is it useful?"
2. "How does quantization work and what are the tradeoffs?"
3. "When would you fine-tune a model vs use prompt engineering?"
4. "What is QLoRA?"
