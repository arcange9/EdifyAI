# Chapter 7 — Mathematics for AI: Linear Algebra

## Learning Objectives

By the end of this chapter, you will:
- Understand vectors, matrices, and their operations
- Know what dot products and matrix multiplication mean visually
- Understand eigenvalues and dimensionality reduction
- See how linear algebra powers every neural network

---

## Why This Matters

Linear algebra is the language of AI. Every model — from simple regression to GPT — is built on vectors and matrices. When a neural network makes a prediction, it's literally doing matrix multiplication. When an embedding model represents text, it stores it as a vector. When you compute similarity for RAG, you're doing dot products.

```
┌──────────────────────────────────────────────────────┐
│  WHERE LINEAR ALGEBRA APPEARS IN AI                   │
│                                                      │
│  Neural Network Layer:  output = W @ x + b           │
│  Embeddings:            text → vector of numbers     │
│  PCA (dim. reduction):  eigenvalues & eigenvectors   │
│  Similarity (RAG):     dot product of vectors         │
│  Gradient computation:  vector operations             │
│  Image processing:     image = 3D matrix of pixels   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Concept Explanation

### Vectors — The Building Block

A vector is a list of numbers. That's it. But this simple concept is incredibly powerful.

```
┌──────────────────────────────────────────────────────────┐
│  WHAT A VECTOR REPRESENTS IN AI                           │
│                                                          │
│  A student's features:   [age, study_hours, attendance]  │
│                          = [18, 5.5, 0.9]                │
│                                                          │
│  An image pixel:         [R, G, B]                       │
│                          = [255, 128, 0]                 │
│                                                          │
│  A word embedding:       [0.2, -0.5, 0.8, ...] (300 dim) │
│                                                          │
│  A model's prediction:   [prob_class_0, prob_class_1]    │
│                          = [0.3, 0.7]                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Vectors Visually

```
  y
  ↑
 3│       ●  v = [2, 3]     (2 right, 3 up)
  │      /
 2│     /
  │    /
 1│   /
  │  /
  0●────────→ x
  0  1  2  3

  Vector v = [2, 3]
  Length (magnitude) = √(2² + 3²) = √13 ≈ 3.61
  Direction = from origin to the point (2, 3)
```

### Matrices — Organized Data

A matrix is a 2D grid of numbers. In AI, matrices represent:
- Datasets (rows = samples, columns = features)
- Model weights (the "knowledge" of a neural network)
- Transformations (rotate, scale, project data)

```
┌──────────────────────────────────────────────────────────┐
│  MATRIX AS A DATASET                                      │
│                                                          │
│         Age  Hours  Score                                 │
│  ┌─────────────────────────┐                            │
│  │  18    5.5    85        │  ← Student 1               │
│  │  20    7.0    92        │  ← Student 2               │
│  │  19    3.0    65        │  ← Student 3               │
│  │  21    8.0    95        │  ← Student 4               │
│  └─────────────────────────┘                            │
│                                                          │
│  Shape: (4, 3) → 4 samples, 3 features                   │
│  Each ROW is a sample (vector)                           │
│  Each COLUMN is a feature (vector)                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Matrix as Model Weights

```
┌──────────────────────────────────────────────────────────┐
│  MATRIX AS NEURAL NETWORK WEIGHTS                         │
│                                                          │
│  Input vector x = [x₁, x₂, x₃]                          │
│                                                          │
│         ┌─────── W ───────┐                              │
│         │  w₁₁  w₁₂  w₁₃  │                              │
│  W =    │  w₂₁  w₂₂  w₂₃  │   ← Weight matrix            │
│         │  w₃₁  w₃₂  w₃₃  │                              │
│         │  w₄₁  w₄₂  w₄₃  │                              │
│         └──────────────────┘                              │
│                                                          │
│  Output = W @ x = [y₁, y₂, y₃, y₄]                      │
│                                                          │
│  Each wᵢⱼ determines how much input xⱼ                  │
│  contributes to output yᵢ                                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Mathematics — Step by Step

### 1. Vector Operations

**Addition:** Add corresponding elements.
```
  [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
```
Visual: place vectors tip-to-tail, the sum is the resultant.

**Scalar multiplication:** Multiply every element.
```
  3 × [1, 2, 3] = [3, 6, 9]
```

**Dot product:** Multiply corresponding elements and sum.
```
  [1, 2, 3] · [4, 5, 6] = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

🔧 **UNDER THE HOOD:** The dot product measures *similarity*. If two vectors point in the same direction, their dot product is large and positive. If they're perpendicular, it's zero. If opposite, it's negative. This is why dot products are used for semantic similarity in RAG.

### 2. Matrix Multiplication

```
  A (m×n) @ B (n×p) = C (m×p)

  ┌─── A ───┐     ┌─── B ───┐     ┌─── C ───┐
  │ a₁₁ a₁₂ │     │ b₁₁ b₁₂ │     │ c₁₁ c₁₂ │
  │ a₂₁ a₂₂ │  ×  │ b₂₁ b₂₂ │  =  │ c₂₁ c₂₂ │
  └─────────┘     └─────────┘     └─────────┘
   (2×2)            (2×2)           (2×2)

  c₁₁ = a₁₁×b₁₁ + a₁₂×b₂₁
  c₁₂ = a₁₁×b₁₂ + a₁₂×b₂₂
  c₂₁ = a₂₁×b₁₁ + a₂₂×b₂₁
  c₂₂ = a₂₁×b₁₂ + a₂₂×b₂₂
```

📌 **IMPORTANT:** Matrix multiplication is NOT commutative: A @ B ≠ B @ A (usually). The inner dimensions must match: (m×n) @ (n×p) = (m×p). If they don't match, you can't multiply.

### 3. Transpose

Flipping rows and columns: A^T turns rows into columns and vice versa.
```
  A = [1, 2, 3]    →    A^T = [1]
                           [2]
                           [3]

  B = [1, 2]       →    B^T = [1, 3]
      [3, 4]               [2, 4]
```

### 4. Eigenvalues and Eigenvectors (Simplified)

```
┌──────────────────────────────────────────────────────────┐
│  EIGENVECTORS & EIGENVALUES                              │
│                                                          │
│  When you multiply a matrix A by vector v:              │
│                                                          │
│    A @ v = λv                                            │
│                                                          │
│  If v doesn't change direction (only stretches/shrinks),│
│  then v is an eigenvector and λ is its eigenvalue.      │
│                                                          │
│  Why it matters:                                         │
│  • PCA finds eigenvectors of the covariance matrix      │
│  • These eigenvectors are the "most important"          │
│    directions in your data                                │
│  • Eigenvalues tell you how important each direction is │
│                                                          │
│  In deep learning, eigenvalues help us understand:      │
│  • How fast gradient descent converges                  │
│  • Whether a network will be stable during training     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Code — Linear Algebra in Practice

```python
"""
Chapter 7 — Linear Algebra for AI
All examples use NumPy (pip install numpy matplotlib)
"""
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. Vector Operations ──────────────────────
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print("=== Vector Operations ===")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {v1 + v2}")           # [5, 7, 9]
print(f"v1 * 3  = {v1 * 3}")            # [3, 6, 9]
print(f"v1 · v2 = {np.dot(v1, v2)}")    # 32

# Magnitude (length) of a vector
magnitude = np.sqrt(np.sum(v1 ** 2))   # √(1+4+9) = √14
print(f"|v1| = {magnitude:.4f}")

# Unit vector (direction only, length = 1)
unit_v1 = v1 / magnitude
print(f"Unit v1 = {unit_v1}")
print(f"|unit_v1| = {np.sqrt(np.sum(unit_v1**2)):.4f}")

# ─── 2. Cosine Similarity (used in RAG!) ──────
# This is THE formula for semantic search
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors.
    Returns value between -1 and 1.
    1 = identical direction, 0 = perpendicular, -1 = opposite.
    """
    dot = np.dot(a, b)
    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))
    return dot / (norm_a * norm_b)

# Simulate word embeddings (simplified)
king = np.array([0.9, 0.8, 0.1, 0.05])
queen = np.array([0.85, 0.9, 0.1, 0.02])
apple = np.array([0.1, 0.05, 0.9, 0.8])

print(f"\n=== Cosine Similarity ===")
print(f"cos(king, queen) = {cosine_similarity(king, queen):.4f}")  # High
print(f"cos(king, apple) = {cosine_similarity(king, apple):.4f}")  # Low
print("→ king and queen are similar; king and apple are not")

# ─── 3. Matrix Multiplication ──────────────────
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(f"\n=== Matrix Multiplication ===")
print(f"A @ B =\n{A @ B}")
print(f"B @ A =\n{B @ A}")  # Different result!
print("→ Order matters in matrix multiplication")

# ─── 4. Neural Network Layer ──────────────────
print(f"\n=== Neural Network Forward Pass ===")
# Input: 3 features
x = np.array([0.5, -0.3, 0.8])

# Weights: 3 inputs → 2 neurons (shape: 3x2)
W = np.array([[0.1, -0.2],
              [0.3,  0.4],
              [-0.5, 0.6]])

# Bias: 2 values (one per neuron)
b = np.array([0.1, -0.1])

# Forward pass: z = W^T @ x + b  (or x @ W + b depending on convention)
# Here we use: z = x @ W + b
z = x @ W + b

print(f"Input x: {x}")
print(f"Weights W:\n{W}")
print(f"Bias b: {b}")
print(f"Output z = x @ W + b = {z}")

# Apply activation (ReLU: max(0, z))
a = np.maximum(0, z)
print(f"After ReLU: {a}")

# ─── 5. Eigenvalues & PCA ─────────────────────
print(f"\n=== Eigenvalues & PCA ===")

# Create 2D data with correlation
np.random.seed(42)
data = np.random.randn(200, 2)
data[:, 1] = data[:, 0] * 0.8 + data[:, 1] * 0.3  # Add correlation

# Compute covariance matrix
cov_matrix = np.cov(data.T)
print(f"Covariance matrix:\n{cov_matrix}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# The eigenvector with the largest eigenvalue is the
# "most important direction" in the data (PCA component 1)
idx = np.argsort(eigenvalues)[::-1]  # Sort descending
print(f"Most important direction (PC1): {eigenvectors[:, idx[0]]}")
print(f"Explained variance ratio: {eigenvalues[idx] / eigenvalues.sum()}")

# ─── 6. Visualize PCA ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original data with principal components
axes[0].scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
origin = [0, 0]
for i, (val, vec) in enumerate(zip(eigenvalues[idx], eigenvectors.T[idx])):
    scale = val * 3
    axes[0].quiver(*origin, vec[0]*scale, vec[1]*scale,
                   color=['red', 'blue'][i], scale=1, scale_units='xy',
                   label=f'PC{i+1} (λ={val:.2f})')
axes[0].set_title('Data with Principal Components')
axes[0].legend()
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Projected onto first component (1D)
projected = data @ eigenvectors[:, idx[0]]
axes[1].hist(projected, bins=30, color='green', alpha=0.7)
axes[1].set_title('Projection onto PC1 (Dimensionality Reduction)')
axes[1].set_xlabel('Value along PC1')

plt.tight_layout()
plt.savefig('ch7_pca.png', dpi=150)
plt.show()
print("\n✓ PCA visualization saved")

print("\n" + "=" * 50)
print("✓ Chapter 7 Linear Algebra complete!")
print("=" * 50)
```

---

## Real-World Example

```
┌──────────────────────────────────────────────────────────────┐
│  LINEAR ALGEBRA IN REAL AI SYSTEMS                           │
│                                                              │
│  Google Search:                                              │
│    Pages → vectors → matrix                                  │
│    Search query → vector → dot product with page vectors    │
│    Highest dot products = most relevant results              │
│                                                              │
│  Neural Networks:                                            │
│    Every layer = matrix multiplication + activation          │
│    GPT-4 has ~1 trillion parameters stored in matrices      │
│                                                              │
│  RAG Systems:                                                │
│    Documents → embedding vectors                             │
│    Query → embedding vector                                  │
│    cosine_similarity(query, documents) → top results        │
│                                                              │
│  Image Compression:                                           │
│    PCA (eigenvalues) reduces image data to fewer dimensions │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Dimension mismatch.* Always check shapes before multiplying: `(m,n) @ (n,p) = (m,p)`.
2. *Confusing element-wise (*) with matrix multiplication (@).* They're completely different.
3. *Forgetting to normalize before PCA.* PCA is sensitive to scale. Standardize first.
4. *Not understanding what dot product means.* It's not just a formula — it measures similarity.

---

## Engineering Perspective

🧠 **THINK LIKE AN ENGINEER:**

```
When you see a matrix in AI code, ask:
1. What do the rows represent? (samples? neurons? features?)
2. What do the columns represent?
3. What operation is being performed and why?
4. What is the shape before and after the operation?

When choosing embeddings for RAG:
1. What dimension? (768, 1536, 3072 — higher = more detail, slower)
2. How is similarity computed? (cosine, dot product, Euclidean)
3. What was the model trained on? (general text, code, scientific papers)
```

---

## Mini-Project — Vector Similarity Engine

Build a simple semantic search engine using cosine similarity:

```python
"""
Mini-Project: Vector Similarity Engine
Demonstrates how semantic search works using cosine similarity.
"""
import numpy as np

# Simulate document embeddings (in real life, use an embedding model)
documents = [
    {"id": 1, "text": "How to train a neural network", "vec": [0.9, 0.8, 0.1]},
    {"id": 2, "text": "Introduction to Python programming", "vec": [0.1, 0.2, 0.9]},
    {"id": 3, "text": "Deep learning with PyTorch", "vec": [0.85, 0.75, 0.2]},
    {"id": 4, "text": "Pandas data manipulation guide", "vec": [0.15, 0.25, 0.85]},
    {"id": 5, "text": "Understanding transformers architecture", "vec": [0.8, 0.7, 0.15]},
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_vec, documents, top_k=3):
    """Search for most similar documents to query vector."""
    results = []
    for doc in documents:
        sim = cosine_similarity(query_vec, doc["vec"])
        results.append((doc["text"], sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Simulate a query: "neural networks" → [0.88, 0.78, 0.12]
query = np.array([0.88, 0.78, 0.12])
results = search(query, documents)

print("=== Semantic Search Results ===")
print(f"Query: 'neural networks'\n")
for text, score in results:
    print(f"  [{score:.4f}] {text}")
```

---

## Exercises

### Easy
1. Calculate the dot product of [1, 2] and [3, 4] by hand, then verify with NumPy.
2. Create a 3x3 identity matrix and multiply it by any vector. What happens?
3. Compute the magnitude (length) of [3, 4]. (Hint: Pythagorean theorem!)

### Medium
4. Implement cosine similarity from scratch (no NumPy, only pure Python). Compare results with NumPy version.
5. Given two matrices A (2x3) and B (3x2), compute A@B and B@A. Are they the same shape?
6. Create a 2D dataset, compute its covariance matrix, and find eigenvalues.

### Advanced
7. Implement PCA from scratch using eigenvalue decomposition. Reduce 5D data to 2D.
8. Build a function that performs a complete neural network forward pass with multiple layers.

---

## Chapter Summary

```
┌──────────────────────────────────────────────────────────┐
│                   CHAPTER 7 KEY TAKEAWAYS                │
│                                                          │
│  • Vectors = lists of numbers (features, embeddings)     │
│  • Matrices = 2D grids (datasets, weights)               │
│  • Dot product = similarity measure (core of RAG)        │
│  • Matrix multiplication = neural network layers         │
│  • Transpose = flip rows and columns                     │
│  • Eigenvalues = "importance" of data directions         │
│  • PCA uses eigenvalues for dimensionality reduction     │
│  • Always check shapes before operations!                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Interview Questions

1. "Explain what a dot product represents geometrically."
2. "How does cosine similarity work and why is it used in RAG?"
3. "What are eigenvalues and why do they matter in ML?"
4. "Why does matrix multiplication require matching inner dimensions?"

---

## Knowledge Check

1. What is the result of [2, 3] · [4, 1]?
2. Can you multiply a (3×4) matrix by a (4×2) matrix? What's the result shape?
3. What does a cosine similarity of 0 mean vs 1?
4. What is the purpose of PCA?
5. In the equation y = Wx + b, what are W, x, and b in a neural network?
