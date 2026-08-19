# Chapter 4 — NumPy: Arrays and Vectorized Computing

## Learning Objectives

By the end of this chapter, you will:
- Understand what NumPy is and why it's the backbone of AI computing
- Create, index, and manipulate arrays
- Use vectorized operations (no loops needed)
- Apply broadcasting and linear algebra operations
- Understand how NumPy connects to every other AI library

---

## Why This Matters

Every AI library — scikit-learn, PyTorch, TensorFlow, Hugging Face — is built on top of NumPy or uses its conventions. When you train a model, the data flows through arrays. When you compute gradients, it's NumPy operations under the hood.

📌 **IMPORTANT:** NumPy is the single most important library in AI. If you master only one tool from Month 1, make it NumPy.

---

## Concept Explanation

### Why NumPy Exists

```
┌──────────────────────────────────────────────────────────┐
│  PYTHON LISTS vs NUMPY ARRAYS                            │
│                                                          │
│  Python List:                                            │
│  [3, 5, 2, 8, 1]                                        │
│   │   │   │   │   │                                     │
│   ▼   ▼   ▼   ▼   ▼                                     │
│  Each element is a Python object with overhead.           │
│  Operations use Python loops (slow).                     │
│                                                          │
│  NumPy Array:                                            │
│  [3, 5, 2, 8, 1]                                        │
│  ──────────────                                         │
│  Stored as contiguous memory (like C arrays).            │
│  Operations use optimized C/Fortran code (fast).         │
│                                                          │
│  Speed difference: 10x to 100x for numerical operations  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### What "Vectorized" Means

```
REGULAR PYTHON (loop-based):
┌──────────────────────────────────────┐
│  a = [1, 2, 3, 4, 5]                 │
│  b = [10, 20, 30, 40, 50]           │
│  c = []                              │
│  for i in range(len(a)):             │
│      c.append(a[i] + b[i])          │
│                                      │
│  Python processes one element at     │
│  a time. Slow for large data.        │
└──────────────────────────────────────┘

NUMPY (vectorized):
┌──────────────────────────────────────┐
│  a = np.array([1, 2, 3, 4, 5])       │
│  b = np.array([10, 20, 30, 40, 50]) │
│  c = a + b  # [11, 22, 33, 44, 55]  │
│                                      │
│  NumPy processes ALL elements at     │
│  once in optimized C code. Fast.     │
└──────────────────────────────────────┘
```

---

## Visual Explanation — Dimensions in NumPy

```
┌─────────────────────────────────────────────────────────────┐
│  NUMPY ARRAY DIMENSIONS                                     │
│                                                             │
│  0-D (scalar):  42                                          │
│                                                             │
│  1-D (vector):  [1, 2, 3, 4, 5]    shape: (5,)              │
│  ───────────────────────────                              │
│  Used for: feature vectors, time series                    │
│                                                             │
│  2-D (matrix):  [[1, 2, 3],       shape: (2, 3)            │
│                  [4, 5, 6]]                                │
│  ┌─────┬─────┬─────┐                                       │
│  │  1  │  2  │  3  │                                       │
│  ├─────┼─────┼─────┤                                       │
│  │  4  │  5  │  6  │                                       │
│  └─────┴─────┴─────┘                                       │
│  Used for: datasets (rows=samples, cols=features)           │
│                                                             │
│  3-D (tensor):  [[[1,2],[3,4]],    shape: (2, 2, 2)        │
│                  [[5,6],[7,8]]]                           │
│  Used for: images (height × width × channels), batches     │
│                                                             │
│  4-D (batch of images):  shape: (batch, H, W, channels)    │
│  Used for: deep learning input batches                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — NumPy for AI

```python
"""
Chapter 4 — NumPy: Arrays and Vectorized Computing

pip install numpy
"""

import numpy as np

# ════════════════════════════════════════════════════════
# SECTION 1: CREATING ARRAYS
# ════════════════════════════════════════════════════════

# From a list
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Shape: {arr.shape}")
print(f"Type: {arr.dtype}")  # int64

# From a list of lists (2D)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"\nMatrix shape: {matrix.shape}")  # (2, 3)

# Special arrays — common in AI initialization
zeros = np.zeros(5)               # [0, 0, 0, 0, 0]
ones = np.ones((3, 3))            # 3x3 matrix of 1s
identity = np.eye(3)               # Identity matrix
random = np.random.rand(4)         # 4 random numbers [0, 1)
random_int = np.random.randint(0, 10, size=(3, 3))  # 3x3 random ints
range_arr = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)   # [0, 0.25, 0.5, 0.75, 1.0]

print(f"\nZeros: {zeros}")
print(f"Ones:\n{ones}")
print(f"Identity:\n{identity}")
print(f"Random: {random}")
print(f"Arange: {range_arr}")
print(f"Linspace: {linspace}")

# ─── AI-specific array creation ─────────────────
# Initialize model weights (small random values)
weights = np.random.randn(3, 4) * 0.01  # Normal distribution, scaled
print(f"\nModel weights (3x4):\n{weights}")


# ════════════════════════════════════════════════════════
# SECTION 2: INDEXING AND SLICING
# ════════════════════════════════════════════════════════

data = np.array([[10, 20, 30, 40],
                 [50, 60, 70, 80],
                 [90, 100, 110, 120]])

print(f"\nData shape: {data.shape}")  # (3, 4)
print(f"First row: {data[0]}")
print(f"Last row: {data[-1]}")
print(f"First column: {data[:, 0]}")
print(f"Element [1, 2]: {data[1, 2]}")

# Slicing: [start:stop:step]
print(f"Rows 0-1, cols 1-3: \n{data[0:2, 1:3]}")
print(f"Every other column: \n{data[:, ::2]}")

# ─── Boolean indexing (FILTERING) ───────────────
# Critical for AI: filter data based on conditions
scores = np.array([45, 78, 92, 33, 88, 67, 55, 95])
passed = scores >= 60
print(f"\nScores: {scores}")
print(f"Passed mask: {passed}")
print(f"Passing scores: {scores[passed]}")
print(f"Failing scores: {scores[~passed]}")
print(f"Number passed: {np.sum(passed)}")


# ════════════════════════════════════════════════════════
# SECTION 3: VECTORIZED OPERATIONS
# ════════════════════════════════════════════════════════

# Element-wise math — NO LOOPS NEEDED
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

print(f"\na + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a ** 2 = {a ** 2}")
print(f"np.sqrt(a) = {np.sqrt(a)}")
print(f"np.exp(a) = {np.exp(a)}")

# Aggregation functions
print(f"\nSum: {a.sum()}")
print(f"Mean: {a.mean()}")
print(f"Std: {a.std()}")
print(f"Min: {a.min()}")
print(f"Max: {a.max()}")
print(f"Argmax: {a.argmax()}")  # Index of max value

# ─── AI example: softmax ─────────────────────────
# Softmax converts raw scores to probabilities
def softmax(x):
    """Convert logits to probabilities. Used in classification."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

logits = np.array([2.0, 1.0, 0.1, -0.5])
probabilities = softmax(logits)
print(f"\nLogits: {logits}")
print(f"Softmax probabilities: {probabilities}")
print(f"Sum (should be 1.0): {probabilities.sum():.4f}")


# ════════════════════════════════════════════════════════
# SECTION 4: BROADCASTING
# ════════════════════════════════════════════════════════

# Broadcasting lets you operate on arrays of different shapes
# NumPy "stretches" the smaller array to match

# Example 1: Add scalar to array
arr = np.array([1, 2, 3, 4, 5])
print(f"\narr + 10 = {arr + 10}")  # [11, 12, 13, 14, 15]

# Example 2: Add 1D array to 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_vector = np.array([10, 20, 30])
print(f"\nMatrix + row_vector:\n{matrix + row_vector}")

# Example 3: Normalize columns (AI preprocessing)
# Subtract mean and divide by std for each feature column
data = np.random.randn(100, 3)  # 100 samples, 3 features
mean = data.mean(axis=0)  # Mean of each column
std = data.std(axis=0)   # Std of each column
normalized = (data - mean) / std  # Broadcasting!

print(f"\nOriginal mean: {mean}")
print(f"Normalized mean: {normalized.mean(axis=0)}")  # ~0
print(f"Normalized std: {normalized.std(axis=0)}")    # ~1


# ════════════════════════════════════════════════════════
# SECTION 5: LINEAR ALGEBRA OPERATIONS
# ════════════════════════════════════════════════════════

# Dot product — THE most important operation in neural networks
# Used in every layer: output = weights · inputs + bias

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"\nDot product: {dot_product}")

# Matrix multiplication (used in neural network forward pass)
# In Python 3.5+, you can use @ operator
weights = np.array([[0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]])
inputs = np.array([1.0, 2.0, 3.0])

# weights @ inputs = [0.1*1 + 0.2*2 + 0.3*3, 0.4*1 + 0.5*2 + 0.6*3]
output = weights @ inputs
print(f"Weights @ inputs = {output}")

# ─── Simulated neural network forward pass ──────
print("\n--- Simulated Neural Network Forward Pass ---")

# Layer 1: 3 inputs → 4 neurons
W1 = np.random.randn(3, 4) * 0.1
b1 = np.zeros(4)
x = np.array([0.5, -0.3, 0.8])  # Input vector

# Forward pass through layer 1
z1 = x @ W1 + b1          # Weighted sum
a1 = np.maximum(0, z1)    # ReLU activation function

print(f"Input: {x}")
print(f"Layer 1 (z1): {z1}")
print(f"Layer 1 (a1 after ReLU): {a1}")

# Layer 2: 4 neurons → 2 outputs
W2 = np.random.randn(4, 2) * 0.1
b2 = np.zeros(2)
z2 = a1 @ W2 + b2
output = softmax(z2)

print(f"Layer 2 (z2): {z2}")
print(f"Output (probabilities): {output}")
print(f"Predicted class: {output.argmax()}")


# ════════════════════════════════════════════════════════
# SECTION 6: RESHAPING AND MANIPULATING
# ════════════════════════════════════════════════════════

# Reshaping is essential when preparing data for models
arr = np.arange(12)
print(f"\nOriginal: {arr}")

# Reshape to 3x4
reshaped = arr.reshape(3, 4)
print(f"Reshaped (3x4):\n{reshaped}")

# Flatten back to 1D
flat = reshaped.flatten()
print(f"Flattened: {flat}")

# Transpose (swap rows and columns)
transposed = reshaped.T
print(f"Transposed (4x3):\n{transposed}")

# Concatenation (combining datasets)
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

vertical = np.vstack([a, b])   # Stack vertically (add rows)
horizontal = np.hstack([a, b])  # Stack horizontally (add columns)
print(f"\nVertical stack:\n{vertical}")
print(f"Horizontal stack:\n{horizontal}")


# ════════════════════════════════════════════════════════
# SECTION 7: PRACTICAL AI EXAMPLES
# ════════════════════════════════════════════════════════

# ─── Example 1: Calculate MSE (Mean Squared Error) ──
predictions = np.array([3.1, 4.9, 2.8, 7.2, 5.0])
actual = np.array([3.0, 5.0, 3.0, 7.0, 5.0])
mse = np.mean((predictions - actual) ** 2)
print(f"\nMSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")

# ─── Example 2: Mini-batch gradient simulation ─────
# Simulate how training data is split into batches
full_data = np.random.randn(1000, 5)  # 1000 samples, 5 features
batch_size = 32
n_batches = len(full_data) // batch_size

print(f"\nDataset: {full_data.shape}")
print(f"Batch size: {batch_size}")
print(f"Number of batches: {n_batches}")

# Get one batch
batch_0 = full_data[0:batch_size]
print(f"Batch 0 shape: {batch_0.shape}")

# Shuffle and create batches (like real training)
indices = np.random.permutation(len(full_data))
for i in range(n_batches):
    batch = full_data[indices[i*batch_size:(i+1)*batch_size]]
    # Process batch here...
print(f"Last batch shape: {batch.shape}")

# ─── Example 3: Distance calculation (used in k-NN) ──
# Calculate Euclidean distance between two points
def euclidean_distance(a, b):
    """Calculate Euclidean distance between two vectors."""
    return np.sqrt(np.sum((a - b) ** 2))

point_a = np.array([1.0, 2.0, 3.0])
point_b = np.array([4.0, 5.0, 6.0])
dist = euclidean_distance(point_a, point_b)
print(f"\nDistance between {point_a} and {point_b}: {dist:.4f}")

# Calculate distance from one point to many points
query = np.array([0.0, 0.0])
dataset = np.array([[1, 1], [2, 2], [0.5, 0.5], [5, 5]])
distances = np.sqrt(np.sum((dataset - query) ** 2, axis=1))
nearest = distances.argmin()
print(f"Nearest point to {query}: {dataset[nearest]} (distance: {distances[nearest]:.4f})")

print("\n" + "=" * 50)
print("✓ Chapter 4 NumPy complete!")
print("=" * 50)
```

---

## Real-World Example

```
┌──────────────────────────────────────────────────────────────┐
│  NUMPY IN PRODUCTION                                          │
│                                                              │
│  PyTorch tensors → GPU version of NumPy arrays               │
│  scikit-learn → Input/output is NumPy arrays                │
│  Hugging Face → Tokenizers output NumPy arrays              │
│  Embeddings → Vectors stored as NumPy arrays                │
│  Image processing → Pixels as NumPy arrays                  │
│                                                              │
│  Every AI library uses NumPy conventions:                   │
│  - shape tells you the data dimensions                       │
│  - dtype tells you the data type                            │
│  - @ operator for matrix multiplication                      │
│  - Broadcasting for batch operations                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Using Python loops on NumPy arrays.* This defeats the purpose. Use vectorized operations.
2. *Modifying views accidentally.* `arr2 = arr1` creates a reference, not a copy. Use `arr2 = arr1.copy()`.
3. *Integer division.* In Python 3, `/` gives float, `//` gives int. Be aware when working with NumPy.
4. *Shape mismatches.* Always check `.shape` before operations. Broadcasting can hide errors.

---

## Engineering Perspective

🧠 **THINK LIKE AN ENGINEER:**

```
When working with arrays in production AI:
1. Always check .shape before operations
2. Use .copy() when you need an independent array
3. Prefer vectorized operations over loops (100x speedup)
4. Set np.random.seed() for reproducibility
5. Use float32 instead of float64 to save memory
6. Understand broadcasting rules to avoid silent bugs
```

---

## Mini-Project — Matrix Calculator

Build a tool that can:
1. Create two random matrices of user-specified dimensions
2. Add, subtract, and multiply them
3. Calculate the transpose
4. Check if multiplication is possible (inner dimensions must match)
5. Calculate the determinant (for square matrices)

```python
"""
Mini-Project: Matrix Calculator
"""
import numpy as np

class MatrixCalculator:
    def __init__(self):
        self.matrices = {}

    def create(self, name, rows, cols):
        """Create a random matrix."""
        self.matrices[name] = np.random.randn(rows, cols)
        print(f"Created {name}: {self.matrices[name].shape}")
        print(self.matrices[name])

    def add(self, name_a, name_b):
        """Add two matrices (must be same shape)."""
        if self.matrices[name_a].shape != self.matrices[name_b].shape:
            print("Error: Shapes don't match for addition")
            return None
        result = self.matrices[name_a] + self.matrices[name_b]
        print(f"{name_a} + {name_b}:\n{result}")
        return result

    def multiply(self, name_a, name_b):
        """Multiply two matrices (a's cols must equal b's rows)."""
        if self.matrices[name_a].shape[1] != self.matrices[name_b].shape[0]:
            print(f"Error: Cannot multiply {self.matrices[name_a].shape} "
                  f"× {self.matrices[name_b].shape}")
            return None
        result = self.matrices[name_a] @ self.matrices[name_b]
        print(f"{name_a} @ {name_b}:\n{result}")
        return result

    def transpose(self, name):
        """Return the transpose of a matrix."""
        result = self.matrices[name].T
        print(f"{name} transposed:\n{result}")
        return result

    def determinant(self, name):
        """Calculate determinant (square matrices only)."""
        m = self.matrices[name]
        if m.shape[0] != m.shape[1]:
            print(f"Error: {name} is not square ({m.shape})")
            return None
        det = np.linalg.det(m)
        print(f"det({name}) = {det:.4f}")
        return det

# Run the calculator
calc = MatrixCalculator()
calc.create("A", 3, 3)
calc.create("B", 3, 2)
calc.add("A", "B")          # Will error — different shapes
calc.multiply("A", "B")    # A (3x3) @ B (3x2) = (3x2)
calc.transpose("B")        # B.T is (2x3)
calc.determinant("A")      # Works — A is 3x3
```

---

## Exercises

### Easy
1. Create a 5x5 array of random integers (0-100). Find the maximum value in each row.
2. Create two arrays and compute their dot product.
3. Use boolean indexing to find all values greater than 50 in a random array.

### Medium
4. Implement standardization (z-score normalization) using NumPy broadcasting.
5. Write a function that computes the Euclidean distance between two arrays without using a loop.
6. Create a 3D array representing 10 images of 28x28 pixels (like MNIST). Calculate the mean image.

### Advanced
7. Implement matrix multiplication from scratch using only loops, then compare performance with `@` operator.
8. Implement a simple k-NN classifier using only NumPy (no scikit-learn).

---

## Chapter Summary

```
┌──────────────────────────────────────────────────────────┐
│                   CHAPTER 4 KEY TAKEAWAYS                │
│                                                          │
│  • NumPy arrays are the foundation of AI computing       │
│  • Vectorized operations replace loops (100x faster)    │
│  • Broadcasting enables operations on different shapes    │
│  • Dot product (@) is the core of neural networks        │
│  • Reshape/transpose for data preparation                │
│  • Boolean indexing for filtering data                   │
│  • Always check .shape before operations                 │
│  • Use .copy() when you need an independent array        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Interview Questions

1. "What is broadcasting in NumPy and how does it work?"
2. "What's the difference between `np.dot()`, `@`, and `np.multiply()`?"
3. "How would you normalize a dataset using NumPy?"
4. "Why is NumPy faster than pure Python for numerical operations?"

---

## Knowledge Check

1. What is the shape of `np.array([[1,2],[3,4],[5,6]])`?
2. What does `arr[:, 0]` do?
3. How do you create a 4x4 identity matrix?
4. What is the output of `np.array([1,2,3]) + np.array([[1],[2],[3]])`?
5. Why do we subtract `np.max(x)` before applying `np.exp()` in softmax?
