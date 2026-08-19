# Chapter 17 — Training Neural Networks: Backpropagation & Gradient Descent

## Learning Objectives

- Understand backpropagation step by step (the chain rule in action)
- Know loss functions: MSE for regression, cross-entropy for classification
- Implement a complete training loop from scratch
- Understand optimizers: SGD, momentum, Adam

---

## Why This Matters

Backpropagation is THE algorithm that makes deep learning possible. Without it, training neural networks would be impossibly slow. Every framework (PyTorch, TensorFlow) uses backpropagation under the hood. Understanding it means you understand how AI actually *learns*.

---

## Concept Explanation

### The Training Loop

```
┌──────────────────────────────────────────────────────────────┐
│  NEURAL NETWORK TRAINING LOOP                               │
│                                                              │
│  FOR each epoch:                                            │
│    FOR each batch of data:                                  │
│      1. FORWARD PASS: compute predictions                   │
│         x → layer1 → layer2 → ... → predictions              │
│                                                              │
│      2. COMPUTE LOSS: how wrong are we?                     │
│         loss = loss_function(predictions, targets)           │
│                                                              │
│      3. BACKWARD PASS: compute gradients (backpropagation)  │
│         loss ← gradients flow backward ← ... ← layer2 ← layer1│
│                                                              │
│      4. UPDATE WEIGHTS: adjust to reduce loss               │
│         weights -= learning_rate × gradients                  │
│                                                              │
│  Repeat until loss is small enough                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Loss Functions

```
┌─────────────────────────────────────────────────────────────┐
│  LOSS FUNCTIONS                                            │
│                                                             │
│  MSE (Mean Squared Error) — for regression:               │
│  L = (1/n) × Σ(y_true - y_pred)²                         │
│  Penalizes large errors heavily                             │
│                                                             │
│  Cross-Entropy — for classification:                      │
│  Binary:   L = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]            │
│  Multi:    L = -Σ y_true · log(softmax(y_pred))           │
│                                                             │
│  Why cross-entropy?                                        │
│  • Works naturally with softmax output                    │
│  • Gradient is simpler than MSE + sigmoid                 │
│  • Heavily penalizes confident wrong predictions           │
│                                                             │
│  When prediction = 0.01 but true = 1:                     │
│    MSE loss = (1-0.01)² = 0.98                            │
│    CE loss  = -log(0.01) = 4.6  ← much bigger penalty!   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Backpropagation — The Chain Rule Applied

```
┌──────────────────────────────────────────────────────────────┐
│  BACKPROPAGATION (SIMPLIFIED)                               │
│                                                              │
│  Network: x → [Layer1] → a1 → [Layer2] → a2 → loss          │
│                                                              │
│  Forward:                                                   │
│    z1 = W1·x + b1   →  a1 = ReLU(z1)                       │
│    z2 = W2·a1 + b2  →  a2 = softmax(z2)                    │
│    L = cross_entropy(a2, y)                                 │
│                                                              │
│  Backward (compute gradients using chain rule):             │
│    dL/dz2 = a2 - y           (output layer gradient)        │
│    dL/dW2 = a1^T · dL/dz2   (weight gradient for layer 2)   │
│    dL/db2 = sum(dL/dz2)     (bias gradient for layer 2)     │
│                                                              │
│    dL/da1 = W2^T · dL/dz2   (pass gradient to previous layer)│
│    dL/dz1 = dL/da1 · ReLU'(z1) (ReLU derivative: 1 if z>0) │
│    dL/dW1 = x^T · dL/dz1    (weight gradient for layer 1)   │
│    dL/db1 = sum(dL/dz1)     (bias gradient for layer 1)     │
│                                                              │
│  Update:                                                    │
│    W -= lr × dL/dW                                          │
│    b -= lr × dL/db                                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Complete Training Loop from Scratch

```python
"""
Chapter 17 — Backpropagation & Training
Building a trainable neural network from scratch in NumPy.
pip install numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# ════════════════════════════════════════════════════════

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# ════════════════════════════════════════════════════════
# LOSS FUNCTIONS AND THEIR DERIVATIVES
# ════════════════════════════════════════════════════════

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_gradient(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_pred, y_true):
    """y_pred: softmax probabilities, y_true: one-hot encoded."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_gradient(y_pred, y_true):
    """Gradient of cross-entropy w.r.t. logits (before softmax)."""
    return (y_pred - y_true) / y_true.shape[0]


# ════════════════════════════════════════════════════════
# TRAINABLE NEURAL NETWORK
# ════════════════════════════════════════════════════════

class NeuralNetwork:
    """A trainable multi-layer neural network using backpropagation."""

    def __init__(self, layer_sizes, lr=0.01):
        self.lr = lr
        self.weights = []
        self.biases = []

        # Initialize weights with He initialization
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass. Returns activations for each layer."""
        activations = [X]
        pre_activations = []

        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            pre_activations.append(z)

            if i == len(self.weights) - 1:  # Output layer
                a = softmax(z) if self.weights[-1].shape[1] > 1 else sigmoid(z)
            else:  # Hidden layers
                a = relu(z)

            activations.append(a)

        return activations, pre_activations

    def backward(self, activations, pre_activations, y_true):
        """Backward pass. Returns gradients for weights and biases."""
        n = y_true.shape[0]
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        # Output layer gradient (softmax + cross-entropy)
        delta = (activations[-1] - y_true) / n

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            grad_weights[i] = activations[i].T @ delta
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:  # Don't need gradient for input
                delta = (delta @ self.weights[i].T) * relu_derivative(pre_activations[i-1])

        return grad_weights, grad_biases

    def train_step(self, X, y):
        """One training step: forward + backward + update."""
        # Forward pass
        activations, pre_activations = self.forward(X)

        # Compute loss
        if y.shape[1] > 1:  # Multi-class
            loss = cross_entropy_loss(activations[-1], y)
        else:  # Binary
            loss = mse_loss(activations[-1], y)

        # Backward pass
        grad_weights, grad_biases = self.backward(activations, pre_activations, y)

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grad_weights[i]
            self.biases[i] -= self.lr * grad_biases[i]

        return loss

    def predict(self, X):
        """Return predicted class indices."""
        activations, _ = self.forward(X)
        return activations[-1].argmax(axis=1)

    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """Train for multiple epochs."""
        history = {'loss': [], 'accuracy': []}
        n = len(X)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n)
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_indices = indices[start:start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            # Compute accuracy
            if y.shape[1] > 1:
                preds = self.predict(X)
                acc = np.mean(preds == y.argmax(axis=1))
            else:
                preds = (activations[-1] > 0.5).astype(int) if 'activations' in dir() else 0
                acc = 0

            history['loss'].append(avg_loss)
            history['accuracy'].append(acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={acc:.4f}")

        return history


# ════════════════════════════════════════════════════════
# TRAIN ON CLASSIFICATION DATA
# ════════════════════════════════════════════════════════

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_classification(n_samples=500, n_features=10, n_classes=3,
                           n_informative=8, random_state=42)

# Convert to one-hot encoding
n_classes = 3
y_onehot = np.zeros((len(y), n_classes))
y_onehot[np.arange(len(y)), y] = 1

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train network
np.random.seed(42)
net = NeuralNetwork([10, 16, 12, 3], lr=0.05)
print("=== Training Neural Network ===")
print(f"Architecture: 10 → 16 → 12 → 3")
print(f"Parameters: {sum(W.size + b.size for W, b in zip(net.weights, net.biases))}")

history = net.train(X_train, y_train, epochs=100, batch_size=32)

# Evaluate
train_acc = np.mean(net.predict(X_train) == y_train.argmax(axis=1))
test_acc = np.mean(net.predict(X_test) == y_test.argmax(axis=1))
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# Visualize training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['loss'], 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['accuracy'], 'g-', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('ch17_training.png', dpi=150)
plt.show()
print("✓ Training visualization saved")

print("\n" + "=" * 50)
print("✓ Chapter 17 complete!")
print("=" * 50)
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Forgetting to zero gradients between batches.* Accumulates wrong gradients. In PyTorch: `optimizer.zero_grad()`.
2. *Learning rate too high.* Loss explodes (goes to NaN). Always check for NaN during training.
3. *Not shuffling data.* Same order every epoch → model learns order, not patterns.
4. *Wrong loss function.* Using MSE for classification. Use cross-entropy.
5. *Forgetting to scale inputs.* Unscaled inputs → large activations → unstable training.

---

## Engineering Perspective

🧠 **THINK LIKE AN ENGINEER:**

```
Training debugging checklist:
1. Loss decreasing? → If not, check lr (too high) or architecture
2. Loss = NaN? → lr too high, or numerical instability (add clipping)
3. Training accuracy stuck? → model too simple (add neurons/layers)
4. Test accuracy much lower? → overfitting (add regularization, dropout)
5. Both train and test low? → underfitting (more complex model or better features)
6. Training very slow? → check batch size, try Adam optimizer, scale inputs
```

---

## Mini-Project — MNIST from Scratch

Train the network on MNIST digits (simplified):

```python
"""
Mini-Project: MNIST Classification from Scratch
"""
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()
X = digits.data / 16.0  # Normalize pixels (0-1)
y = digits.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42)

np.random.seed(42)
net = NeuralNetwork([64, 32, 16, 10], lr=0.1)
print("Training on MNIST (8x8 digits)...")
history = net.train(X_train, y_train, epochs=100, batch_size=64)

train_acc = np.mean(net.predict(X_train) == y_train.argmax(axis=1))
test_acc = np.mean(net.predict(X_test) == y_test.argmax(axis=1))
print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
```

---

## Exercises

### Easy
1. What's the difference between MSE and cross-entropy loss?
2. Why do we shuffle training data each epoch?
3. What happens if the learning rate is too high?

### Medium
4. Implement momentum in the NeuralNetwork class. Compare convergence with and without.
5. Train the network on XOR data. How many hidden neurons do you need?
6. Plot the loss curve. What does a plateau followed by sudden drop suggest?

### Advanced
7. Implement Adam optimizer in the NeuralNetwork class. Test convergence speed.
8. Add L2 regularization to the loss function. Show it reduces overfitting.

---

## Chapter Summary

```
• Training loop: forward → loss → backward → update
• Backpropagation = chain rule applied layer by layer
• MSE for regression, cross-entropy for classification
• Gradients tell us HOW to adjust weights to reduce loss
• Optimizers: SGD (simple), Momentum (faster), Adam (adaptive)
• Always: shuffle data, scale inputs, monitor for NaN
```

---

## Interview Questions

1. "Explain backpropagation in your own words."
2. "Why is cross-entropy used instead of MSE for classification?"
3. "What's the role of the learning rate in training?"
4. "How does Adam differ from standard gradient descent?"

---

## Knowledge Check

1. What are the 4 steps in the training loop?
2. Why is the chain rule needed for backpropagation?
3. When would you use MSE vs cross-entropy?
4. What does a loss of NaN indicate?
5. Why does shuffling data improve training?
