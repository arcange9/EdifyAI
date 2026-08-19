# Chapter 16 — Neural Networks: The Building Blocks

## Learning Objectives

- Understand what a neuron is and how it processes information
- Know the major activation functions and when to use each
- Understand forward propagation step by step
- Build a neural network layer from scratch in NumPy

---

## Why This Matters

Every modern AI system — from ChatGPT to Tesla's Autopilot — is built on neural networks. Understanding the building block (the neuron) is like understanding the transistor in electronics. Everything else is just scale and arrangement.

---

## Concept Explanation

### The Biological Inspiration

```
┌──────────────────────────────────────────────────────────┐
│  BIOLOGICAL NEURON vs ARTIFICIAL NEURON                   │
│                                                          │
│  BIOLOGICAL:                          ARTIFICIAL:        │
│                                                          │
│  Dendrites (inputs)                   x₁, x₂, x₃ (inputs)│
│      ↓                                    ↓              │
│  Cell body (sums signals)              Σ(wᵢ × xᵢ) + b     │
│      ↓                                    ↓              │
│  Axon (output)                         activation(z)      │
│      ↓                                    ↓              │
│  Synapses (to other neurons)           output to next     │
│                                                          │
│  Real neurons fire or don't.            We use continuous│
│  (binary signal)                        activation funcs. │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### The Perceptron — Simplest Neural Network

```
┌──────────────────────────────────────────────────────────────┐
│  THE PERCEPTRON                                               │
│                                                              │
│    x₁ ──(w₁)──┐                                             │
│    x₂ ──(w₂)──┼──→ [Σ] ──→ [z = w₁x₁ + w₂x₂ + b] ──→ [activation] ──→ output│
│    x₃ ──(w₃)──┘                                             │
│                                                              │
│  1. Multiply each input by its weight: wᵢ × xᵢ             │
│  2. Sum them all up: z = Σ(wᵢ × xᵢ) + b                    │
│  3. Apply activation function: output = f(z)                │
│                                                              │
│  The weights (w) and bias (b) are what the network LEARNS.  │
│  The activation function (f) is chosen by the engineer.       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Activation Functions

```
┌─────────────────────────────────────────────────────────────┐
│  ACTIVATION FUNCTIONS                                       │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ 1│         _________                              │      │
│  │  │        /                                         │      │
│  │  │  ____/  ← ReLU: f(x) = max(0, x)                 │      │
│  │  │ /                                               │      │
│  │ 0│/________________                                │      │
│  │  │      0    x                                     │      │
│  └──────────────────────────────────────────────────┘      │
│  Most common in hidden layers. Simple, fast, works well.  │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ 1│       ___________                               │      │
│  │  │      /                                           │      │
│  │ .5│ ___/  ← Sigmoid: f(x) = 1/(1+e^(-x))          │      │
│  │  │ /                                               │      │
│  │ 0│/________________                                │      │
│  └──────────────────────────────────────────────────┘      │
│  Used in output layer for binary classification (0 to 1).  │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ 1│  ___                                           │      │
│  │  │ /   \   ← Tanh: f(x) = tanh(x)                 │      │
│  │  │/     \                                         │      │
│  │ 0│       \___                                     │      │
│  └──────────────────────────────────────────────────┘      │
│  Like sigmoid but centered at 0. Range [-1, 1].             │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Each output = e^xᵢ / Σ(e^xⱼ)                    │      │
│  │  Softmax: outputs sum to 1 (probabilities)        │      │
│  │  Used in output layer for multi-class               │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  RULES:                                                     │
│  Hidden layers → ReLU (default)                             │
│  Binary output → Sigmoid                                    │
│  Multi-class output → Softmax                               │
│  Regression output → Linear (no activation)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Layer Networks (Forward Propagation)

```
┌──────────────────────────────────────────────────────────────┐
│  NEURAL NETWORK ARCHITECTURE                                 │
│                                                              │
│  INPUT LAYER     HIDDEN LAYER 1    HIDDEN LAYER 2   OUTPUT   │
│  (3 features)    (4 neurons)        (4 neurons)     (2 classes)│
│                                                              │
│  x₁ ●━━━━━━━━━━━━━●  ← w₁₁                          ● y₁     │
│       ╲         ╱   ╲                                   │   │
│        ╲       ╱     ╲                                   │   │
│  x₂ ●━━━━●━━━━●━━━━━━━●━━━━━━━━━━━━━━━━━●━━━━━━━━━● y₂     │
│        ╱       ╲     ╱                                   │   │
│       ╱         ╲   ╱                                      │
│  x₃ ●━━━━━━━━━━━━━●                                      │   │
│                                                              │
│  FORWARD PROPAGATION (data flows left → right):            │
│                                                              │
│  1. Input:  x = [x₁, x₂, x₃]                               │
│  2. Layer 1: z₁ = W₁ @ x + b₁   →   a₁ = ReLU(z₁)          │
│  3. Layer 2: z₂ = W₂ @ a₁ + b₂  →   a₂ = ReLU(z₂)          │
│  4. Output: z₃ = W₃ @ a₂ + b₃   →   ŷ = Softmax(z₃)        │
│                                                              │
│  W₁ shape: (4, 3)  ← 4 neurons, 3 inputs each              │
│  W₂ shape: (4, 4)  ← 4 neurons, 4 inputs each              │
│  W₃ shape: (2, 4)  ← 2 outputs, 4 inputs each              │
│                                                              │
│  Each layer transforms the representation.                  │
│  Deeper layers learn more abstract patterns.                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

🔧 **UNDER THE HOOD:** A neural network is literally a sequence of matrix multiplications and activation functions. That's it. The "intelligence" comes from the *learned weights* — the numbers in those matrices that are adjusted during training to produce correct outputs.

---

## Code — Neural Network from Scratch

```python
"""
Chapter 16 — Neural Networks: Building Blocks
pip install numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════
# SECTION 1: ACTIVATION FUNCTIONS
# ════════════════════════════════════════════════════════

def relu(x):
    """ReLU: max(0, x). Most common for hidden layers."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid: 1/(1+e^(-x)). Output in (0,1). For binary output."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    """Tanh: output in (-1, 1). Centered at 0."""
    return np.tanh(x)

def softmax(x):
    """Softmax: outputs sum to 1. For multi-class output."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: allows small negative values. Prevents 'dying ReLU'."""
    return np.where(x > 0, x, alpha * x)

# Visualize activation functions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
x = np.linspace(-5, 5, 200)

for ax, func, name in zip(axes.flat,
    [(relu, "ReLU"), (sigmoid, "Sigmoid"), (tanh, "Tanh"),
     (softmax, "Softmax"), (leaky_relu, "Leaky ReLU"),
     (lambda v: np.maximum(0, v), "ReLU (zoom)")]):
    if name == "Softmax":
        y = func(np.array([[0.5, 1.5, 2.5, -0.5, 0.0]]))
        ax.bar(range(5), y[0], color='purple')
        ax.set_title(f'{name} (example)')
    elif name == "ReLU (zoom)":
        y = func(x)
        ax.plot(x, y, 'r-', linewidth=2)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(name)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, 5)
    else:
        y = func(x)
        ax.plot(x, y, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch16_activations.png', dpi=150)
plt.show()
print("✓ Activation functions visualized")


# ════════════════════════════════════════════════════════
# SECTION 2: SINGLE NEURON (PERCEPTRON)
# ════════════════════════════════════════════════════════

class Neuron:
    """A single artificial neuron."""
    def __init__(self, n_inputs):
        """Initialize with random weights and zero bias."""
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

    def forward(self, x):
        """Forward pass: weighted sum + activation."""
        z = np.dot(self.weights, x) + self.bias
        return relu(z)

    def __str__(self):
        return f"Neuron(w={self.weights}, b={self.bias:.4f})"

# Create a neuron with 3 inputs
np.random.seed(42)
neuron = Neuron(3)
print(f"\n{neuron}")

# Feed it some data
x = np.array([0.5, -0.3, 0.8])
output = neuron.forward(x)
print(f"Input: {x}")
print(f"Output: {output:.4f}")
print(f"Calculation: {neuron.weights} @ {x} + {neuron.bias:.4f} = {np.dot(neuron.weights, x) + neuron.bias:.4f}")


# ════════════════════════════════════════════════════════
# SECTION 3: NEURAL NETWORK LAYER
# ════════════════════════════════════════════════════════

class DenseLayer:
    """A fully connected layer of neurons."""
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        """Initialize weights and biases."""
        # He initialization for ReLU networks
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.bias = np.zeros(n_neurons)
        self.activation = activation

    def forward(self, x):
        """Forward pass through the layer."""
        z = x @ self.weights + self.bias

        if self.activation == 'relu':
            a = relu(z)
        elif self.activation == 'sigmoid':
            a = sigmoid(z)
        elif self.activation == 'tanh':
            a = tanh(z)
        elif self.activation == 'softmax':
            a = softmax(z)
        elif self.activation == 'linear':
            a = z
        else:
            a = z

        return a

# Create a layer: 3 inputs → 4 neurons, ReLU
np.random.seed(42)
layer = DenseLayer(3, 4, activation='relu')
print(f"\nLayer: 3 inputs → 4 neurons (ReLU)")
print(f"Weights shape: {layer.weights.shape}")
print(f"Bias shape: {layer.bias.shape}")

x = np.array([0.5, -0.3, 0.8])
output = layer.forward(x)
print(f"Input:  {x}")
print(f"Output: {output}")


# ════════════════════════════════════════════════════════
# SECTION 4: FULL NEURAL NETWORK
# ════════════════════════════════════════════════════════

class SimpleNeuralNetwork:
    """A multi-layer neural network for classification."""
    def __init__(self, layer_sizes):
        """layer_sizes = [n_inputs, hidden1, hidden2, ..., n_outputs]"""
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            activation = 'relu'  # Hidden layers use ReLU
            if i == len(layer_sizes) - 2:  # Last layer
                activation = 'softmax' if layer_sizes[-1] > 1 else 'sigmoid'
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], activation))

    def forward(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def summary(self):
        """Print network architecture."""
        print("\n=== Network Architecture ===")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i+1}: {layer.weights.shape[0]} → "
                  f"{layer.weights.shape[1]} ({layer.activation})")

# Create network: 4 inputs → 8 → 6 → 3 outputs (3-class classification)
np.random.seed(42)
net = SimpleNeuralNetwork([4, 8, 6, 3])
net.summary()

# Forward pass
x = np.array([0.1, 0.5, -0.3, 0.8])
output = net.forward(x)
print(f"\nInput:  {x}")
print(f"Output: {output}")
print(f"Sum (should be ~1.0 for softmax): {output.sum():.4f}")
print(f"Predicted class: {output.argmax()}")


# ════════════════════════════════════════════════════════
# SECTION 5: BATCH PROCESSING
# ════════════════════════════════════════════════════════

# Neural networks process batches (multiple samples at once)
# This is critical for efficiency

np.random.seed(42)
batch_size = 5
n_features = 4
batch = np.random.randn(batch_size, n_features)  # 5 samples, 4 features each

print(f"\n=== Batch Processing ===")
print(f"Batch shape: {batch.shape}")

# The network handles batches automatically (matrix multiplication)
batch_output = net.forward(batch)
print(f"Output shape: {batch_output.shape}")  # (5, 3) — 5 predictions, 3 classes each
print(f"Predicted classes: {batch_output.argmax(axis=1)}")


# ════════════════════════════════════════════════════════
# SECTION 6: VISUALIZE FORWARD PROPAGATION
# ════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 8))

# Draw network
layer_sizes = [4, 8, 6, 3]
layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']

for layer_idx, (size, name) in enumerate(zip(layer_sizes, layer_names)):
    x_pos = layer_idx * 3
    for neuron_idx in range(size):
        y_pos = (size - neuron_idx - 1) * 0.8 - size * 0.4
        color = 'skyblue' if layer_idx == 0 else 'lightcoral' if layer_idx == len(layer_sizes)-1 else 'lightgreen'
        circle = plt.Circle((x_pos, y_pos), 0.2, color=color, ec='black')
        ax.add_patch(circle)
        ax.text(x_pos, y_pos, f"{neuron_idx+1}", ha='center', va='center', fontsize=8)

    ax.text(x_pos, -size * 0.4 - 0.5, name, ha='center', fontsize=10, fontweight='bold')

    # Draw connections
    if layer_idx < len(layer_sizes) - 1:
        next_size = layer_sizes[layer_idx + 1]
        for i in range(size):
            y1 = (size - i - 1) * 0.8 - size * 0.4
            for j in range(next_size):
                y2 = (next_size - j - 1) * 0.8 - next_size * 0.4
                ax.plot([x_pos + 0.2, x_pos + 2.8], [y1, y2],
                       'gray', alpha=0.3, linewidth=0.5)

ax.set_xlim(-1, 10)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Neural Network: Forward Propagation (4→8→6→3)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ch16_network_architecture.png', dpi=150)
plt.show()
print("✓ Network architecture visualized")

print("\n" + "=" * 50)
print("✓ Chapter 16 complete!")
print("=" * 50)
```

---

## Real-World Example

```
┌──────────────────────────────────────────────────────────────┐
│  NEURAL NETWORKS IN THE REAL WORLD                           │
│                                                              │
│  Image Recognition (CNN):                                    │
│    Input: pixels → Hidden: edges → shapes → objects         │
│    Each layer learns more abstract features                  │
│                                                              │
│  ChatGPT (Transformer):                                     │
│    Input: tokens → Hidden: attention patterns → text        │
│    Billions of neurons arranged in transformer blocks        │
│                                                              │
│  Recommendations:                                             │
│    Input: user features → Hidden: preferences → rating       │
│    Netflix uses neural nets for personalization             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Wrong activation function.* Using sigmoid in hidden layers of deep networks causes vanishing gradients. Use ReLU.
2. *Not initializing weights properly.* All zeros = all neurons learn the same thing. Use He or Xavier initialization.
3. *Forgetting that networks process batches.* Always shape inputs as (batch_size, n_features), not (n_features,).
4. *Too few/many neurons.* Too few → underfit. Too many → overfit + slow training. Start with a moderate size.

---

## Mini-Project — Perceptron Learning

Build a perceptron that learns to classify AND/OR logic:

```python
"""
Mini-Project: Perceptron that learns AND/OR logic
"""
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.weights = np.random.randn(n_inputs)
        self.bias = 0.0
        self.lr = lr

    def predict(self, x):
        return 1 if np.dot(self.weights, x) + self.bias > 0 else 0

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            if errors == 0:
                print(f"  Converged at epoch {epoch+1}")
                break

# AND gate: output 1 only when both inputs are 1
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0, 0, 0, 1])

p_and = Perceptron(2, lr=0.1)
p_and.train(X_and, y_and, epochs=100)

print("AND gate results:")
for xi, target in zip(X_and, y_and):
    print(f"  {xi} → {p_and.predict(xi)} (expected {target})")

# OR gate: output 1 when any input is 1
X_or = np.array([[0,0], [0,1], [1,0], [1,1]])
y_or = np.array([0, 1, 1, 1])

p_or = Perceptron(2, lr=0.1)
p_or.train(X_or, y_or, epochs=100)

print("\nOR gate results:")
for xi, target in zip(X_or, y_or):
    print(f"  {xi} → {p_or.predict(xi)} (expected {target})")

# XOR: Can a single perceptron learn XOR? (No!)
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0, 1, 1, 0])

p_xor = Perceptron(2, lr=0.1)
p_xor.train(X_xor, y_xor, epochs=1000)
print("\nXOR gate results (single perceptron CANNOT learn XOR):")
for xi, target in zip(X_xor, y_xor):
    print(f"  {xi} → {p_xor.predict(xi)} (expected {target})")
print("→ A single perceptron can't learn XOR. We need multi-layer networks!")
```

---

## Exercises

### Easy
1. What does ReLU output for x = -5? For x = 3?
2. Draw a perceptron with 2 inputs. Label all components.
3. If weights = [0.5, 0.5] and bias = -0.5, what's the output for input [1, 1] using step activation?

### Medium
4. Implement a DenseLayer class with sigmoid activation. Test on random input.
5. Build a network with [2, 4, 4, 2] architecture. What's the total number of parameters?
6. Show that sigmoid squashes large values — what happens to sigmoid(100) and sigmoid(-100)?

### Advanced
7. Implement He and Xavier weight initialization. Compare their effects on training.
8. Build a multi-layer network that can learn XOR (needs at least 1 hidden layer).

---

## Chapter Summary

```
• Neuron = weighted sum + activation function
• ReLU for hidden layers, sigmoid/softmax for output
• Forward propagation = sequential matrix multiplications
• Networks process batches: (batch_size, features)
• Weights are initialized randomly (He/Xavier)
• Deeper layers learn more abstract patterns
• Single perceptrons can't learn XOR (need hidden layers)
```

---

## Interview Questions

1. "Why is ReLU preferred over sigmoid for hidden layers?"
2. "What is the vanishing gradient problem?"
3. "How does a neural network process a batch of inputs?"
4. "Why do we need non-linear activation functions?"

---

## Knowledge Check

1. What are the inputs, weights, bias, and output of a neuron?
2. Why can't a single perceptron solve XOR?
3. What activation would you use for: a) hidden layer, b) binary classification output, c) multi-class output, d) regression output?
4. What shape should input data have for batch processing?
5. Why do we initialize weights randomly instead of to zero?
