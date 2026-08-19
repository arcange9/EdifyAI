# Chapter 18 — Building a Neural Network from Scratch

## Learning Objectives

- Build a complete, trainable neural network using only NumPy
- Train it on the MNIST digit dataset
- Understand every component: forward pass, backward pass, optimization
- Compare performance with scikit-learn

---

## Why This Matters

Using PyTorch or TensorFlow without understanding what happens inside is like driving without knowing how a car works. Building from scratch *once* gives you intuition that lasts forever. After this chapter, every framework becomes transparent.

---

## Complete Implementation

```python
"""
Chapter 18 — Complete Neural Network from Scratch
A full implementation: forward pass, backpropagation, training loop,
mini-batch gradient descent, and evaluation on MNIST.

pip install numpy matplotlib scikit-learn
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import time

class NeuralNetwork:
    def __init__(self, sizes, lr=0.1, l2=0.001):
        self.sizes = sizes
        self.lr = lr
        self.l2 = l2
        self.weights = []
        self.biases = []
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        zs = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            zs.append(z)
            if i == len(self.weights) - 1:
                a = self._softmax(z)
            else:
                a = np.maximum(0, z)
            activations.append(a)
        return activations, zs

    def _softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def backward(self, activations, zs, y):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        n = y.shape[0]

        delta = (activations[-1] - y) / n

        for i in reversed(range(len(self.weights))):
            grads_w[i] = activations[i].T @ delta + self.l2 * self.weights[i]
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (zs[i-1] > 0)

        return grads_w, grads_b

    def train(self, X, y, X_val, y_val, epochs=50, batch_size=32):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            epoch_loss = 0
            n_batches = 0

            for start in range(0, len(X), batch_size):
                idx = indices[start:start + batch_size]
                X_batch, y_batch = X[idx], y[idx]

                activations, zs = self.forward(X_batch)
                loss = -np.mean(np.sum(y_batch * np.log(activations[-1] + 1e-15), axis=1))
                grads_w, grads_b = self.backward(activations, zs, y_batch)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * grads_w[i]
                    self.biases[i] -= self.lr * grads_b[i]

                epoch_loss += loss
                n_batches += 1

            train_acc = self.accuracy(X, y)
            val_acc = self.accuracy(X_val, y_val)

            self.history['train_loss'].append(epoch_loss / n_batches)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1].argmax(axis=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y.argmax(axis=1))

    def count_params(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))


# ─── Train on MNIST (8x8 digits) ─────────────────
digits = load_digits()
X = digits.data / 16.0
y_raw = digits.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_raw)

print("=== Neural Network from Scratch ===")
print(f"Data: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
print(f"Input: {X.shape[1]} features (8x8 pixels)")

np.random.seed(42)
net = NeuralNetwork([64, 32, 16, 10], lr=0.1, l2=0.001)
print(f"Architecture: {net.sizes}")
print(f"Parameters: {net.count_params()}")

start = time.time()
net.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
elapsed = time.time() - start

print(f"\nFinal Test Accuracy: {net.accuracy(X_test, y_test):.4f}")
print(f"Training time: {elapsed:.1f}s")

# ─── Visualize ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(net.history['train_loss'], 'b-')
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(net.history['train_acc'], 'b-', label='Train')
axes[1].plot(net.history['val_acc'], 'r-', label='Validation')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Confusion matrix
preds = net.predict(X_test)
cm = confusion_matrix(y_test.argmax(axis=1), preds)
axes[2].imshow(cm, cmap='Blues')
axes[2].set_title('Confusion Matrix')
for i in range(10):
    for j in range(10):
        axes[2].text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=6, color='white' if cm[i,j] > cm.max()/2 else 'black')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')

plt.tight_layout()
plt.savefig('ch18_mnist_scratch.png', dpi=150)
plt.show()

# ─── Show predictions ───────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
test_preds = net.predict(X_test)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    true_label = y_test[i].argmax()
    pred_label = test_preds[i]
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
    ax.axis('off')
plt.suptitle('MNIST Predictions (green=correct, red=wrong)', fontsize=14)
plt.tight_layout()
plt.savefig('ch18_predictions.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("✓ Chapter 18 complete!")
print("=" * 50)
```

---

## What We Built

```
┌──────────────────────────────────────────────────────────────┐
│  OUR NEURAL NETWORK — INSIDE AND OUT                       │
│                                                              │
│  Architecture: 64 → 32 → 16 → 10                            │
│  Parameters:  64×32 + 32 + 32×16 + 16 + 16×10 + 10 = 2762  │
│                                                              │
│  Forward pass:  3 matrix multiplications + ReLU + softmax  │
│  Backward pass: 3 gradient computations via chain rule      │
│  Optimizer:     Mini-batch gradient descent with L2         │
│  Loss:          Cross-entropy                                │
│                                                              │
│  This is EXACTLY what PyTorch does.                          │
│  The only difference: PyTorch automates the backward pass.   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Exercises

### Easy
1. How many parameters does [100, 50, 20, 5] have?
2. Change the architecture to [64, 64, 10]. Does it overfit more or less?
3. What happens if you set L2 regularization to 0?

### Medium
4. Add momentum to the optimizer. Show it converges faster.
5. Implement learning rate decay (halve LR every 20 epochs).
6. Train on a harder dataset (make_classification with 20 features).

### Advanced
7. Add dropout to the network. Show it reduces overfitting.
8. Implement early stopping: stop training when validation accuracy stops improving.

---

## Chapter Summary

```
• Built complete NN: forward + backward + training loop
• 2762 parameters learned from 1437 training samples
• ~95% accuracy on MNIST digits (from scratch!)
• This IS what PyTorch does — just automated
• Every framework is a convenience layer over these fundamentals
```

---

## Interview Questions

1. "How would you implement backpropagation from scratch?"
2. "What is L2 regularization and why is it useful?"
3. "How do you choose the number of layers and neurons?"
4. "What would you do if your neural network is overfitting?"
