# Chapter 19 — PyTorch: Deep Learning Framework

## Learning Objectives

- Understand what PyTorch is and why it's the most popular DL framework
- Master tensors, autograd, and the training loop in PyTorch
- Build and train a neural network using PyTorch
- Use datasets, dataloaders, and GPU acceleration

---

## Why This Matters

Building neural networks from scratch in NumPy (Chapter 18) taught you what happens inside. PyTorch automates the tedious parts — especially backpropagation via `autograd` — while giving you full control. 90% of AI research and production code uses PyTorch.

---

## Concept Explanation

### PyTorch vs From Scratch

```
┌──────────────────────────────────────────────────────────────┐
│  FROM SCRATCH (NumPy) vs PYTORCH                            │
│                                                              │
│  FROM SCRATCH:                         PYTORCH:            │
│                                                              │
│  Forward: manual matmul               model(x)              │
│  Backward: manual chain rule          loss.backward()       │
│  Update: manual weight -= lr*grad     optimizer.step()     │
│  Gradients: manual computation         autograd (automatic) │
│  GPU: not supported                    .to('cuda')           │
│  Batching: manual slicing              DataLoader            │
│                                                              │
│  PyTorch handles the MATH of gradients                      │
│  You focus on the ARCHITECTURE and DATA                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### The PyTorch Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  PYTORCH WORKFLOW                                            │
│                                                             │
│  1. DEFINE MODEL (nn.Module)                                │
│     class MyModel(nn.Module):                               │
│         def __init__(self): define layers                  │
│         def forward(self, x): connect layers              │
│                                                             │
│  2. CHOOSE LOSS + OPTIMIZER                                 │
│     criterion = nn.CrossEntropyLoss()                       │
│     optimizer = torch.optim.Adam(model.parameters())        │
│                                                             │
│  3. TRAINING LOOP                                           │
│     for epoch in range(epochs):                             │
│         for batch in dataloader:                            │
│             optimizer.zero_grad()  ← clear old gradients   │
│             output = model(batch.x)  ← forward pass         │
│             loss = criterion(output, batch.y) ← compute loss│
│             loss.backward()  ← autograd computes gradients  │
│             optimizer.step()  ← update weights              │
│                                                             │
│  4. EVALUATE                                                │
│     with torch.no_grad():  ← no gradients needed           │
│         predictions = model(test_data)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Tensors — PyTorch's NumPy

```python
# Tensors are like NumPy arrays but with superpowers:
# 1. Automatic gradient computation (autograd)
# 2. GPU acceleration (.to('cuda'))
# 3. Deep learning operations (built-in layers)

import torch

# Create tensors
x = torch.randn(3, 4)          # Random tensor (like np.random.randn)
y = torch.zeros(5)             # Zeros
z = torch.tensor([1, 2, 3])   # From list

# Operations (same as NumPy)
result = x @ x.T              # Matrix multiplication
sum_x = x.sum()               # Reduction
mean_x = x.mean(dim=0)        # Mean along dimension

# The magic: requires_grad=True
w = torch.randn(3, 4, requires_grad=True)
# Now PyTorch tracks ALL operations on w
# When you call .backward(), it computes gradients automatically
```

---

## Code — Complete PyTorch Training

```python
"""
Chapter 19 — PyTorch Deep Learning
pip install torch torchvision matplotlib numpy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ════════════════════════════════════════════════════════
# SECTION 1: TENSORS AND AUTOGRAD
# ════════════════════════════════════════════════════════

print("=== Tensors and Autograd ===\n")

# Create tensor with gradient tracking
x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)

# Do some computation
y = (x ** 2).sum()  # y = x₁² + x₂² + x₃² = 4 + 9 + 16 = 29

print(f"  x = {x}")
print(f"  y = x² sum = {y}")

# Compute gradients: dy/dx = 2x
y.backward()

print(f"  dy/dx = {x.grad}")  # Should be [4, 6, 8]
print(f"  (Expected: 2*x = [4, 6, 8])")


# ════════════════════════════════════════════════════════
# SECTION 2: DEFINE A MODEL
# ════════════════════════════════════════════════════════

class MNISTModel(nn.Module):
    """A simple feedforward neural network for classification."""
    def __init__(self, input_size=64, hidden1=128, hidden2=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)  # No softmax (CrossEntropyLoss includes it)
        return x

# ════════════════════════════════════════════════════════
# SECTION 3: PREPARE DATA
# ════════════════════════════════════════════════════════

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16.0
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Training samples: {len(X_train)}")
print(f"Batches per epoch: {len(train_loader)}")

# ════════════════════════════════════════════════════════
# SECTION 4: TRAINING LOOP
# ════════════════════════════════════════════════════════

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = MNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params}")

history = {'loss': [], 'acc': []}
epochs = 30

print(f"\n=== Training for {epochs} epochs ===")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()          # Clear gradients
        outputs = model(batch_x)        # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()                 # Backward pass (autograd!)
        optimizer.step()                # Update weights

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total
    history['loss'].append(avg_loss)
    history['acc'].append(accuracy)

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.4f}")

# ════════════════════════════════════════════════════════
# SECTION 5: EVALUATION
# ════════════════════════════════════════════════════════

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t.to(device))
    _, test_preds = test_outputs.max(1)
    test_acc = (test_preds == y_test_t.to(device)).float().mean().item()

print(f"\nTest accuracy: {test_acc:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['loss'], 'b-')
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['acc'], 'g-')
axes[1].set_title('Training Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch19_pytorch_training.png', dpi=150)
plt.show()

# Show predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
preds = test_preds.cpu().numpy()
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    color = 'green' if preds[i] == y_test[i] else 'red'
    ax.set_title(f'T:{y_test[i]} P:{preds[i]}', color=color)
    ax.axis('off')
plt.tight_layout()
plt.savefig('ch19_pytorch_predictions.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("✓ Chapter 19 complete!")
print("=" * 50)
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Forgetting `optimizer.zero_grad()`.* Gradients accumulate by default. Always zero them.
2. *Not calling `model.train()` and `model.eval()`.* Affects dropout and batch normalization.
3. *Using `torch.no_grad()` during evaluation.* Saves memory and speeds up inference.
4. *Forgetting to move data to GPU.* Model and data must be on same device.

---

## Chapter Summary

```
• PyTorch automates backpropagation with autograd
• nn.Module → define model, forward() → connect layers
• Training loop: zero_grad → forward → loss → backward → step
• DataLoader handles batching and shuffling
• model.train() vs model.eval() switches behavior
• torch.no_grad() for evaluation (saves memory)
• .to(device) for GPU acceleration
```

---

## Interview Questions

1. "What does autograd do?"
2. "Why do you need to call zero_grad() before each batch?"
3. "What's the difference between model.train() and model.eval()?"
4. "How does DataLoader help with training?"
