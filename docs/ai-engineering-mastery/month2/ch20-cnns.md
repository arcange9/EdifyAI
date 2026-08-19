# Chapter 20 — Convolutional Neural Networks (CNNs)

## Learning Objectives

- Understand convolutions and why they work for images
- Know pooling, padding, and feature maps
- Build a CNN in PyTorch for image classification
- Understand transfer learning

---

## Why This Matters

CNNs revolutionized computer vision. Face ID, medical image diagnosis, self-driving cars, content moderation — all powered by CNNs. Even in the LLM era, CNNs remain essential for any task involving images.

---

## Concept Explanation

### Why Not Regular Neural Networks for Images?

```
┌──────────────────────────────────────────────────────────────┐
│  REGULAR NN vs CNN FOR IMAGES                                │
│                                                              │
│  A 28×28 image = 784 pixels → 784 input neurons             │
│  A 224×224 image = 50,176 pixels → 50,176 input neurons    │
│                                                              │
│  Regular NN: each neuron connects to ALL pixels              │
│  → 50,176 × 1000 hidden = 50 million weights for ONE layer │
│  → Too many parameters, ignores spatial structure            │
│                                                              │
│  CNN: small filter scans the image                           │
│  → 3×3 filter = only 9 weights, reused across image         │
│  → Detects local patterns (edges, shapes, textures)        │
│  → Much fewer parameters, much better performance           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### How Convolutions Work

```
┌─────────────────────────────────────────────────────────────┐
│  CONVOLUTION OPERATION                                       │
│                                                             │
│  Image (5x5):      Filter (3x3):       Output (3x3):        │
│  ┌───────────┐    ┌───────────┐     ┌───────────┐          │
│  │ 1 2 0 1 1│    │ 1  0 -1  │     │ r₁₁ r₁₂ r₁₃│         │
│  │ 0 3 1 2 0│    │ 1  0 -1  │     │            │          │
│  │ 1 2 0 0 1│    │ 1  0 -1  │     │ r₂₁ r₂₂ r₂₃│         │
│  │ 0 1 3 1 0│    └───────────┘     │            │          │
│  │ 2 0 1 2 1│                      │ r₃₁ r₃₂ r₃₃│         │
│  └───────────┘                      └───────────┘          │
│                                                             │
│  The filter slides across the image:                        │
│  At each position, multiply overlapping values and sum     │
│                                                             │
│  r₁₁ = (1×1 + 2×0 + 0×-1) + (0×1 + 3×0 + 1×-1) + ...     │
│                                                             │
│  Different filters detect different patterns:               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│  │ 1  1  1  │  │ -1 0  1  │  │  0  1  0  │              │
│  │ 0  0  0  │  │ -1 0  1  │  │  1 -4  1  │              │
│  │-1 -1 -1  │  │ -1 0  1  │  │  0  1  0  │              │
│  └───────────┘  └───────────┘  └───────────┘              │
│  Horizontal     Vertical        Edge/Laplacian              │
│  edge filter    edge filter     detector                    │
│                                                             │
│  In CNNs, these filter values are LEARNED during training    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### CNN Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  CNN ARCHITECTURE                                            │
│                                                             │
│  INPUT IMAGE                                                 │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────────────┐                                       │
│  │  CONV LAYER 1    │  → Learns edges, simple patterns      │
│  │  (32 filters)    │                                       │
│  └────────┬─────────┘                                       │
│    │      ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  POOLING         │  → Reduces size (2x downsampling)    │
│  │  (MaxPool 2x2)   │                                       │
│  └────────┬─────────┘                                       │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────────────┐                                       │
│  │  CONV LAYER 2    │  → Learns shapes, textures            │
│  │  (64 filters)    │                                       │
│  └────────┬─────────┘                                       │
│    │      ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  POOLING         │  → Further reduce size                │
│  └────────┬─────────┘                                       │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────────────┐                                       │
│  │  FLATTEN         │  → Convert 2D to 1D vector            │
│  └────────┬─────────┘                                       │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────────────┐                                       │
│  │  DENSE (FC)      │  → Classification based on features   │
│  │  (128 neurons)   │                                       │
│  └────────┬─────────┘                                       │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────────────┐                                       │
│  │  OUTPUT          │  → Class probabilities (softmax)      │
│  │  (10 classes)    │                                       │
│  └──────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Transfer Learning

```
┌──────────────────────────────────────────────────────────────┐
│  TRANSFER LEARNING                                           │
│                                                              │
│  "Don't reinvent the wheel — use a pre-trained wheel"        │
│                                                              │
│  1. Take a model trained on millions of images (ImageNet)    │
│  2. Remove the final classification layer                    │
│  3. Add your own layer for YOUR classes                       │
│  4. Fine-tune on YOUR data (much less data needed)          │
│                                                              │
│  Why it works:                                               │
│  • Early layers detect edges, shapes (universal features)   │
│  • These features are useful for ANY image task              │
│  • You only need to retrain the final layers                 │
│  • Works with as few as 100 images!                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — CNN in PyTorch

```python
"""
Chapter 20 — CNNs with PyTorch
pip install torch torchvision matplotlib numpy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ─── Data preparation ──────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
test_data = datasets.MNIST('.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# ─── CNN Model ────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 1→32 channels, 3x3 filter
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32→64 channels
        self.pool = nn.MaxPool2d(2, 2)                  # 2x2 pooling
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)         # After 2 pooling: 28→14→7
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))    # Conv1 + ReLU
        x = self.pool(x)                  # Downsample: 28→14
        x = torch.relu(self.conv2(x))    # Conv2 + ReLU
        x = self.pool(x)                  # Downsample: 14→7
        x = x.view(-1, 64 * 7 * 7)       # Flatten
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))      # Dense + ReLU
        x = self.fc2(x)                   # Output (no softmax needed)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_params = sum(p.numel() for p in model.parameters())
print(f"CNN parameters: {n_params:,}")

# ─── Training ─────────────────────────────────
epochs = 5
history = {'loss': [], 'acc': []}

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    acc = correct / total
    history['loss'].append(epoch_loss / len(train_loader))
    history['acc'].append(acc)
    print(f"Epoch {epoch+1}: loss={history['loss'][-1]:.4f}, acc={acc:.4f}")

# ─── Evaluation ────────────────────────────────
model.eval()
test_correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_correct += output.max(1)[1].eq(target).sum().item()

print(f"\nTest accuracy: {test_correct/len(test_data):.4f}")

# ─── Visualize ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history['loss'], 'b-')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['acc'], 'g-')
axes[1].set_title('Training Accuracy')
axes[1].set_ylim(0.95, 1.0)
axes[1].grid(True, alpha=0.3)

# Visualize filters from first conv layer
filters = model.conv1.weight.data[:8]
for i in range(8):
    axes[2].subplot(2, 4, i+1) if hasattr(axes[2], 'subplot') else None
grid = axes[2].imshow(filters[i, 0].numpy(), cmap='viridis')
axes[2].set_title('Learned Conv Filters')

plt.tight_layout()
plt.savefig('ch20_cnn.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("✓ Chapter 20 complete!")
print("=" * 50)
```

---

## Mini-Project — Image Classifier (CNN)

Train a CNN on the CIFAR-10 dataset (color images: airplanes, cars, cats, dogs, etc.):

```python
"""
Project 5: Image Classifier (CNN) on CIFAR-10
"""
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10('.', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CIFAR10CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# Evaluate
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        correct += model(data).max(1)[1].eq(target).sum().item()
print(f"CIFAR-10 Test accuracy: {correct/len(test_data):.4f}")
```

---

## Exercises

### Easy
1. How many parameters does a 3×3 filter with 1 input channel and 32 output channels have?
2. What does max pooling do? Why reduce image size?
3. Why are CNNs better than fully-connected networks for images?

### Medium
4. Add batch normalization to the CNN. Compare training speed.
5. Train with and without dropout. Does dropout prevent overfitting?
6. Visualize the feature maps at each layer. What patterns do you see?

### Advanced
7. Implement transfer learning using a pre-trained ResNet from torchvision.
8. Build a CNN that can classify your own photos (use a small dataset).

---

## Chapter Summary

```
• Convolutions = learnable filters that detect patterns
• Pooling reduces spatial dimensions
• Early layers detect edges → later layers detect objects
• Transfer learning: use pre-trained models, add your head
• CNNs are still best for computer vision tasks
• PyTorch makes building CNNs straightforward
```

---

## Interview Questions

1. "Why are CNNs better than regular NNs for images?"
2. "What is transfer learning and when would you use it?"
3. "What does a convolution filter learn?"
4. "Explain the role of pooling in CNNs."
