# Chapter 13 — Supervised Learning: Classification

## Learning Objectives

- Understand classification (predicting categories, not numbers)
- Use logistic regression, k-NN, and SVM
- Know when to use which classifier
- Evaluate with precision, recall, F1 (not just accuracy)

---

## Why This Matters

Spam detection, disease diagnosis, fraud detection, image recognition — all are classification. Along with regression, this is the bread and butter of applied ML.

---

## Concept Explanation

### Classification vs Regression

```
REGRESSION:         CLASSIFICATION:
Predict a number    Predict a category

y = $350,000        y = "spam"
y = 72.5°F          y = "cat"
y = 1,234 users     y = "positive"

Output: continuous  Output: discrete labels
```

### Binary vs Multi-class

```
Binary:     2 classes   (spam/not spam, positive/negative)
Multi-class: 3+ classes (cat/dog/bird, A/B/C/D grade)
```

### Logistic Regression (for Classification)

```
┌──────────────────────────────────────────────────────────┐
│  LOGISTIC REGRESSION                                     │
│                                                          │
│  Despite the name, it's for CLASSIFICATION not regression│
│                                                          │
│  1. Compute: z = w₁x₁ + w₂x₂ + ... + b                 │
│  2. Apply sigmoid: p = 1 / (1 + e^(-z))                 │
│  3. If p >= 0.5 → class 1, else class 0                  │
│                                                          │
│  The sigmoid function:                                   │
│                                                          │
│  1.0 ─   ──────___________                               │
│       │  /                                               │
│  0.5 ──/───────  ← decision boundary (p=0.5)             │
│      /│                                                  │
│  0.0 _│__________                                        │
│      │  │                                                │
│     -3  0  3  → z                                        │
│                                                          │
│  Squeezes any value into [0, 1] = probability            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### k-Nearest Neighbors (k-NN)

```
┌──────────────────────────────────────────────────────────┐
│  K-NEAREST NEIGHBORS                                     │
│                                                          │
│  "Show me your friends and I'll tell you who you are"   │
│                                                          │
│  To classify a new point:                                │
│  1. Find the k nearest training points                   │
│  2. Vote: majority class wins                            │
│                                                          │
│  Visual (k=3):                                           │
│                                                          │
│    ● ●   ? ← new point to classify                       │
│    ● ● ●○     (find 3 nearest neighbors)                 │
│    ● ● ○○○  → 2 blue, 1 orange → class = blue            │
│                                                          │
│  No training needed (lazy learner)                       │
│  Prediction is slow with large data                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Evaluation Metrics (CRITICAL)

```
┌─────────────────────────────────────────────────────────────┐
│  CLASSIFICATION METRICS                                    │
│                                                             │
│  CONFUSION MATRIX:                                         │
│                    Predicted                                │
│                  ┌──────────┬──────────┐                    │
│                  │ Positive │ Negative │                    │
│  ┌───────────────┼──────────┼──────────┤                    │
│  │ Actual Pos    │    TP    │    FN    │                    │
│  │───────────────┼──────────┼──────────┤                    │
│  │ Actual Neg    │    FP    │    TN    │                    │
│  └───────────────┴──────────┴──────────┘                    │
│                                                             │
│  Accuracy  = (TP + TN) / Total                              │
│  Precision = TP / (TP + FP) — of predicted positive,      │
│              how many are actually positive?                │
│  Recall    = TP / (TP + FN) — of actual positive,         │
│              how many did we find?                          │
│  F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)│
│              — harmonic mean of precision and recall       │
│                                                             │
│  When accuracy is misleading:                               │
│  If 99% of emails are not spam, a model that always        │
│  predicts "not spam" has 99% accuracy but is useless!      │
│  Use precision and recall instead.                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — Classification with scikit-learn

```python
"""
Chapter 13 — Classification
pip install scikit-learn numpy pandas matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score, accuracy_score)

# ─── 1. Sigmoid Function (Logistic Regression core) ───
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid(z), 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision threshold (0.5)')
plt.xlabel('z')
plt.ylabel('P(class=1)')
plt.title('Sigmoid Function — Core of Logistic Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch13_sigmoid.png', dpi=150)
plt.show()

# ─── 2. Binary Classification ─────────────────────
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           class_sep=1.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "k-NN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "SVM (RBF)": SVC(kernel='rbf', random_state=42),
    "SVM (Linear)": SVC(kernel='linear', random_state=42),
}

print("=== Binary Classification Results ===\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"  {name}:")
    print(f"    Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

# ─── 3. Multi-class Classification (Iris) ────────
print("\n=== Multi-class: Iris Dataset ===\n")
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris)

model = LogisticRegression(max_iter=200)
model.fit(X_train_i, y_train_i)
y_pred_i = model.predict(X_test_i)

print(f"  Accuracy: {accuracy_score(y_test_i, y_pred_i):.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test_i, y_pred_i, target_names=iris.target_names))

print(f"  Confusion Matrix:")
print(confusion_matrix(y_test_i, y_pred_i))

# ─── 4. When Accuracy is Misleading ──────────────
print("\n=== Imbalanced Classes: Why Accuracy is Misleading ===\n")

np.random.seed(42)
y_imbalanced = np.array([0]*990 + [1]*10)  # 99% class 0, 1% class 1
y_dumb = np.zeros(1000)  # Always predict class 0

print(f"  Dumb model (always predicts class 0):")
print(f"    Accuracy: {accuracy_score(y_imbalanced, y_dumb):.4f} (99%!)")
print(f"    But it detected 0 out of 10 positive cases!")
print(f"    Recall: {recall_score(y_imbalanced, y_dumb):.4f}")
print(f"    F1: {f1_score(y_imbalanced, y_dumb):.4f}")
print(f"  → Accuracy alone is USELESS for imbalanced data!")

# ─── 5. Decision Boundaries ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
model_dict = {
    "Logistic Regression": LogisticRegression(),
    "k-NN (k=5)": KNeighborsClassifier(5),
    "k-NN (k=1)": KNeighborsClassifier(1),
    "SVM (RBF)": SVC(kernel='rbf'),
}

for idx, (name, model) in enumerate(model_dict.items()):
    ax = axes[idx // 2, idx % 2]
    model.fit(X_train, y_train)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                          np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm',
              edgecolors='black', s=20)
    ax.set_title(f'{name}\nAcc={model.score(X_test, y_test):.3f}')

plt.tight_layout()
plt.savefig('ch13_decision_boundaries.png', dpi=150)
plt.show()
print("✓ Decision boundaries saved")

print("\n" + "=" * 50)
print("✓ Chapter 13 complete!")
print("=" * 50)
```

---

## Mini-Project — Image Classifier (Simple)

```python
"""
Project 3: Simple Image Classifier
Uses scikit-learn on the digits dataset (8x8 pixel images)
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target  # 1797 samples, 64 features each

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf'),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.4f}")

# Visualize a few predictions
best_model = SVC(kernel='rbf')
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test[:8])
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Pred: {predictions[i]}, True: {y_test[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('ch13_digit_classification.png', dpi=150)
plt.show()
```

---

## Exercises

### Easy
1. What does the sigmoid function output for z=0? Why?
2. Train a logistic regression model. What do the coefficients mean?
3. If precision=0.8 and recall=0.6, what's the F1 score?

### Medium
4. Compare k-NN with k=1, k=5, and k=20. Which overfits?
5. Create a binary classification problem with 95% class 0 and 5% class 1. Show why accuracy is misleading.
6. Plot decision boundaries for LogisticRegression and k-NN on the same data.

### Advanced
7. Implement logistic regression from scratch using gradient descent and the sigmoid function.
8. Build a multi-class classifier that handles 10 classes and reports per-class precision/recall.

---

## Chapter Summary

```
• Classification predicts categories, not numbers
• Logistic regression uses sigmoid to output probabilities
• k-NN: majority vote of nearest neighbors
• Accuracy is misleading for imbalanced data
• Use precision, recall, and F1 together
• Confusion matrix shows TP/FP/TN/FN breakdown
```

---

## Interview Questions

1. "When is accuracy a bad metric?"
2. "Explain the tradeoff between precision and recall."
3. "How does k-NN work and what's a good value for k?"
4. "How would you handle a dataset with 99% negative and 1% positive examples?"
