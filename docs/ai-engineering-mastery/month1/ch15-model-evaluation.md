# Chapter 15 — Model Evaluation, Overfitting & Cross-Validation

## Learning Objectives

- Master all evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Detect and prevent overfitting and underfitting
- Implement k-fold cross-validation
- Use learning curves to diagnose model problems

---

## Why This Matters

A model that scores 95% in the notebook but 60% in production is worse than useless — it's misleading. Proper evaluation is what separates a student from an engineer. You must be able to trust your metrics.

---

## Concept Explanation

### Evaluation Metrics for Classification

```
┌─────────────────────────────────────────────────────────────┐
│  ALL CLASSIFICATION METRICS                              │
│                                                             │
│  ┌───────────┬───────────┐                                │
│  │  Actual + │  Actual - │                                │
│  ├───────────┼───────────┤                                │
│  │ Pred + TP │ Pred + FP │                                │
│  │ Pred - FN │ Pred - TN │                                │
│  └───────────┴───────────┘                                │
│                                                             │
│  Accuracy  = (TP + TN) / (TP + TN + FP + FN)             │
│  Precision = TP / (TP + FP)                               │
│  Recall    = TP / (TP + FN)                               │
│  F1        = 2 × P × R / (P + R)                         │
│                                                             │
│  ROC Curve: plot True Positive Rate vs False Positive Rate│
│  AUC: area under ROC curve (0.5 = random, 1.0 = perfect)  │
│                                                             │
│  When to use which:                                       │
│  • Balanced data → accuracy is fine                      │
│  • Imbalanced data → precision, recall, F1              │
│  • Need ranking quality → ROC-AUC                        │
│  • Medical diagnosis → recall (don't miss positive cases)│
│  • Spam filter → precision (don't block good emails)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Validation

```
┌──────────────────────────────────────────────────────────────┐
│  K-FOLD CROSS-VALIDATION                                     │
│                                                              │
│  Data: [═══════════════════════════════════════]            │
│                                                              │
│  Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]                 │
│  Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]                 │
│  Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]                 │
│  Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]                 │
│  Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]                 │
│                                                              │
│  Each fold: train on 4/5, test on 1/5, record score        │
│  Final score = mean of 5 scores ± std                       │
│                                                              │
│  Why better than single split?                              │
│  • Uses all data for testing (no lucky/unlucky split)     │
│  • Shows variance (is the model stable?)                  │
│  • More honest estimate of real performance                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Learning Curves

```
┌──────────────────────────────────────────────────────────────┐
│  LEARNING CURVES — Diagnose Model Health                    │
│                                                              │
│  GOOD MODEL:                                                │
│  Acc ↑ ─────────── (train and test converge, both high)    │
│         /  ●●●●                                            │
│         /●●●●                                              │
│         ●●●●                                               │
│         ─────────────→ data size                            │
│                                                              │
│  OVERFITTING:                                               │
│  Acc ↑ ────●●●●●●● (train stays high)                      │
│         ●●●                                                │
│         ●●     ●●●● (test stays low, big gap)             │
│         ─────────────→ data size                            │
│  Fix: more data, regularization, simpler model              │
│                                                              │
│  UNDERFITTING:                                              │
│  Acc ↑ ────●●●● (both low)                                 │
│         ●●●●●                                              │
│         ●●●●●●●                                            │
│         ─────────────→ data size                            │
│  Fix: more complex model, better features                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Complete Evaluation Toolkit

```python
"""
Chapter 15 — Model Evaluation, Overfitting, Cross-Validation
pip install scikit-learn numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import (train_test_split, cross_val_score,
                                       learning_curve, StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, roc_auc_score,
                              classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)

# ─── 1. Full Classification Evaluation ──────────
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42, weights=[0.7, 0.3])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree (deep)": DecisionTreeClassifier(random_state=42),
    "Decision Tree (depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

print("=== Complete Model Evaluation ===\n")
print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("─" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name:<25} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")

# ─── 2. ROC Curves ─────────────────────────────
plt.figure(figsize=(10, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch15_roc_curves.png', dpi=150)
plt.show()

# ─── 3. Cross-Validation ────────────────────────
print("\n=== 5-Fold Cross-Validation ===\n")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"  {name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")

# ─── 4. Learning Curves ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate([
    ("Underfitting", LogisticRegression(max_iter=100, random_state=42)),
    ("Overfitting", DecisionTreeClassifier(max_depth=20, random_state=42)),
    ("Good Fit", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
]):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1', random_state=42)

    axes[idx].plot(train_sizes, train_scores.mean(axis=1), 'bo-', label='Train')
    axes[idx].plot(train_sizes, val_scores.mean(axis=1), 'ro-', label='Validation')
    axes[idx].fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                            train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
    axes[idx].fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                            val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='red')
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Training Size')
    axes[idx].set_ylabel('F1 Score')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch15_learning_curves.png', dpi=150)
plt.show()

# ─── 5. Confusion Matrix ───────────────────────
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax,
                                        display_labels=['Negative', 'Positive'])
plt.title('Confusion Matrix — Random Forest')
plt.tight_layout()
plt.savefig('ch15_confusion_matrix.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("✓ Chapter 15 complete! Month 1 finished!")
print("=" * 50)
```

---

## Mini-Project — Model Evaluation Report

Build a function that takes any model + data and generates a complete evaluation report:
1. Train/test split
2. 5-fold cross-validation
3. All metrics (accuracy, precision, recall, F1, AUC)
4. Confusion matrix
5. ROC curve
6. Learning curve
7. Diagnosis (overfitting, underfitting, good fit)

---

## Exercises

### Easy
1. Calculate precision, recall, and F1 for TP=40, FP=10, FN=5, TN=45.
2. What does an AUC of 0.5 mean? What does 1.0 mean?
3. Run 5-fold CV on a model. Report mean ± std.

### Medium
4. Create an imbalanced dataset (90/10). Show why accuracy is misleading.
5. Plot learning curves for an overfitting model and identify the problem.
6. Compare ROC curves for 3 models. Which has the best ranking ability?

### Advanced
7. Implement stratified k-fold cross-validation from scratch (no sklearn).
8. Build an automated model selection tool that tries multiple models and reports the best one.

---

## Chapter Summary

```
• Accuracy is not enough — use precision, recall, F1, AUC
• Cross-validation gives honest, stable estimates
• Learning curves diagnose overfitting/underfitting
• ROC-AUC measures ranking quality (not just classification)
• Confusion matrix shows exactly where errors happen
• Always evaluate on data the model has NEVER seen
```

---

## Interview Questions

1. "How do you evaluate an imbalanced classification model?"
2. "Explain what a learning curve tells you and how to use it."
3. "What's the difference between precision and recall? When would you prioritize one over the other?"
4. "How does k-fold cross-validation work and why is it better than a single train/test split?"

---

## Knowledge Check

1. If a model has 99% accuracy but 0% recall, what does that mean?
2. What does a large gap between train and validation scores indicate?
3. Why use stratified k-fold instead of regular k-fold?
4. What does AUC measure?
5. Name two ways to fix overfitting.
