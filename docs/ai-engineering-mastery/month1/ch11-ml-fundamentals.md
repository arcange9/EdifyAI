# Chapter 11 — Machine Learning Fundamentals

## Learning Objectives

- Understand the ML workflow end-to-end
- Know train/validation/test splits and why they matter
- Understand bias-variance tradeoff
- Use scikit-learn for the first time

---

## Why This Matters

Everything in Month 2 (deep learning) builds on the fundamentals here. If you don't understand train/test splits, cross-validation, and the bias-variance tradeoff, you'll build models that look good in development but fail in production.

---

## Concept Explanation

### The ML Workflow

```
┌──────────────────────────────────────────────────────────────┐
│  THE MACHINE LEARNING WORKFLOW                               │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Problem  │──→│  Data    │──→│ Features │──→│  Model   │  │
│  │ Define   │   │ Collect  │   │ Engineer │   │  Train   │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                     │        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐         │        │
│  │  Deploy  │←──│ Tune     │←──│ Evaluate │←────────┘        │
│  │  Monitor │   │ Improve  │   │ Metrics  │                  │
│  └──────────┘   └──────────┘   └──────────┘                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Train / Validation / Test Split

```
┌─────────────────────────────────────────────────────────────┐
│  DATA SPLITS                                                │
│                                                             │
│  ┌───────────────┬──────────────┬──────────────┐            │
│  │  TRAINING     │  VALIDATION  │  TEST         │            │
│  │  (60-70%)     │  (15-20%)    │  (15-20%)     │            │
│  │               │              │               │            │
│  │  Model learns │  Tune hyper- │  Final,      │            │
│  │  patterns     │  parameters  │  honest       │            │
│  │  from this    │  on this     │  evaluation   │            │
│  │               │              │               │            │
│  │  Used during  │  Used during │  Used ONCE    │            │
│  │  fit()       │  tuning      │  at the end   │            │
│  └───────────────┴──────────────┴──────────────┘            │
│                                                             │
│  Why 3 splits?                                              │
│  Train: model learns → can overfit                         │
│  Validation: we tune → can overfit hyperparams             │
│  Test: never touched until final check → honest score     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Bias-Variance Tradeoff

```
┌──────────────────────────────────────────────────────────────┐
│  THE BIAS-VARIANCE TRADEOFF                                  │
│                                                              │
│  HIGH BIAS (underfitting)                                    │
│  ┌─────────────────┐                                         │
│  │   Model is too  │  Training error: HIGH                   │
│  │   simple        │  Test error: HIGH                       │
│  │                 │  Example: predicting house price        │
│  │  Too few        │  using only number of bedrooms          │
│  │  parameters     │                                         │
│  └─────────────────┘                                         │
│                                                              │
│  GOOD FIT (balanced)                                         │
│  ┌─────────────────┐                                         │
│  │  Model captures │  Training error: LOW                    │
│  │  the right      │  Test error: LOW                         │
│  │  patterns       │  Example: using sqft, location,         │
│  │                 │  bedrooms, age → good predictions       │
│  └─────────────────┘                                         │
│                                                              │
│  HIGH VARIANCE (overfitting)                                 │
│  ┌─────────────────┐                                         │
│  │  Model is too   │  Training error: VERY LOW                │
│  │  complex        │  Test error: HIGH                        │
│  │                 │  Example: memorizing every training     │
│  │  Too many       │  example including noise                 │
│  │  parameters     │                                         │
│  └─────────────────┘                                         │
│                                                              │
│  Total Error = Bias² + Variance + Irreducible Error         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### How to Detect Overfitting vs Underfitting

```
┌──────────────────────────────────────────────────┐
│  DIAGNOSIS BY ERROR PATTERNS                     │
│                                                  │
│  Training Loss │ Test Loss │ Diagnosis           │
│  ──────────────┼───────────┼──────────────────  │
│     HIGH       │   HIGH    │ UNDERFITTING        │
│  (model too    │           │ (increase capacity) │
│   simple)      │           │                      │
│     LOW        │   HIGH    │ OVERFITTING         │
│  (memorized    │           │ (reduce complexity, │
│   data)        │           │  add regularization) │
│     LOW        │   LOW     │ GOOD FIT ✓          │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Code — First ML Pipeline with scikit-learn

```python
"""
Chapter 11 — ML Fundamentals with scikit-learn
pip install scikit-learn numpy pandas matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─── Load/Create Data ─────────────────────────
np.random.seed(42)
n = 1000
X = pd.DataFrame({
    "sqft": np.random.uniform(500, 5000, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 4, n),
    "age": np.random.randint(0, 50, n),
})
# Target: price based on features + noise
y = (150 * X["sqft"] + 20000 * X["bedrooms"] + 15000 * X["bathrooms"]
     - 500 * X["age"] + np.random.normal(0, 10000, n))

# ─── 3-Way Split ───────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ─── Train Multiple Models ────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree (deep)": DecisionTreeRegressor(max_depth=20, random_state=42),
    "Decision Tree (shallow)": DecisionTreeRegressor(max_depth=3, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    results.append({"model": name, "train_r2": train_r2, "val_r2": val_r2})
    gap = train_r2 - val_r2
    diagnosis = "OVERFITTING" if gap > 0.1 else "UNDERFITTING" if val_r2 < 0.5 else "GOOD"
    print(f"\n  {name}:")
    print(f"    Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Gap: {gap:.4f}")
    print(f"    Diagnosis: {diagnosis}")

# ─── Cross-Validation ─────────────────────────
print("\n=== Cross-Validation ===")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")

# ─── Visualize ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model comparison
names = [r["model"] for r in results]
train_scores = [r["train_r2"] for r in results]
val_scores = [r["val_r2"] for r in results]
x = np.arange(len(names))
axes[0].bar(x - 0.15, train_scores, 0.3, label="Train R²", color="blue")
axes[0].bar(x + 0.15, val_scores, 0.3, label="Val R²", color="orange")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=45, ha="right")
axes[0].set_ylabel("R² Score")
axes[0].set_title("Train vs Validation: Overfitting Check")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning curve (show how data size affects performance)
train_sizes = [50, 100, 200, 400, 600, len(X_train)]
train_scores_lc = []
val_scores_lc = []
for size in train_sizes:
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train[:size], y_train[:size])
    train_scores_lc.append(r2_score(y_train[:size], model.predict(X_train[:size])))
    val_scores_lc.append(r2_score(y_val, model.predict(X_val)))

axes[1].plot(train_sizes, train_scores_lc, 'bo-', label="Train R²")
axes[1].plot(train_sizes, val_scores_lc, 'ro-', label="Val R²")
axes[1].set_xlabel("Training Set Size")
axes[1].set_ylabel("R² Score")
axes[1].set_title("Learning Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch11_ml_fundamentals.png', dpi=150)
plt.show()
print("✓ Visualization saved")

print("\n" + "=" * 50)
print("✓ Chapter 11 complete!")
print("=" * 50)
```

---

## Exercises

### Easy
1. Split a dataset of 1000 samples into 70% train, 15% validation, 15% test.
2. Train a LinearRegression model and print train and test R² scores.
3. If train accuracy is 99% and test accuracy is 60%, what's wrong?

### Medium
4. Use 5-fold cross-validation to evaluate a model. Report mean and std.
5. Create a learning curve that shows how performance changes with more training data.
6. Compare a decision tree with max_depth=2 vs max_depth=20. Which overfits?

### Advanced
7. Implement train/validation/test split from scratch (no sklearn).
8. Build a function that automatically diagnoses overfitting/underfitting from train/val errors.

---

## Chapter Summary

```
• 3-way split: train (learn), validation (tune), test (final check)
• High train + high test error = underfitting
• Low train + high test error = overfitting
• Cross-validation: more robust than single split
• scikit-learn: fit() to train, predict() to test
• Learning curves show if more data would help
```

---

## Interview Questions

1. "Explain the bias-variance tradeoff."
2. "Why do we need three data splits instead of two?"
3. "How would you detect overfitting in your model?"
4. "What is cross-validation and why is it better than a single train/test split?"
