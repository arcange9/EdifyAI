# Chapter 14 — Decision Trees & Random Forests

## Learning Objectives

- Understand how decision trees split data
- Know entropy, Gini impurity, and information gain
- Build and tune random forests
- Use ensemble methods to improve accuracy

---

## Why This Matters

Random forests and gradient-boosted trees are still the most effective models for tabular data in production. Companies like banks, insurance, and healthcare use tree-based models more than deep learning for structured data.

---

## Concept Explanation

### Decision Tree — How It Works

```
┌──────────────────────────────────────────────────────────┐
│  DECISION TREE                                           │
│                                                          │
│  Should I study tonight?                                 │
│                                                          │
│               [Exam tomorrow?]                            │
│              /              \                              │
│           Yes               No                            │
│           /                  \                             │
│     [Grade < 80?]         [Other hw?]                    │
│      /         \            /        \                    │
│    Yes         No         Yes         No                  │
│     │          │          │          │                   │
│  STUDY!     Relax      STUDY!     Relax                   │
│                                                          │
│  Each node asks a yes/no question about a feature.       │
│  The tree splits data to maximize information gain.     │
│  Leaves contain the prediction.                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### How Trees Choose Splits

```
┌──────────────────────────────────────────────────────────┐
│  IMPURITY MEASURES                                       │
│                                                          │
│  GINI IMPURITY (default in sklearn):                     │
│  Gini = 1 - Σ(pᵢ)²                                      │
│  where pᵢ = probability of class i in the node         │
│                                                          │
│  Lower Gini = purer node (good split)                   │
│  Gini=0 means all samples are same class                │
│                                                          │
│  Example: node has [5 positive, 5 negative]              │
│  Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5              │
│                                                          │
│  After split: left=[5 pos, 0 neg], right=[0 pos, 5 neg] │
│  Gini_left = 1 - (1²) = 0                                │
│  Gini_right = 1 - (1²) = 0                               │
│  → Perfect split! (each child is pure)                   │
│                                                          │
│  INFORMATION GAIN:                                       │
│  Gain = Impurity(parent) - weighted avg Impurity(children)│
│  Tree picks the split that maximizes information gain    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Random Forests — Ensemble of Trees

```
┌──────────────────────────────────────────────────────────────┐
│  RANDOM FOREST                                               │
│                                                              │
│  "Many trees are better than one"                             │
│                                                              │
│  Training:                                                   │
│  1. Create N decision trees (e.g., 100)                      │
│  2. Each tree trains on a random subset of data (bagging)   │
│  3. Each split considers only a random subset of features    │
│  4. Trees are different from each other (decorrelated)      │
│                                                              │
│  Prediction:                                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐               │
│  │Tree 1│ │Tree 2│ │Tree 3│ │Tree 4│ │Tree 5│               │
│  │ → A  │ │ → B  │ │ → A  │ │ → A  │ │ → B  │               │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘               │
│      │        │        │        │        │                   │
│      └────────┴────────┴───┬────┴────────┘                   │
│                             │                                 │
│                    Majority Vote → A                          │
│                                                              │
│  Why it works:                                               │
│  • Single trees overfit → forests average them out          │
│  • Random subsets prevent all trees from making same        │
│    mistakes                                                  │
│  • "Wisdom of the crowd"                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Trees & Forests

```python
"""
Chapter 14 — Decision Trees & Random Forests
pip install scikit-learn numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# ─── 1. Single Decision Tree ──────────────────
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Deep tree (overfits)
deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
deep_tree.fit(X_train, y_train)
print(f"Deep tree — Train: {deep_tree.score(X_train, y_train):.4f}, "
      f"Test: {deep_tree.score(X_test, y_test):.4f}")

# Shallow tree (underfits)
shallow_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
shallow_tree.fit(X_train, y_train)
print(f"Shallow tree — Train: {shallow_tree.score(X_train, y_train):.4f}, "
      f"Test: {shallow_tree.score(X_test, y_test):.4f}")

# Visualize the tree
fig, ax = plt.subplots(figsize=(15, 8))
plot_tree(shallow_tree, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, ax=ax, fontsize=8)
plt.title('Decision Tree (max_depth=2)')
plt.tight_layout()
plt.savefig('ch14_decision_tree.png', dpi=150)
plt.show()

# ─── 2. Tree Depth vs Overfitting ──────────────
depths = range(1, 20)
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, 'bo-', label='Train Accuracy')
plt.plot(depths, test_scores, 'ro-', label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Depth vs Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch14_tree_depth.png', dpi=150)
plt.show()

# ─── 3. Random Forest ──────────────────────────
n_estimators_list = [1, 5, 10, 50, 100, 200]
rf_scores = []

for n in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    score = cross_val_score(rf, X_train, y_train, cv=5).mean()
    rf_scores.append(score)
    print(f"  RF ({n} trees): CV accuracy = {score:.4f}")

# ─── 4. Feature Importance ─────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n=== Feature Importances ===")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {iris.feature_names[idx]}: {importances[idx]:.4f}")

plt.figure(figsize=(8, 4))
plt.bar(range(len(importances)), importances[indices], color='steelblue')
plt.xticks(range(len(importances)),
           [iris.feature_names[i] for i in indices], rotation=45)
plt.title('Random Forest Feature Importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('ch14_feature_importance.png', dpi=150)
plt.show()

# ─── 5. Compare Models ─────────────────────────
models = {
    "Decision Tree (deep)": DecisionTreeClassifier(max_depth=None, random_state=42),
    "Decision Tree (depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("\n=== Model Comparison ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    test_acc = model.score(X_test, y_test)
    print(f"  {name}: CV={scores.mean():.4f}±{scores.std():.4f}, Test={test_acc:.4f}")

print("\n✓ Chapter 14 complete!")
```

---

## Exercises

### Easy
1. Train a decision tree and visualize it. What feature does it split on first?
2. What is Gini impurity of a node with 3 classes, each having equal probability?
3. Train a random forest with 10 trees. Compare accuracy to a single tree.

### Medium
4. Plot tree depth vs train/test accuracy. Where does overfitting begin?
5. Use feature importance to identify the top 2 features in a dataset.
6. Compare Random Forest vs Gradient Boosting on the same data.

### Advanced
7. Implement Gini impurity calculation from scratch and verify against sklearn.
8. Build a custom random forest that samples both rows and columns.

---

## Chapter Summary

```
• Trees split by maximizing information gain (minimizing impurity)
• Deep trees overfit → shallow trees underfit
• Random forests average many trees → reduce overfitting
• Feature importance tells you which features matter most
• Tree-based models excel on tabular data
```
