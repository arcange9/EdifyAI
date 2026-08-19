# Chapter 6 — Data Visualization with Matplotlib

## Learning Objectives

By the end of this chapter, you will:
- Create line charts, bar charts, scatter plots, and histograms
- Understand which chart type fits which data
- Build multi-panel dashboards
- Use visualization to understand data before feeding it to models

---

## Why This Matters

Before you train a model, you must *see* your data. Visualization reveals patterns, outliers, and problems that numbers alone hide. Every AI Engineer starts a project by plotting the data.

📌 **IMPORTANT:** A model trained on bad data produces bad predictions. Visualization is how you catch problems early.

---

## Concept Explanation

### Chart Types and When to Use Them

```
┌─────────────────────────────────────────────────────────────┐
│  CHART TYPE        USE WHEN...                    AI USE CASE│
├─────────────────────────────────────────────────────────────┤
│ Line chart        Showing change over time       Training  │
│                                                   loss curve│
│                                                             │
│ Bar chart         Comparing categories           Model      │
│                                                   accuracy  │
│                                                             │
│ Scatter plot      Showing relationships          Feature    │
│                   between two variables           correlation│
│                                                             │
│ Histogram         Showing distribution           Data       │
│                   of one variable                 spread    │
│                                                             │
│ Box plot          Showing outliers & quartiles   Outlier    │
│                                                   detection  │
│                                                             │
│ Heatmap           Showing correlation matrix    Feature     │
│                                                   selection  │
│                                                             │
│ Subplots          Multiple charts together      EDA         │
│                                                   dashboard  │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — Visualization for AI

```python
"""
Chapter 6 — Data Visualization with Matplotlib

pip install matplotlib numpy pandas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════
# SECTION 1: LINE CHARTS — Training Loss Curves
# ════════════════════════════════════════════════════════

# Simulate training loss over epochs
epochs = range(1, 51)
train_loss = [1.0 * (0.95 ** e) for e in epochs]
val_loss = [1.05 * (0.96 ** e) + 0.05 for e in epochs]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
plt.plot(epochs, val_loss, 'r--', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss — Overfitting Check')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch6_loss_curve.png', dpi=150)
plt.show()
print("✓ Loss curve saved")


# ════════════════════════════════════════════════════════
# SECTION 2: BAR CHARTS — Model Comparison
# ════════════════════════════════════════════════════════

models = ['Logistic\nReg', 'Decision\nTree', 'Random\nForest', 'KNN', 'SVM']
accuracies = [0.82, 0.78, 0.91, 0.85, 0.88]

plt.figure(figsize=(8, 5))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
bars = plt.bar(models, accuracies, color=colors, edgecolor='black')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.1%}', ha='center', fontweight='bold')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig('ch6_model_comparison.png', dpi=150)
plt.show()
print("✓ Model comparison saved")


# ════════════════════════════════════════════════════════
# SECTION 3: SCATTER PLOTS — Feature Relationships
# ════════════════════════════════════════════════════════

np.random.seed(42)
study_hours = np.random.uniform(1, 10, 100)
scores = 30 + 7 * study_hours + np.random.normal(0, 8, 100)

plt.figure(figsize=(10, 6))
plt.scatter(study_hours, scores, alpha=0.6, edgecolors='black')

# Add trend line
z = np.polyfit(study_hours, scores, 1)
p = np.poly1d(z)
plt.plot(study_hours, p(study_hours), 'r-', linewidth=2, label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')

plt.xlabel('Study Hours')
plt.ylabel('Test Score')
plt.title('Study Hours vs Test Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch6_scatter.png', dpi=150)
plt.show()
print("✓ Scatter plot saved")


# ════════════════════════════════════════════════════════
# SECTION 4: HISTOGRAMS — Data Distribution
# ════════════════════════════════════════════════════════

# Generate two distributions
normal_data = np.random.normal(75, 10, 1000)
skewed_data = np.random.exponential(20, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Normal distribution
axes[0].hist(normal_data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(normal_data.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {normal_data.mean():.1f}')
axes[0].set_title('Normal Distribution (Test Scores)')
axes[0].set_xlabel('Score')
axes[0].set_ylabel('Count')
axes[0].legend()

# Skewed distribution
axes[1].hist(skewed_data, bins=30, color='orange', edgecolor='black', alpha=0.7)
axes[1].axvline(skewed_data.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {skewed_data.mean():.1f}')
axes[1].set_title('Skewed Distribution (Response Times)')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('ch6_histograms.png', dpi=150)
plt.show()
print("✓ Histograms saved")


# ════════════════════════════════════════════════════════
# SECTION 5: SUBPLOTS — EDA Dashboard
# ════════════════════════════════════════════════════════

np.random.seed(42)
data = pd.DataFrame({
    "feature_1": np.random.randn(500),
    "feature_2": np.random.randn(500) * 2 + 1,
    "feature_3": np.random.randn(500) * 0.5 + 3,
    "target": np.random.choice([0, 1], 500)
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Scatter colored by class
mask_0 = data["target"] == 0
mask_1 = data["target"] == 1
axes[0, 0].scatter(data.loc[mask_0, "feature_1"], data.loc[mask_0, "feature_2"],
                   alpha=0.5, label="Class 0", s=10)
axes[0, 0].scatter(data.loc[mask_1, "feature_1"], data.loc[mask_1, "feature_2"],
                   alpha=0.5, label="Class 1", s=10, color='red')
axes[0, 0].set_title("Feature 1 vs Feature 2")
axes[0, 0].legend()

# Top-right: Distribution of target
axes[0, 1].hist(data.loc[mask_0, "feature_3"], bins=20, alpha=0.5, label="Class 0")
axes[0, 1].hist(data.loc[mask_1, "feature_3"], bins=20, alpha=0.5, label="Class 1")
axes[0, 1].set_title("Feature 3 Distribution by Class")
axes[0, 1].legend()

# Bottom-left: Correlation heatmap
corr = data[["feature_1", "feature_2", "feature_3", "target"]].corr()
im = axes[1, 0].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(len(corr.columns)))
axes[1, 0].set_yticks(range(len(corr.columns)))
axes[1, 0].set_xticklabels(corr.columns, rotation=45)
axes[1, 0].set_yticklabels(corr.columns)
# Add correlation values
for i in range(len(corr)):
    for j in range(len(corr)):
        axes[1, 0].text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center')
axes[1, 0].set_title("Correlation Heatmap")
plt.colorbar(im, ax=axes[1, 0])

# Bottom-right: Box plot
axes[1, 1].boxplot([data.loc[mask_0, "feature_2"], data.loc[mask_1, "feature_2"]],
                    labels=["Class 0", "Class 1"])
axes[1, 1].set_title("Feature 2 by Class (Box Plot)")
axes[1, 1].set_ylabel("Feature 2")

plt.suptitle("EDA Dashboard — AI Data Exploration", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ch6_dashboard.png', dpi=150)
plt.show()
print("✓ EDA Dashboard saved")


# ════════════════════════════════════════════════════════
# SECTION 6: PRACTICAL — Visualizing Model Performance
# ════════════════════════════════════════════════════════

# Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred, classes=["Negative", "Positive"]):
    """Plot a confusion matrix for binary classification."""
    from collections import Counter
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues')

    for i in range(2):
        for j in range(2):
            color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    fontsize=20, color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('ch6_confusion_matrix.png', dpi=150)
    plt.show()

    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"  Accuracy: {(tp+tn)/(tp+fp+tn+fn):.2%}")
    print(f"  Precision: {tp/(tp+fp) if tp+fp else 0:.2%}")
    print(f"  Recall: {tp/(tp+fn) if tp+fn else 0:.2%}")

# Demo
np.random.seed(42)
y_true = np.random.choice([0, 1], 100)
y_pred = np.random.choice([0, 1], 100, p=[0.6, 0.4])
plot_confusion_matrix(y_true, y_pred)

print("\n" + "=" * 50)
print("✓ Chapter 6 Visualization complete!")
print("=" * 50)
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Wrong chart type.* Don't use a pie chart for 20 categories. Don't use a line chart for categorical data.
2. *Not labeling axes.* Every chart needs x-label, y-label, and title.
3. *Too many colors.* Use color to convey information, not decoration.
4. *No grid.* Grids make charts easier to read. Always add `plt.grid(True, alpha=0.3)`.

---

## Mini-Project — EDA Dashboard

Pick any dataset (or generate one) and create a 4-panel dashboard:
1. Scatter plot of two features
2. Histogram of one feature
3. Bar chart of a categorical variable
4. Correlation heatmap

Save it as a single image and examine it.

---

## Exercises

### Easy
1. Plot a line chart showing y = x² for x from 0 to 10.
2. Create a bar chart with 5 categories and custom colors.
3. Make a histogram of 1000 random numbers from a normal distribution.

### Medium
4. Create a subplot with 2x2 grid showing 4 different chart types.
5. Add a trend line to a scatter plot using np.polyfit.
6. Color scatter points by their class label.

### Advanced
7. Build an interactive confusion matrix visualizer that accepts any binary classification results.
8. Create a function that automatically generates an EDA report (charts + statistics) for any DataFrame.

---

## Chapter Summary

```
┌──────────────────────────────────────────────────────────┐
│                   CHAPTER 6 KEY TAKEAWAYS                │
│                                                          │
│  • Always visualize data before training models          │
│  • Line: trends over time (loss curves)                  │
│  • Bar: comparing categories (model accuracy)            │
│  • Scatter: relationships (feature correlation)          │
│  • Histogram: distribution (data spread)                 │
│  • Heatmap: correlation matrix                           │
│  • Box plot: outliers and quartiles                      │
│  • Subplots: multi-chart dashboards for EDA             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Interview Questions

1. "How would you visualize model training progress?"
2. "What chart would you use to detect outliers?"
3. "How do you visualize a correlation matrix?"
4. "What is a confusion matrix and how do you interpret it?"

---

## Knowledge Check

1. When would you use a scatter plot vs a line chart?
2. What does a correlation heatmap show?
3. How do you create a figure with multiple subplots?
4. What does `plt.tight_layout()` do and why is it useful?
5. What chart shows the distribution of a single variable?
