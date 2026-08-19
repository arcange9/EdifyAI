# Chapter 10 — Data Preprocessing & Feature Engineering

## Learning Objectives

- Master data cleaning: handle missing values, duplicates, outliers
- Normalize and standardize features for ML
- Encode categorical variables (one-hot, label, ordinal)
- Engineer new features that improve model performance
- Use PCA for dimensionality reduction

---

## Why This Matters

```
┌──────────────────────────────────────────────────────────┐
│  THE DATA PIPELINE                                       │
│                                                          │
│  Raw Data → Clean → Transform → Engineer → Model        │
│     60%      10%      15%       10%      5%             │
│                                                          │
│  85% of an AI Engineer's job happens BEFORE the model.  │
│  The quality of your data determines the quality of     │
│  your AI — more than any model choice ever could.        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Concept Explanation

### The Preprocessing Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  COMPLETE PREPROCESSING WORKFLOW                            │
│                                                              │
│  1. LOAD          →  Read CSV, JSON, SQL, API                │
│                                                              │
│  2. INSPECT       →  shape, dtypes, describe(), isnull()    │
│                                                              │
│  3. CLEAN         →  Remove duplicates                      │
│                      Handle missing values                  │
│                      Fix data types                          │
│                      Remove outliers                         │
│                                                              │
│  4. TRANSFORM     →  Normalize / Standardize                │
│                      Encode categories                       │
│                      Handle dates/times                      │
│                                                              │
│  5. ENGINEER      →  Create new features                     │
│                      Combine features                        │
│                      Extract information                    │
│                      Select important features              │
│                                                              │
│  6. SPLIT         →  Train / Validation / Test               │
│                      (BEFORE preprocessing to avoid leakage) │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

📌 **IMPORTANT:** Always split data into train/test BEFORE fitting preprocessing transforms (like scaling). If you scale before splitting, information from the test set leaks into training — this is called *data leakage* and gives falsely optimistic results.

### Normalization vs Standardization

```
┌─────────────────────────────────────────────────────────────┐
│  NORMALIZATION (Min-Max Scaling)                           │
│                                                             │
│  x_new = (x - min) / (max - min)                           │
│                                                             │
│  Result: all values between 0 and 1                        │
│  Use when: you know the data is bounded                    │
│  Example: pixel values 0-255 → 0.0-1.0                     │
│                                                             │
│  STANDARDIZATION (Z-score)                                  │
│                                                             │
│  x_new = (x - mean) / std                                  │
│                                                             │
│  Result: mean = 0, std = 1                                 │
│  Use when: data has outliers or unknown bounds             │
│  Example: features with different scales (age, income)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Encoding Categorical Variables

```
┌─────────────────────────────────────────────────────────────┐
│  ENCODING METHODS                                           │
│                                                             │
│  ONE-HOT ENCODING                                           │
│  Color: ["red", "blue", "green"]                           │
│  → red: [1,0,0], blue: [0,1,0], green: [0,0,1]           │
│  Use when: categories have no order (nominal)              │
│  Warning: creates many columns for many categories         │
│                                                             │
│  LABEL ENCODING                                             │
│  Size: ["small", "medium", "large"]                        │
│  → small: 0, medium: 1, large: 2                          │
│  Use when: categories have order (ordinal)                  │
│                                                             │
│  TARGET ENCODING                                            │
│  City → replace with average target value for that city    │
│  Use when: many categories and they correlate with target  │
│  Warning: can cause overfitting                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — Complete Preprocessing Pipeline

```python
"""
Chapter 10 — Data Preprocessing & Feature Engineering
pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════
# CREATE MESSY REAL-WORLD DATA
# ════════════════════════════════════════════════════════

np.random.seed(42)
n = 500
data = pd.DataFrame({
    "age": np.random.normal(25, 8, n).clip(15, 60),
    "income": np.random.lognormal(10, 0.5, n),  # Skewed income
    "study_hours": np.random.uniform(0, 12, n),
    "city": np.random.choice(["Kigali", "Lagos", "Nairobi", "Accra"], n),
    "education": np.random.choice(["high_school", "bachelor", "master", "phd"], n),
    "score": np.random.normal(70, 15, n).clip(0, 100)
})

# Add messiness
data.loc[data.sample(20).index, "income"] = np.nan
data.loc[data.sample(10).index, "age"] = np.nan
data.loc[data.sample(5).index, "study_hours"] = np.nan
data = data.append({"age": 200, "income": 999999, "study_hours": 50,
                    "city": "Kigali", "education": "phd", "score": 100},
                   ignore_index=True)  # Outlier row

# ════════════════════════════════════════════════════════
# STEP 1: SPLIT FIRST (avoid data leakage!)
# ════════════════════════════════════════════════════════

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# ════════════════════════════════════════════════════════
# STEP 2: CLEAN
# ════════════════════════════════════════════════════════

print("\n=== Cleaning ===")

# Remove duplicates
train_data = train_data.drop_duplicates()

# Handle missing values
for df in [train_data, test_data]:
    df["income"] = df["income"].fillna(train_data["income"].median())
    df["age"] = df["age"].fillna(train_data["age"].median())
    df["study_hours"] = df["study_hours"].fillna(train_data["study_hours"].median())

# Remove outliers (beyond 3 std from mean)
for col in ["age", "income", "study_hours"]:
    mean, std = train_data[col].mean(), train_data[col].std()
    lower, upper = mean - 3*std, mean + 3*std
    train_data = train_data[(train_data[col] >= lower) & (train_data[col] <= upper)]

print(f"After cleaning: Train={len(train_data)}")

# ════════════════════════════════════════════════════════
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ════════════════════════════════════════════════════════

print("\n=== Encoding ===")

# One-hot for city (nominal — no order)
train_data = pd.get_dummies(train_data, columns=["city"], prefix="city")
test_data = pd.get_dummies(test_data, columns=["city"], prefix="city")
# Align columns (test might miss some categories)
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Label encoding for education (ordinal — has order)
edu_order = {"high_school": 0, "bachelor": 1, "master": 2, "phd": 3}
train_data["education"] = train_data["education"].map(edu_order)
test_data["education"] = test_data["education"].map(edu_order)

print(f"Columns after encoding: {train_data.columns.tolist()}")

# ════════════════════════════════════════════════════════
# STEP 4: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════

print("\n=== Feature Engineering ===")

for df in [train_data, test_data]:
    # Ratio feature
    df["income_per_age"] = df["income"] / df["age"]
    # Interaction feature
    df["age_x_study"] = df["age"] * df["study_hours"]
    # Binned feature
    df["age_group"] = pd.cut(df["age"], bins=[15, 20, 25, 30, 100],
                              labels=[0, 1, 2, 3]).astype(int)

print(f"Final columns: {train_data.columns.tolist()}")

# ════════════════════════════════════════════════════════
# STEP 5: SCALE NUMERICAL FEATURES
# ════════════════════════════════════════════════════════

print("\n=== Scaling ===")

numeric_cols = ["age", "income", "study_hours", "income_per_age", "age_x_study"]

scaler = StandardScaler()
# Fit on TRAIN only, transform both
train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])

print(f"Train means (should be ~0): {train_data[numeric_cols].mean().values}")
print(f"Train stds (should be ~1):  {train_data[numeric_cols].std().values}")

# ════════════════════════════════════════════════════════
# STEP 6: PCA — Dimensionality Reduction
# ════════════════════════════════════════════════════════

print("\n=== PCA ===")

# Separate target
X_train = train_data.drop("score", axis=1)
y_train = train_data["score"]

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_pca = pca.fit_transform(X_train)

print(f"Original features: {X_train.shape[1]}")
print(f"After PCA: {X_train_pca.shape[1]} components")
print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total variance retained: {sum(pca.explained_variance_ratio_):.4f}")

# Visualize PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(pca.explained_variance_ratio_)+1),
            pca.explained_variance_ratio_, color='steelblue')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('PCA: Variance Explained by Each Component')
axes[0].grid(True, alpha=0.3)

# Show cumulative
cumsum = np.cumsum(pca.explained_variance_ratio_)
axes[1].plot(range(1, len(cumsum)+1), cumsum, 'bo-', linewidth=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('PCA: Cumulative Variance')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch10_pca.png', dpi=150)
plt.show()
print("✓ PCA visualization saved")

print("\n" + "=" * 50)
print("✓ Chapter 10 complete!")
print("=" * 50)
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Data leakage.* Fitting scaler on all data (including test) before splitting. Always fit on train only.
2. *One-hot encoding without alignment.* Test set may have different categories. Use `reindex` to align.
3. *Not handling outliers.* Outliers skew mean-based scaling. Use robust methods or remove them.
4. *Creating too many features.* More features ≠ better. Can cause overfitting (curse of dimensionality).

---

## Mini-Project — House Price Data Pipeline

Build a complete preprocessing pipeline for a house price dataset:
1. Generate synthetic house data (sqft, bedrooms, bathrooms, location, age, price)
2. Add missing values and outliers
3. Clean, encode, engineer features (price_per_sqft, total_rooms)
4. Scale and split
5. Show before/after statistics

---

## Exercises

### Easy
1. Normalize [10, 20, 30, 40, 50] to [0, 1] range using min-max scaling.
2. One-hot encode ["cat", "dog", "cat", "bird"] manually.
3. Standardize [100, 200, 300] using z-score normalization.

### Medium
4. Write a function that automatically detects and handles missing values (fill with mean for numeric, mode for categorical).
5. Build a preprocessing pipeline class that chains: scaling → encoding → PCA.
6. Demonstrate data leakage: show how fitting scaler on all data inflates test accuracy.

### Advanced
7. Implement PCA from scratch using NumPy eigenvalue decomposition.
8. Build a feature selection function that removes low-variance and highly-correlated features.

---

## Chapter Summary

```
CLEAN → ENCODE → ENGINEER → SCALE → SPLIT
• Split before scaling (no leakage!)
• One-hot for nominal, label for ordinal
• Standardize when scales differ
• Engineer features that capture relationships
• PCA reduces dimensions while keeping variance
```

---

## Interview Questions

1. "What is data leakage and how do you prevent it?"
2. "When would you use normalization vs standardization?"
3. "How does PCA work and when would you use it?"
4. "What is feature engineering and why does it matter?"

---

## Knowledge Check

1. Why must you split data before fitting a scaler?
2. What's the difference between one-hot and label encoding?
3. What does PCA's "explained variance ratio" tell you?
4. Name three ways to handle missing values.
5. What is the "curse of dimensionality"?
