# Chapter 5 — Pandas: Data Manipulation

## Learning Objectives

By the end of this chapter, you will:
- Understand DataFrames and Series — the core Pandas data structures
- Load, explore, and clean real-world datasets
- Filter, group, merge, and transform data
- Prepare data for machine learning models

---

## Why This Matters

Real-world AI doesn't start with clean arrays. It starts with messy CSV files, databases, and APIs. Pandas is how you turn messy data into the clean NumPy arrays that models need.

📌 **IMPORTANT:** 70% of an AI Engineer's job is data preparation. Pandas is your primary tool for this.

---

## Concept Explanation

### The Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  FROM RAW DATA TO AI MODEL                                   │
│                                                             │
│  CSV/JSON/API ──→ PANDAS ──→ NUMPY ──→ MODEL                │
│                   │           │          │                  │
│                 Load         Array      Train               │
│                 Clean        Format     Predict              │
│                 Filter                                                    │
│                 Transform                                                │
│                                                             │
│  Pandas handles the messy middle part.                      │
│  NumPy handles the efficient math.                          │
│  The model handles the learning.                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Data Structures

```
┌─────────────────────────────────────────────────────────────┐
│  PANDAS DATA STRUCTURES                                     │
│                                                             │
│  Series (1D):  One column of data                           │
│  ┌─────┬──────┐                                            │
│  │ idx │ value│                                             │
│  ├─────┼──────┤                                            │
│  │  0  │  45  │                                             │
│  │  1  │  78  │                                             │
│  │  2  │  92  │                                             │
│  └─────┴──────┘                                            │
│                                                             │
│  DataFrame (2D): Table of data (like a spreadsheet)        │
│  ┌─────┬───────┬──────┬───────┐                           │
│  │ idx │ name  │ age  │ score │                            │
│  ├─────┼───────┼──────┼───────┤                           │
│  │  0  │ Alice │  20  │  85   │                            │
│  │  1  │ Bob   │  22  │  92   │                            │
│  │  2  │ Carol │  19  │  78   │                            │
│  └─────┴───────┴──────┴───────┘                           │
│                                                             │
│  A DataFrame is a collection of Series.                    │
│  Think of it as a dictionary of columns.                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — Pandas for AI

```python
"""
Chapter 5 — Pandas: Data Manipulation

pip install pandas numpy
"""

import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════
# SECTION 1: CREATING DATAFRAMES
# ════════════════════════════════════════════════════════

# From a dictionary
data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "age": [20, 22, 19, 21, 23],
    "math_score": [85, 72, 90, 68, 95],
    "science_score": [78, 80, 88, 75, 92],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)
print("=== Student Dataset ===")
print(df)
print(f"\nShape: {df.shape}")  # (5, 5)
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")

# From a list of dictionaries (common when data comes from APIs)
api_data = [
    {"name": "Frank", "age": 20, "math_score": 82, "science_score": 85, "passed": True},
    {"name": "Grace", "age": 21, "math_score": 91, "science_score": 88, "passed": True},
]
df2 = pd.DataFrame(api_data)
print(f"\nFrom API format:\n{df2}")


# ════════════════════════════════════════════════════════
# SECTION 2: LOADING AND SAVING DATA
# ════════════════════════════════════════════════════════

# Save to CSV (then reload)
df.to_csv("students.csv", index=False)
print("\n✓ Saved to students.csv")

# Load from CSV
loaded_df = pd.read_csv("students.csv")
print(f"Loaded from CSV:\n{loaded_df.head()}")

# Other formats you'll use:
# pd.read_json("data.json")
# pd.read_excel("data.xlsx")
# pd.read_sql("SELECT * FROM table", connection)


# ════════════════════════════════════════════════════════
# SECTION 3: EXPLORING DATA (EDA — Exploratory Data Analysis)
# ════════════════════════════════════════════════════════

# First look at the data
print("\n=== EXPLORATORY DATA ANALYSIS ===")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nLast 2 rows:\n{df.tail(2)}")
print(f"\nDataset info:")
df.info()
print(f"\nStatistical summary:\n{df.describe()}")
print(f"\nNumber of unique values per column:")
print(df.nunique())
print(f"\nMissing values per column:")
print(df.isnull().sum())


# ════════════════════════════════════════════════════════
# SECTION 4: SELECTING AND FILTERING
# ════════════════════════════════════════════════════════

# Select a single column (returns a Series)
ages = df["age"]
print(f"\nAges:\n{ages}")

# Select multiple columns
scores = df[["name", "math_score", "science_score"]]
print(f"\nScores:\n{scores}")

# ─── Filtering with conditions ─────────────────
# This is how you filter data in AI preprocessing

# Students who passed
passed = df[df["passed"] == True]
print(f"\nStudents who passed:\n{passed}")

# Students with math score > 80
high_math = df[df["math_score"] > 80]
print(f"\nHigh math scores:\n{high_math}")

# Multiple conditions (use & for AND, | for OR)
# Note: must use parentheses around each condition!
smart_young = df[(df["math_score"] > 85) & (df["age"] < 22)]
print(f"\nYoung students with high math scores:\n{smart_young}")

# ─── loc and iloc ──────────────────────────────
# loc: select by label
# iloc: select by position

print(f"\nRow by label (loc):\n{df.loc[0]}")  # First row
print(f"\nRow by position (iloc):\n{df.iloc[1]}")  # Second row
print(f"\nSpecific cell: {df.loc[2, 'name']}")  # Row 2, name column
print(f"\nFirst 3 rows, 2 columns:\n{df.iloc[:3, :2]}")  # Slice


# ════════════════════════════════════════════════════════
# SECTION 5: ADDING, MODIFYING, AND DROPPING COLUMNS
# ════════════════════════════════════════════════════════

# Add a new column (feature engineering!)
df["average_score"] = (df["math_score"] + df["science_score"]) / 2
print(f"\nWith average:\n{df}")

# Add a categorical column
df["grade"] = df["average_score"].apply(
    lambda x: "A" if x >= 90 else "B" if x >= 80 else "C" if x >= 70 else "F"
)
print(f"\nWith grades:\n{df}")

# Modify an existing column
df["math_score"] = df["math_score"].astype(float)
print(f"\nMath score dtype: {df['math_score'].dtype}")

# Drop a column
df_dropped = df.drop("passed", axis=1)
print(f"\nWithout 'passed' column:\n{df_dropped.columns.tolist()}")

# Drop rows
df_filtered = df.drop(0)  # Drop first row
print(f"After dropping row 0: {df_filtered.shape}")


# ════════════════════════════════════════════════════════
# SECTION 6: GROUPING AND AGGREGATION
# ════════════════════════════════════════════════════════

# Create a larger dataset for grouping
np.random.seed(42)
n = 100
big_data = pd.DataFrame({
    "student_id": range(1, n + 1),
    "classroom": np.random.choice(["A", "B", "C"], n),
    "subject": np.random.choice(["Math", "Science", "English"], n),
    "score": np.random.randint(50, 100, n),
    "study_hours": np.random.uniform(1, 10, n).round(1)
})
big_data["passed"] = big_data["score"] >= 60

print(f"\n=== Large Dataset ({len(big_data)} rows) ===")
print(big_data.head())

# Group by classroom and calculate mean score
by_classroom = big_data.groupby("classroom")["score"].mean()
print(f"\nAverage score by classroom:\n{by_classroom}")

# Group by subject
by_subject = big_data.groupby("subject")["score"].agg(["mean", "min", "max", "std"])
print(f"\nScore stats by subject:\n{by_subject}")

# Group by multiple columns
multi_group = big_data.groupby(["classroom", "subject"])["score"].mean()
print(f"\nAverage by classroom AND subject:\n{multi_group}")

# Count passed per classroom
pass_rates = big_data.groupby("classroom")["passed"].agg(["count", "sum", "mean"])
pass_rates.columns = ["total", "passed", "pass_rate"]
print(f"\nPass rates by classroom:\n{pass_rates}")


# ════════════════════════════════════════════════════════
# SECTION 7: HANDLING MISSING DATA
# ════════════════════════════════════════════════════════

# Create data with missing values
messy_data = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol", "David", "Eve", "Frank"],
    "math": [85, np.nan, 90, 68, np.nan, 95],
    "science": [78, 80, np.nan, 75, 92, 88],
    "age": [20, 22, 19, np.nan, 23, 21]
})

print("\n=== Handling Missing Data ===")
print(f"Data with NaN:\n{messy_data}")
print(f"\nMissing count:\n{messy_data.isnull().sum()}")

# Strategy 1: Drop rows with missing values
dropped = messy_data.dropna()
print(f"\nAfter dropping NaN rows: {dropped.shape}")

# Strategy 2: Fill with mean (common in AI)
filled = messy_data.copy()
filled["math"] = filled["math"].fillna(filled["math"].mean())
filled["science"] = filled["science"].fillna(filled["science"].mean())
filled["age"] = filled["age"].fillna(filled["age"].median())
print(f"\nAfter filling NaN:\n{filled}")

# Strategy 3: Forward fill (use previous value)
ffill = messy_data.fillna(method="ffill")
print(f"\nForward fill:\n{ffill}")

# Strategy 4: Fill with a specific value
zero_filled = messy_data.fillna(0)
print(f"\nZero fill:\n{zero_filled}")


# ════════════════════════════════════════════════════════
# SECTION 8: MERGING AND JOINING
# ════════════════════════════════════════════════════════

# In AI, you often combine data from multiple sources
students = pd.DataFrame({
    "student_id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "classroom": ["A", "B", "A", "C", "B"]
})

grades = pd.DataFrame({
    "student_id": [1, 2, 3, 4, 5],
    "math": [85, 72, 90, 68, 95],
    "science": [78, 80, 88, 75, 92]
})

# Merge on student_id (like SQL JOIN)
merged = students.merge(grades, on="student_id")
print(f"\n=== Merged Data ===\n{merged}")

# Different types of joins:
# inner: only matching rows (default)
# left: all rows from left, NaN for missing right
# right: all rows from right, NaN for missing left
# outer: all rows from both, NaN where missing

# Concatenate (stack DataFrames)
more_students = pd.DataFrame({
    "student_id": [6, 7],
    "name": ["Frank", "Grace"],
    "classroom": ["A", "C"]
})
all_students = pd.concat([students, more_students], ignore_index=True)
print(f"\nConcatenated:\n{all_students}")


# ════════════════════════════════════════════════════════
# SECTION 9: PREPARING DATA FOR MACHINE LEARNING
# ════════════════════════════════════════════════════════

print("\n=== Preparing Data for ML ===")

# Create a dataset that needs preprocessing
ml_data = pd.DataFrame({
    "age": [18, 20, 19, 22, 21, 23, 20, 19, 25, 18],
    "study_hours": [2, 5, 8, 3, 7, 10, 4, 6, 9, 1],
    "gender": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
    "classroom": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
    "score": [55, 78, 92, 60, 88, 95, 65, 82, 90, 50]
})

# Step 1: Handle categorical variables
# One-hot encoding (convert categories to 0/1 columns)
encoded = pd.get_dummies(ml_data, columns=["gender", "classroom"],
                         drop_first=False)
print(f"\nAfter one-hot encoding:\n{encoded.columns.tolist()}")
print(encoded.head())

# Step 2: Feature scaling (will cover in detail in Chapter 10)
# Normalize: (x - min) / (max - min) → [0, 1]
for col in ["age", "study_hours"]:
    min_val = encoded[col].min()
    max_val = encoded[col].max()
    encoded[col + "_normalized"] = (encoded[col] - min_val) / (max_val - min_val)

# Step 3: Separate features (X) and target (y)
X = encoded.drop("score", axis=1)
y = encoded["score"]

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Feature columns: {X.columns.tolist()}")

# Step 4: Train/test split (we'll use scikit-learn for this in Ch 11)
# For now, manual split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTraining: {X_train.shape}, Testing: {X_test.shape}")

print("\n" + "=" * 50)
print("✓ Chapter 5 Pandas complete!")
print("=" * 50)
```

---

## Real-World Example

```
┌──────────────────────────────────────────────────────────────┐
│  PANDAS IN A REAL AI PIPELINE                                 │
│                                                              │
│  # Load customer data from a database                         │
│  df = pd.read_sql("SELECT * FROM customers", conn)          │
│                                                              │
│  # Clean: remove duplicates, handle missing values           │
│  df = df.drop_duplicates()                                  │
│  df["income"] = df["income"].fillna(df["income"].median())  │
│                                                              │
│  # Feature engineering                                        │
│  df["income_per_age"] = df["income"] / df["age"]            │
│  df["high_value"] = df["income"] > df["income"].quantile(0.9)│
│                                                              │
│  # Encode categorical variables                               │
│  df = pd.get_dummies(df, columns=["country", "plan"])      │
│                                                              │
│  # Prepare for model                                         │
│  X = df.drop("churned", axis=1)                             │
│  y = df["churned"]                                           │
│                                                              │
│  # Convert to NumPy for the model                             │
│  X_array = X.values  # Now it's a NumPy array                │
│  y_array = y.values                                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Chained indexing.* `df["col"]["row"]` is dangerous. Use `df.loc["row", "col"]` instead.
2. *Forgetting `inplace=True`.* Most Pandas operations return a new DataFrame and don't modify the original. Use `df = df.drop(...)` or `df.drop(..., inplace=True)`.
3. *Not handling missing data.* Always check `df.isnull().sum()` before training models.
4. *Using `.values` blindly.* Convert to `.values` (NumPy array) only when ready for the model. Keep as DataFrame for data manipulation.

---

## Engineering Perspective

🧠 **THINK LIKE AN ENGINEER:**

```
Data preprocessing checklist before any ML model:
1. □ Check data types (df.dtypes) — are they correct?
2. □ Check missing values (df.isnull().sum()) — how to handle?
3. □ Check duplicates (df.duplicated().sum()) — remove?
4. □ Check distributions (df.describe()) — any outliers?
5. □ Encode categorical variables (pd.get_dummies())
6. □ Scale numerical features (if needed)
7. □ Split into train/test (before preprocessing to avoid leakage)
```

---

## Mini-Project — Data Analysis Report

Build a complete data analysis pipeline:

```python
"""
Mini-Project: Data Analysis Report
Analyze a synthetic student dataset and generate insights.

project/
├── analysis.py    ← This file
└── report.txt      ← Generated by the program
"""
import pandas as pd
import numpy as np

# ─── Generate synthetic data ──────────────────
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "student_id": range(1, n + 1),
    "name": [f"Student_{i}" for i in range(1, n + 1)],
    "classroom": np.random.choice(["A", "B", "C", "D"], n),
    "subject": np.random.choice(["Math", "Science", "English", "History"], n),
    "score": np.random.normal(75, 15, n).clip(0, 100),
    "study_hours": np.random.uniform(0.5, 12, n),
    "attendance": np.random.uniform(0.5, 1.0, n),
})

# Add some missing values
df.loc[df.sample(10).index, "score"] = np.nan
df.loc[df.sample(5).index, "study_hours"] = np.nan

# ─── Analysis Functions ───────────────────────
def generate_report(df):
    """Generate a complete data analysis report."""
    report = []
    report.append("=" * 60)
    report.append("  STUDENT DATA ANALYSIS REPORT")
    report.append("=" * 60)

    # Basic info
    report.append(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")
    report.append(f"Columns: {', '.join(df.columns.tolist())}")

    # Missing data
    report.append(f"\n--- Missing Data ---")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            report.append(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")

    # Handle missing
    df_clean = df.copy()
    df_clean["score"] = df_clean["score"].fillna(df_clean["score"].mean())
    df_clean["study_hours"] = df_clean["study_hours"].fillna(
        df_clean["study_hours"].median()
    )

    # Statistics
    report.append(f"\n--- Score Statistics ---")
    stats = df_clean["score"].describe()
    for stat, value in stats.items():
        report.append(f"  {stat}: {value:.2f}")

    # By classroom
    report.append(f"\n--- Average Score by Classroom ---")
    by_class = df_clean.groupby("classroom")["score"].agg(["mean", "std", "count"])
    for classroom, row in by_class.iterrows():
        report.append(f"  Class {classroom}: avg={row['mean']:.1f}, "
                      f"std={row['std']:.1f}, n={int(row['count'])}")

    # Correlation
    report.append(f"\n--- Correlation with Score ---")
    numeric_cols = ["study_hours", "attendance", "score"]
    corr = df_clean[numeric_cols].corr()["score"]
    for col, value in corr.items():
        if col != "score":
            report.append(f"  {col}: {value:.3f}")

    # Top students
    report.append(f"\n--- Top 5 Students ---")
    top5 = df_clean.nlargest(5, "score")[["name", "score", "study_hours"]]
    for _, row in top5.iterrows():
        report.append(f"  {row['name']}: {row['score']:.1f} "
                      f"(studied {row['study_hours']:.1f}h)")

    report.append("\n" + "=" * 60)
    return "\n".join(report)

# Generate and save report
report_text = generate_report(df)
print(report_text)

with open("report.txt", "w") as f:
    f.write(report_text)
print("\n✓ Report saved to report.txt")
```

---

## Exercises

### Easy
1. Load any CSV file and print its shape, columns, and data types.
2. Filter a DataFrame to show only rows where a numeric column is above its mean.
3. Add a new column that is the sum of two existing columns.

### Medium
4. Group a dataset by two categorical columns and calculate the mean and count.
5. Handle missing values three different ways (drop, fill with mean, fill with median) and compare results.
6. Perform one-hot encoding on a categorical column and explain how many new columns it creates.

### Advanced
7. Write a function that automatically detects and reports data quality issues (missing values, duplicates, outliers) for any DataFrame.
8. Merge two DataFrames on a non-unique key and handle the resulting duplicates.

---

## Chapter Summary

```
┌──────────────────────────────────────────────────────────┐
│                   CHAPTER 5 KEY TAKEAWAYS                │
│                                                          │
│  • DataFrames are tables; Series are columns              │
│  • EDA: .head(), .info(), .describe(), .isnull()         │
│  • Filter with boolean conditions: df[df["col"] > x]    │
│  • Group by: df.groupby("col")["target"].mean()         │
│  • Handle missing data: dropna, fillna                   │
│  • Merge: df1.merge(df2, on="key")                       │
│  • One-hot encoding: pd.get_dummies()                     │
│  • 70% of AI work is data prep — master Pandas           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Interview Questions

1. "How do you handle missing data in a dataset?"
2. "What's the difference between loc and iloc?"
3. "How would you merge two datasets with different keys?"
4. "What is one-hot encoding and why is it needed for ML?"

---

## Knowledge Check

1. What does `df.describe()` return?
2. How do you select rows where column A > 50 AND column B < 30?
3. What's the difference between `fillna(0)` and `dropna()`?
4. How do you convert a categorical column to numeric for ML?
5. Why should you split train/test BEFORE preprocessing?
