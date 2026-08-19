# Chapter 12 — Supervised Learning: Regression

## Learning Objectives

- Understand linear and polynomial regression
- Implement gradient descent for regression from scratch
- Use scikit-learn for regression tasks
- Evaluate regression models properly

---

## Why This Matters

Regression is predicting a *number* — house prices, temperatures, stock prices, test scores. It's the most common ML task in business. Understanding regression deeply means understanding how models learn from data.

---

## Concept Explanation

### Linear Regression

```
┌──────────────────────────────────────────────────────────┐
│  LINEAR REGRESSION                                       │
│                                                          │
│  Model: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b              │
│                                                          │
│  y = predicted value                                     │
│  x₁, x₂, ... = features (inputs)                       │
│  w₁, w₂, ... = weights (learned)                        │
│  b = bias/intercept (learned)                            │
│                                                          │
│  The model learns w and b to minimize:                    │
│  MSE = (1/n) × Σ(y_true - y_pred)²                     │
│                                                          │
│  Visual:                                                 │
│                                                          │
│  y ↑                                                     │
│    │      ●                                              │
│    │    ●   /  ← regression line                         │
│    │  ●   /                                              │
│    │ ●  /                                                │
│    │● /                                                  │
│    └────────────→ x                                      │
│                                                          │
│  The line minimizes total vertical distance              │
│  from all points (least squares)                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Polynomial Regression

```
Linear:     y = w·x + b
Polynomial: y = w₁·x + w₂·x² + w₃·x³ + b

When data follows a curve, linear can't capture it.
Polynomial adds powers of x as new features.
```

### Loss Function: MSE

```
MSE = (1/n) × Σ(y_true - y_pred)²

y_true = actual value
y_pred = predicted value
n = number of samples

Why squared? 
1. Penalizes large errors more than small ones
2. Mathematically smooth (differentiable everywhere)
3. Always positive (never negative)

RMSE = √MSE (same units as the target variable)
```

---

## Code — Regression from Scratch + scikit-learn

```python
"""
Chapter 12 — Regression
pip install numpy scikit-learn matplotlib pandas
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ════════════════════════════════════════════════════════
# SECTION 1: LINEAR REGRESSION FROM SCRATCH
# ════════════════════════════════════════════════════════

print("=== Linear Regression from Scratch ===\n")

class LinearRegressionScratch:
    """Linear regression trained with gradient descent."""
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y

            # Gradients
            dw = (2 / n_samples) * X.T @ error
            db = (2 / n_samples) * np.sum(error)

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = np.mean(error ** 2)
            self.history.append(loss)

    def predict(self, X):
        return X @ self.w + self.b

# Test it
np.random.seed(42)
X = np.random.randn(200, 3)
true_w = np.array([3, -1, 2])
y = X @ true_w + 5 + np.random.normal(0, 0.5, 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegressionScratch(lr=0.1, epochs=500)
model.fit(X_train, y_train)

print(f"  True weights:     {true_w}")
print(f"  Learned weights:  {model.w}")
print(f"  True bias: 5, Learned bias: {model.b:.4f}")
print(f"  Train R²: {r2_score(y_train, model.predict(X_train)):.4f}")
print(f"  Test R²:  {r2_score(y_test, model.predict(X_test)):.4f}")

# ════════════════════════════════════════════════════════
# SECTION 2: SCIKIT-LEARN REGRESSION
# ════════════════════════════════════════════════════════

print("\n=== scikit-learn Regression ===\n")

# Compare sklearn's implementation
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
print(f"  sklearn weights: {sk_model.coef_}")
print(f"  sklearn bias:    {sk_model.intercept_:.4f}")
print(f"  Test R²:          {sk_model.score(X_test, y_test):.4f}")

# ════════════════════════════════════════════════════════
# SECTION 3: POLYNOMIAL REGRESSION
# ════════════════════════════════════════════════════════

print("\n=== Polynomial Regression ===\n")

# Non-linear data
np.random.seed(42)
X_curve = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
y_curve = 0.5 * X_curve.ravel()**3 - 2 * X_curve.ravel()**2 + \
          1 * X_curve.ravel() + 2 + np.random.normal(0, 1, 100)

degrees = [1, 3, 5, 15]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, degree in enumerate(degrees):
    ax = axes[idx // 2, idx % 2]
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_curve)
    model = LinearRegression()
    model.fit(X_poly, y_curve)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y_curve, y_pred)

    ax.scatter(X_curve, y_curve, s=10, alpha=0.5, label='Data')
    ax.plot(X_curve, y_pred, 'r-', linewidth=2, label=f'Degree {degree} (R²={r2:.3f})')
    ax.set_title(f'Polynomial Degree {degree}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if degree == 1:
        print(f"  Degree {degree}: R²={r2:.4f} — UNDERFITTING (too simple)")
    elif degree == 15:
        print(f"  Degree {degree}: R²={r2:.4f} — OVERFITTING (memorizing noise)")
    else:
        print(f"  Degree {degree}: R²={r2:.4f} — GOOD FIT")

plt.suptitle('Polynomial Regression: Underfitting vs Overfitting', fontsize=14)
plt.tight_layout()
plt.savefig('ch12_polynomial.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════
# SECTION 4: REGULARIZATION (Ridge, Lasso)
# ════════════════════════════════════════════════════════

print("\n=== Regularization ===\n")

# High-degree polynomial with regularization
degree = 15
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X_curve)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y_curve, test_size=0.3, random_state=42)

models = {
    "No regularization": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1),
}

for name, model in models.items():
    model.fit(X_train_p, y_train_p)
    train_r2 = r2_score(y_train_p, model.predict(X_train_p))
    test_r2 = r2_score(y_test_p, model.predict(X_test_p))
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    print(f"  {name}:")
    print(f"    Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"    Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")

print("""
  REGULARIZATION SUMMARY:
  ━━━━━━━━━━━━━━━━━━━━━
  Ridge (L2): shrinks weights toward zero (but never exactly zero)
              Good when many features contribute slightly
  Lasso (L1): can set weights to exactly zero (feature selection)
              Good when few features are important
  Both prevent overfitting by penalizing large weights
""")

print("=" * 50)
print("✓ Chapter 12 complete!")
print("=" * 50)
```

---

## Mini-Project — House Price Predictor

```python
"""
Project 2: House Price Prediction System
Combines preprocessing + regression
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# Generate data
np.random.seed(42)
n = 500
X = pd.DataFrame({
    "sqft": np.random.uniform(800, 4000, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 4, n),
    "age": np.random.randint(0, 50, n),
    "garage": np.random.randint(0, 3, n),
})
y = (150 * X["sqft"] + 20000 * X["bedrooms"] + 15000 * X["bathrooms"]
     - 400 * X["age"] + 8000 * X["garage"] + np.random.normal(0, 15000, n))

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(f"R²: {r2_score(y_test, predictions):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, predictions):,.0f}")

# Predict a new house
new_house = pd.DataFrame({"sqft": [2000], "bedrooms": [3], "bathrooms": [2],
                          "age": [10], "garage": [2]})
predicted_price = pipeline.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:,.0f}")
```

---

## Exercises

### Easy
1. Train a linear regression model on any dataset. Print the coefficients and intercept.
2. What happens to R² when you add a completely random feature?
3. Calculate MSE, RMSE, and MAE for a set of predictions.

### Medium
4. Compare degree 1, 3, and 10 polynomial regression. Plot all three.
5. Train Ridge regression with alpha=0.01, 1.0, and 100. How does alpha affect coefficients?
6. Build a regression pipeline with scaling + polynomial features + Ridge.

### Advanced
7. Implement Ridge regression from scratch (add L2 penalty to gradient descent).
8. Build a house price predictor that selects the best polynomial degree automatically.

---

## Chapter Summary

```
• Linear: y = Wx + b (minimizes MSE)
• Polynomial: adds x², x³ as features for curves
• High degree = overfitting, low degree = underfitting
• Ridge (L2): shrinks weights, never zero
• Lasso (L1): can zero out weights (feature selection)
• Always evaluate with R², RMSE, and MAE
```

---

## Interview Questions

1. "What's the difference between Ridge and Lasso regression?"
2. "How do you choose the polynomial degree for a regression?"
3. "What does R² = 0.85 mean?"
4. "Why might a complex model perform worse than a simple one?"
