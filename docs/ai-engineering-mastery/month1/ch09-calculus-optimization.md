# Chapter 9 — Mathematics for AI: Calculus & Optimization

## Learning Objectives

By the end of this chapter, you will:
- Understand derivatives and why they matter for AI
- See how gradient descent works (the engine of all ML training)
- Understand partial derivatives and the chain rule
- Implement gradient descent from scratch

---

## Why This Matters

Every neural network is trained using calculus. When PyTorch does `loss.backward()`, it's computing derivatives using the chain rule. When a model "learns," it's following the gradient downhill. If you don't understand calculus, you can't understand how models learn.

---

## Concept Explanation

### Derivatives — The Rate of Change

```
┌──────────────────────────────────────────────────────────┐
│  WHAT IS A DERIVATIVE?                                   │
│                                                          │
│  A derivative tells you HOW FAST something is changing   │
│  at a specific point.                                    │
│                                                          │
│  Visual: the slope of the tangent line at a point.        │
│                                                          │
│        y                                                  │
│        │     ╱ ← steep slope (large derivative)           │
│        │    ╱                                              │
│        │   ╱  ← medium slope                               │
│        │  ╱                                                │
│        │ ╱   ← shallow slope (small derivative)           │
│        │╱                                                   │
│        └──────────────→ x                                   │
│        0                                                   │
│                                                          │
│  f'(x) = lim   [f(x+h) - f(x)] / h                       │
│         h→0                                                │
│                                                          │
│  In plain English: "At point x, how much does y change    │
│  when x changes by a tiny amount?"                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Why Derivatives Matter for AI

```
┌──────────────────────────────────────────────────────────────┐
│  THE TRAINING PROCESS (simplified)                           │
│                                                              │
│  1. Model makes predictions                                  │
│  2. We compute the LOSS (how wrong the predictions are)     │
│  3. We compute the DERIVATIVE of loss with respect to       │
│     each weight (gradient)                                   │
│  4. We adjust weights in the OPPOSITE direction of the     │
│     gradient (going downhill)                               │
│  5. Repeat until loss is minimized                          │
│                                                              │
│  The derivative tells us WHICH WAY to adjust each weight    │
│  to reduce the error.                                        │
│                                                              │
│  loss = f(weights)                                           │
│  gradient = d(loss) / d(weight)                              │
│  new_weight = old_weight - learning_rate × gradient        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Gradient Descent — The Heart of Machine Learning

```
┌──────────────────────────────────────────────────────────────┐
│  GRADIENT DESCENT VISUALIZED                                 │
│                                                              │
│  Loss                                                        │
│   │                                                          │
│   │ ●     ← Start here (random weights)                     │
│   │  \                                                        │
│   │   ●   ← Take a step downhill (in the gradient direction) │
│   │    \                                                      │
│   │     ●  ← Another step                                     │
│   │      \                                                    │
│   │       ●  ← Getting closer to minimum                      │
│   │        \                                                  │
│   │         ●  ← Near the minimum (low loss)                  │
│   │          ●  ← Minimum! Best weights found!               │
│   └──────────────────────────────────→ Weights                │
│                                                              │
│  The "step size" is the LEARNING RATE:                      │
│    Too large → overshoot the minimum (bounce around)        │
│    Too small → take forever to reach the minimum            │
│    Just right → smooth descent to minimum                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Partial Derivatives

When a function has multiple inputs, the partial derivative tells you how the output changes when you change ONE input while keeping the others fixed.

```
┌──────────────────────────────────────────────────────────┐
│  PARTIAL DERIVATIVES                                      │
│                                                          │
│  f(x, y) = x² + 3xy + y²                               │
│                                                          │
│  ∂f/∂x = 2x + 3y   (derivative w.r.t. x, treat y as const)│
│  ∂f/∂y = 3x + 2y   (derivative w.r.t. y, treat x as const)│
│                                                          │
│  In neural networks:                                     │
│  Loss depends on ALL weights (thousands to billions)    │
│  Partial derivative of loss w.r.t. each weight =        │
│  "how should I change THIS weight to reduce loss?"       │
│                                                          │
│  The gradient = vector of all partial derivatives        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### The Chain Rule (How Backpropagation Works)

```
┌──────────────────────────────────────────────────────────────┐
│  THE CHAIN RULE                                              │
│                                                              │
│  If y = f(g(x)), then:                                       │
│     dy/dx = df/dg × dg/dx                                   │
│                                                              │
│  In neural networks:                                         │
│  input → layer1 → layer2 → layer3 → loss                    │
│                                                              │
│  To find d(loss)/d(weight_in_layer1):                        │
│  d(loss)/d(w1) = d(loss)/d(layer3) × d(layer3)/d(layer2)   │
│                × d(layer2)/d(layer1) × d(layer1)/d(w1)      │
│                                                              │
│  This CHAIN of derivatives is why it's called               │
│  "backpropagation" — we go backwards through the chain.    │
│                                                              │
│  PyTorch does this automatically with autograd.              │
│  But you need to understand it conceptually.                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Gradient Descent from Scratch

```python
"""
Chapter 9 — Calculus & Optimization for AI
pip install numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════
# SECTION 1: NUMERICAL DERIVATIVES
# ════════════════════════════════════════════════════════

print("=== Numerical Derivatives ===")

def derivative(f, x, h=1e-5):
    """Compute the derivative of f at point x numerically.
    Uses the definition: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# Test on f(x) = x² (derivative should be 2x)
def f_square(x):
    return x ** 2

# Check derivative at several points
for x in [0, 1, 2, 3, 5]:
    numerical_deriv = derivative(f_square, x)
    exact_deriv = 2 * x  # d/dx(x²) = 2x
    print(f"  f'({x}) = {numerical_deriv:.4f} (exact: {exact_deriv})")

# Visualize derivative as tangent line
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_vals = np.linspace(-3, 3, 100)
y_vals = x_vals ** 2

axes[0].plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²')
x_point = 1.5
y_point = x_point ** 2
slope = derivative(f_square, x_point)
tangent_x = np.linspace(x_point - 1, x_point + 1, 50)
tangent_y = y_point + slope * (tangent_x - x_point)
axes[0].plot(tangent_x, tangent_y, 'r--', linewidth=2,
             label=f"Tangent at x={x_point} (slope={slope:.1f})")
axes[0].plot(x_point, y_point, 'ro', markersize=8)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Derivative = Slope of Tangent Line')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ════════════════════════════════════════════════════════
# SECTION 2: GRADIENT DESCENT — 1D
# ════════════════════════════════════════════════════════

print("\n=== Gradient Descent (1D) ===")

# Let's minimize f(x) = x² + 5*sin(x)
def f_1d(x):
    return x**2 + 5 * np.sin(x)

def grad_1d(x):
    """Derivative: 2x + 5*cos(x)"""
    return 2 * x + 5 * np.cos(x)

# Gradient descent
def gradient_descent_1d(f, grad, start, lr=0.1, iterations=100):
    """Find the minimum of f using gradient descent."""
    x = start
    history = [x]
    losses = [f(x)]

    for i in range(iterations):
        gradient = grad(x)
        x = x - lr * gradient  # Move opposite to gradient
        history.append(x)
        losses.append(f(x))

    return x, history, losses

# Run it
best_x, history, losses = gradient_descent_1d(
    f_1d, grad_1d, start=5.0, lr=0.1, iterations=50
)

print(f"  Starting point: x = 5.0")
print(f"  Final: x = {best_x:.6f}")
print(f"  Minimum value: f({best_x:.4f}) = {f_1d(best_x):.6f}")

# Visualize
x_vals = np.linspace(-5, 6, 200)
y_vals = f_1d(x_vals)

axes[1].plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x² + 5sin(x)')
axes[1].plot(history, [f_1d(h) for h in history], 'ro-', markersize=4,
            label='Gradient descent path')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title('Gradient Descent Finding Minimum')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch9_gradient_descent_1d.png', dpi=150)
plt.show()
print("✓ Gradient descent 1D saved")


# ════════════════════════════════════════════════════════
# SECTION 3: GRADIENT DESCENT — LINEAR REGRESSION
# ════════════════════════════════════════════════════════

print("\n=== Gradient Descent for Linear Regression ===")

# The model: y = w * x + b
# The loss: MSE = (1/n) * Σ(y_pred - y_true)²
# Gradients: dL/dw = (2/n) * Σ x * (y_pred - y_true)
#            dL/db = (2/n) * Σ (y_pred - y_true)

# Training data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.1, 16.0, 17.8, 20.1])

# True relationship: y ≈ 2x (slope ≈ 2, intercept ≈ 0)
# Let's see if gradient descent can find this!

def train_linear_regression(X, y, lr=0.01, epochs=100):
    """Train linear regression with gradient descent."""
    n = len(X)
    w = 0.0  # Initialize weight (slope)
    b = 0.0  # Initialize bias (intercept)
    history = []

    for epoch in range(epochs):
        # Forward pass: predictions
        y_pred = w * X + b

        # Compute loss (MSE)
        loss = np.mean((y_pred - y) ** 2)

        # Compute gradients (partial derivatives)
        dw = (2 / n) * np.sum(X * (y_pred - y))
        db = (2 / n) * np.sum(y_pred - y)

        # Update weights (go opposite to gradient)
        w = w - lr * dw
        b = b - lr * db

        history.append({"epoch": epoch, "loss": loss, "w": w, "b": b})

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

    return w, b, history

w, b, history = train_linear_regression(X, y, lr=0.01, epochs=100)

print(f"\n  Final model: y = {w:.4f}x + {b:.4f}")
print(f"  True relationship: y ≈ 2x + 0")

# Visualize training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
losses = [h["loss"] for h in history]
axes[0].plot(losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training Loss Curve')
axes[0].grid(True, alpha=0.3)

# Data and fitted line
axes[1].scatter(X, y, color='blue', s=100, label='Data')
x_line = np.linspace(0, 11, 100)
y_line = w * x_line + b
axes[1].plot(x_line, y_line, 'r-', linewidth=2,
           label=f'Learned: y = {w:.2f}x + {b:.2f}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Linear Regression Result')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch9_linear_regression_gd.png', dpi=150)
plt.show()
print("✓ Linear regression with gradient descent saved")


# ════════════════════════════════════════════════════════
# SECTION 4: LEARNING RATE EFFECTS
# ════════════════════════════════════════════════════════

print("\n=== Learning Rate Comparison ===")

# Compare different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for idx, lr in enumerate(learning_rates):
    w, b, history = train_linear_regression(X, y, lr=lr, epochs=50)
    losses = [h["loss"] for h in history]
    axes[idx].plot(losses, linewidth=2)
    axes[idx].set_title(f'lr = {lr}')
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Loss')
    axes[idx].grid(True, alpha=0.3)
    print(f"  lr={lr}: final loss={losses[-1]:.4f}")

plt.suptitle('Effect of Learning Rate on Training', fontsize=14)
plt.tight_layout()
plt.savefig('ch9_learning_rates.png', dpi=150)
plt.show()
print("\n✓ Learning rate comparison saved")

print("""
  LEARNING RATE ANALYSIS:
  ━━━━━━━━━━━━━━━━━━━━━━
  lr=0.001: Too slow — barely learning
  lr=0.01:  Good — smooth convergence
  lr=0.1:   Fast but might overshoot slightly
  lr=0.5:   Too large — diverging or bouncing
""")


# ════════════════════════════════════════════════════════
# SECTION 5: OPTIMIZERS — BEYOND VANILLA GD
# ════════════════════════════════════════════════════════

print("=== Optimizer Comparison ===")

def gradient_descent_momentum(X, y, lr=0.01, epochs=100, momentum=0.9):
    """GD with momentum — accelerates in consistent gradient directions."""
    n = len(X)
    w, b = 0.0, 0.0
    vw, vb = 0.0, 0.0  # Velocity (momentum)
    losses = []

    for epoch in range(epochs):
        y_pred = w * X + b
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)

        dw = (2 / n) * np.sum(X * (y_pred - y))
        db = (2 / n) * np.sum(y_pred - y)

        # Momentum: velocity accumulates gradient
        vw = momentum * vw - lr * dw
        vb = momentum * vb - lr * db
        w += vw
        b += vb

    return w, b, losses

def gradient_descent_adam(X, y, lr=0.01, epochs=100, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer — adaptive learning rates for each parameter."""
    n = len(X)
    w, b = 0.0, 0.0
    mw, mb = 0.0, 0.0  # First moment (mean)
    vw, vb = 0.0, 0.0  # Second moment (variance)
    losses = []

    for epoch in range(epochs):
        y_pred = w * X + b
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)

        dw = (2 / n) * np.sum(X * (y_pred - y))
        db = (2 / n) * np.sum(y_pred - y)

        # First moment
        mw = beta1 * mw + (1 - beta1) * dw
        mb = beta1 * mb + (1 - beta1) * db

        # Second moment
        vw = beta2 * vw + (1 - beta2) * dw**2
        vb = beta2 * vb + (1 - beta2) * db**2

        # Bias correction
        m_w_hat = mw / (1 - beta1**(epoch + 1))
        m_b_hat = mb / (1 - beta1**(epoch + 1))
        v_w_hat = vw / (1 - beta2**(epoch + 1))
        v_b_hat = vb / (1 - beta2**(epoch + 1))

        # Update
        w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    return w, b, losses

# Compare optimizers
_, _, losses_vanilla = train_linear_regression(X, y, lr=0.01, epochs=100)
_, _, losses_momentum = gradient_descent_momentum(X, y, lr=0.01, epochs=100)
_, _, losses_adam = gradient_descent_adam(X, y, lr=0.01, epochs=100)

plt.figure(figsize=(10, 5))
plt.plot(losses_vanilla, label='Vanilla GD', linewidth=2)
plt.plot(losses_momentum, label='GD + Momentum', linewidth=2)
plt.plot(losses_adam, label='Adam', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('ch9_optimizers.png', dpi=150)
plt.show()

print("  Adam converges fastest because it adapts learning rate per parameter.")
print("  Momentum helps escape shallow valleys and accelerates in consistent directions.")
print("  Vanilla GD is the baseline — simple but slow.")

print("\n" + "=" * 50)
print("✓ Chapter 9 complete!")
print("=" * 50)
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Learning rate too high.* The loss will explode or oscillate. Always start small and increase.
2. *Not normalizing data before gradient descent.* Features on different scales cause gradients to be dominated by large-scale features.
3. *Forgetting to zero gradients between iterations.* In PyTorch, call `optimizer.zero_grad()` before each step.
4. *Using gradient descent on non-convex functions without care.* Can get stuck in local minima.

---

## Engineering Perspective

🧠 **THINK LIKE AN ENGINEER:**

```
Choosing an optimizer:
1. Adam is the default choice — works well in most cases
2. SGD with momentum is good for fine-tuning
3. Use learning rate schedulers (reduce LR over time)
4. Monitor training loss AND validation loss
5. If loss plateaus, try: lower LR, add momentum, check data quality

Math you actually need:
1. Derivatives → understanding how models learn
2. Chain rule → understanding backpropagation
3. Partial derivatives → understanding multi-parameter optimization
4. That's it! You don't need to solve differential equations
```

---

## Mini-Project — Multi-Variable Gradient Descent

```python
"""
Mini-Project: Optimize a function with multiple variables.
Find the minimum of f(x, y) = x² + y² + xy using gradient descent.
"""
import numpy as np
import matplotlib.pyplot as plt

def f_2d(x, y):
    return x**2 + y**2 + x*y

def grad_2d(x, y):
    df_dx = 2*x + y
    df_dy = 2*y + x
    return np.array([df_dx, df_dy])

# Gradient descent
x, y = 5.0, -3.0  # Starting point
lr = 0.1
history = [(x, y)]

for epoch in range(100):
    gradient = grad_2d(x, y)
    x -= lr * gradient[0]
    y -= lr * gradient[1]
    history.append((x, y))

print(f"Minimum found at: ({x:.6f}, {y:.6f})")
print(f"Minimum value: {f_2d(x, y):.6f}")
print(f"(True minimum: (0, 0) with value 0)")

# Visualize
history = np.array(history)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
x_range = np.linspace(-6, 6, 100)
y_range = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = f_2d(X, Y)
ax.contour(X, Y, Z, levels=20, cmap='viridis')
ax.plot(history[:, 0], history[:, 1], 'ro-', markersize=3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gradient Descent in 2D')
plt.tight_layout()
plt.savefig('ch9_gd_2d.png', dpi=150)
plt.show()
```

---

## Exercises

### Easy
1. Compute the derivative of f(x) = 3x² + 2x + 1. Verify with numerical derivative.
2. Run gradient descent on f(x) = x² starting from x=10. How many steps to reach x≈0?
3. What happens if you set the learning rate to 1.0 for f(x) = x²?

### Medium
4. Implement gradient descent for f(x) = x⁴ - 3x³ + 2. Find all local minima.
5. Compare convergence of vanilla GD, momentum, and Adam on a quadratic function.
6. Implement gradient descent with a learning rate scheduler (halve LR every 50 epochs).

### Advanced
7. Implement backpropagation for a 2-layer neural network using only NumPy and the chain rule.
8. Implement a mini-batch gradient descent that processes data in chunks of 32 samples.

---

## Chapter Summary

```
┌──────────────────────────────────────────────────────────┐
│                   CHAPTER 9 KEY TAKEAWAYS                │
│                                                          │
│  • Derivatives = rate of change = slope of tangent        │
│  • Gradient = vector of partial derivatives             │
│  • Gradient descent: move opposite to gradient          │
│  • Learning rate controls step size                      │
│    - Too high: diverges                                  │
│    - Too low: slow                                       │
│    - Just right: smooth convergence                      │
│  • Chain rule: how backpropagation works                 │
│  • Optimizers: Adam > Momentum > Vanilla GD              │
│  • PyTorch's autograd computes derivatives automatically │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Interview Questions

1. "Explain gradient descent in your own words."
2. "What's the chain rule and why is it important for neural networks?"
3. "How does the learning rate affect training? What happens if it's too large?"
4. "What's the difference between Adam and standard gradient descent?"

---

## Knowledge Check

1. What does a derivative tell you about a function at a specific point?
2. In gradient descent, why do we subtract the gradient instead of adding it?
3. What is the chain rule used for in neural networks?
4. Name three factors that affect how fast gradient descent converges.
5. Why does Adam often outperform vanilla gradient descent?
