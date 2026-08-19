# Chapter 8 — Mathematics for AI: Probability & Statistics

## Learning Objectives

By the end of this chapter, you will:
- Understand probability, distributions, and random variables
- Apply Bayes' theorem to AI problems
- Compute statistical measures (mean, variance, correlation)
- Understand hypothesis testing and confidence intervals
- See how statistics powers ML model evaluation

---

## Why This Matters

Machine learning *is* statistical learning. When a model predicts "this email is 85% likely to be spam," that's probability. When we evaluate a model's accuracy, that's statistics. When we split train/test data, that's sampling. You cannot do AI without statistics.

---

## Concept Explanation

### Probability Basics

```
┌──────────────────────────────────────────────────────────┐
│  PROBABILITY IN AI                                       │
│                                                          │
│  P(spam | contains "FREE") = ?                          │
│                                                          │
│  This means: "What's the probability this email is       │
│  spam, given that it contains the word FREE?"            │
│                                                          │
│  P(A)           = probability of event A                 │
│  P(A|B)         = probability of A given B happened      │
│  P(A ∩ B)       = probability of both A AND B            │
│  P(A ∪ B)       = probability of A OR B                  │
│                                                          │
│  Key rules:                                              │
│  0 ≤ P(A) ≤ 1    (probabilities are between 0 and 1)   │
│  P(certain) = 1   (a sure thing)                        │
│  P(impossible) = 0                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Bayes' Theorem — The Heart of ML

```
┌──────────────────────────────────────────────────────────┐
│  BAYES' THEOREM                                          │
│                                                          │
│              P(B|A) × P(A)                               │
│  P(A|B) = ─────────────────                              │
│                   P(B)                                   │
│                                                          │
│  In words:                                               │
│  "Probability of A given B =                             │
│   (Probability of B given A × Probability of A)          │
│   / Probability of B"                                    │
│                                                          │
│  AI Example:                                             │
│  P(spam | "FREE") = P("FREE" | spam) × P(spam) / P("FREE")│
│                                                          │
│  Symbols explained:                                      │
│  P(A)   = prior (what we knew before evidence)          │
│  P(B|A) = likelihood (how likely is evidence given A)   │
│  P(A|B) = posterior (what we now believe after evidence) │
│  P(B)   = evidence (total probability of the evidence)  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Distributions

```
┌─────────────────────────────────────────────────────────────┐
│  KEY DISTRIBUTIONS IN AI                                    │
│                                                             │
│  Normal (Gaussian):                                         │
│       ▲                                                     │
│      /│\        Most common distribution in nature        │
│     / │ \       Model weights often init from normal      │
│    /  │  \      Test scores, heights, errors               │
│   /   │   \                                               │
│  ───────────→                                              │
│  -3  0  3                                                  │
│                                                             │
│  Uniform:                                                   │
│       ┌───┐                                                │
│       │   │      All values equally likely                 │
│       │   │      Random number generation                   │
│       └───┘                                                │
│  ───────────→                                              │
│  0         1                                               │
│                                                             │
│  Bernoulli (Binary):                                        │
│         │                                                   │
│         █ █     0 or 1 only                                │
│         █ █     Spam/not spam, click/no-click              │
│         █ █                                               │
│         0 1                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Statistical Measures

```
┌─────────────────────────────────────────────────────────────┐
│  MEASURES                                                 │
│                                                            │
│  Mean (average):    x̄ = (x₁ + x₂ + ... + xₙ) / n        │
│                     "What's the typical value?"            │
│                                                            │
│  Variance:          σ² = Σ(xᵢ - x̄)² / n                │
│                     "How spread out is the data?"          │
│                                                            │
│  Standard Deviation: σ = √σ²                              │
│                     "Same as variance but in same units"   │
│                                                            │
│  Correlation (r):   -1 ≤ r ≤ 1                            │
│                     +1 = perfect positive correlation      │
│                      0 = no correlation                    │
│                     -1 = perfect negative correlation      │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Code — Probability & Statistics

```python
"""
Chapter 8 — Probability & Statistics for AI
pip install numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ════════════════════════════════════════════════════════
# SECTION 1: BASIC PROBABILITY
# ════════════════════════════════════════════════════════

print("=== Basic Probability ===")

# Simulate rolling a die 10,000 times
np.random.seed(42)
rolls = np.random.randint(1, 7, 10000)

# Probability of each face
for face in range(1, 7):
    prob = np.mean(rolls == face)
    print(f"  P({face}) = {prob:.4f} (expected: 0.1667)")

# Probability of even number
p_even = np.mean(rolls % 2 == 0)
print(f"\n  P(even) = {p_even:.4f} (expected: 0.5000)")

# Joint probability: rolling a 6 AND it's even
p_six_and_even = np.mean((rolls == 6) & (rolls % 2 == 0))
print(f"  P(6 AND even) = {p_six_and_even:.4f} (expected: 0.1667)")


# ════════════════════════════════════════════════════════
# SECTION 2: BAYES' THEOREM — SPAM CLASSIFIER
# ════════════════════════════════════════════════════════

print("\n=== Bayes' Theorem — Spam Classification ===")

# Known probabilities from training data
p_spam = 0.4              # 40% of emails are spam
p_ham = 0.6               # 60% are not spam (ham)
p_free_given_spam = 0.8   # 80% of spam contains "FREE"
p_free_given_ham = 0.1    # 10% of non-spam contains "FREE"

# Bayes: P(spam | "FREE") = P("FREE" | spam) * P(spam) / P("FREE")
p_free = p_free_given_spam * p_spam + p_free_given_ham * p_ham
p_spam_given_free = (p_free_given_spam * p_spam) / p_free

print(f"  P(spam) = {p_spam}")
print(f"  P('FREE' | spam) = {p_free_given_spam}")
print(f"  P('FREE' | ham) = {p_free_given_ham}")
print(f"  P('FREE') = {p_free:.4f}")
print(f"  P(spam | 'FREE') = {p_spam_given_free:.4f}")
print(f"  → An email with 'FREE' is {p_spam_given_free:.1%} likely to be spam")


# ════════════════════════════════════════════════════════
# SECTION 3: DISTRIBUTIONS
# ════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Normal distribution
normal = np.random.normal(75, 10, 10000)
axes[0, 0].hist(normal, bins=50, density=True, alpha=0.7, color='blue')
axes[0, 0].axvline(normal.mean(), color='red', linestyle='--', label=f'Mean: {normal.mean():.1f}')
axes[0, 0].set_title('Normal Distribution (μ=75, σ=10)')
axes[0, 0].legend()

# Uniform distribution
uniform = np.random.uniform(0, 100, 10000)
axes[0, 1].hist(uniform, bins=50, density=True, alpha=0.7, color='green')
axes[0, 1].set_title('Uniform Distribution [0, 100)')

# Bernoulli (binary)
bernoulli = np.random.choice([0, 1], 10000, p=[0.7, 0.3])
axes[1, 0].hist(bernoulli, bins=2, density=True, alpha=0.7, color='orange')
axes[1, 0].set_xticks([0.25, 0.75])
axes[1, 0].set_xticklabels(['0', '1'])
axes[1, 0].set_title('Bernoulli Distribution (p=0.3)')

# Bimodal (two groups)
bimodal = np.concatenate([np.random.normal(30, 5, 5000),
                          np.random.normal(70, 5, 5000)])
axes[1, 1].hist(bimodal, bins=50, density=True, alpha=0.7, color='purple')
axes[1, 1].set_title('Bimodal Distribution (Two Groups)')

plt.suptitle('Probability Distributions in AI', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ch8_distributions.png', dpi=150)
plt.show()
print("\n✓ Distributions saved")


# ════════════════════════════════════════════════════════
# SECTION 4: STATISTICAL MEASURES
# ════════════════════════════════════════════════════════

print("\n=== Statistical Measures ===")

# Create a dataset
data = np.array([65, 78, 82, 55, 90, 72, 88, 63, 95, 70,
                 80, 85, 60, 75, 92, 68, 73, 87, 58, 77])

# Central tendency
mean = data.mean()
median = np.median(data)
mode_result = Counter(data).most_common(1)
mode = mode_result[0][0] if len(set(data)) < len(data) else "No mode"

print(f"  Data: {sorted(data)}")
print(f"  Mean: {mean:.2f}")
print(f"  Median: {median}")
print(f"  Mode: {mode}")

# Spread
variance = data.var()
std_dev = data.std()
data_range = data.max() - data.min()

print(f"\n  Variance: {variance:.2f}")
print(f"  Std Dev: {std_dev:.2f}")
print(f"  Range: {data_range}")
print(f"  Min: {data.min()}, Max: {data.max()}")

# Quartiles
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
print(f"\n  Q1 (25th percentile): {q1}")
print(f"  Q3 (75th percentile): {q3}")
print(f"  IQR (interquartile range): {iqr}")

# Outliers (values beyond 1.5 × IQR from quartiles)
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"  Outliers: {outliers if len(outliers) > 0 else 'None'}")


# ════════════════════════════════════════════════════════
# SECTION 5: CORRELATION
# ════════════════════════════════════════════════════════

print("\n=== Correlation ===")

# Two correlated features
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_scores = np.array([55, 60, 68, 72, 78, 82, 85, 88, 92, 95])
video_games = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0])  # Negative correlation

corr_study_score = np.corrcoef(study_hours, test_scores)[0, 1]
corr_games_score = np.corrcoef(video_games, test_scores)[0, 1]

print(f"  corr(study_hours, test_scores) = {corr_study_score:.4f}")
print(f"  corr(video_games, test_scores) = {corr_games_score:.4f}")
print(f"  → More study = higher scores (positive correlation)")
print(f"  → More games = lower scores (negative correlation)")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(study_hours, test_scores, c='green', s=100)
ax[0].set_xlabel('Study Hours')
ax[0].set_ylabel('Test Score')
ax[0].set_title(f'Positive Correlation (r={corr_study_score:.2f})')
ax[0].grid(True, alpha=0.3)

ax[1].scatter(video_games, test_scores, c='red', s=100)
ax[1].set_xlabel('Video Game Hours')
ax[1].set_ylabel('Test Score')
ax[1].set_title(f'Negative Correlation (r={corr_games_score:.2f})')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch8_correlation.png', dpi=150)
plt.show()
print("✓ Correlation plots saved")


# ════════════════════════════════════════════════════════
# SECTION 6: HYPOTHESIS TESTING (Simplified)
# ════════════════════════════════════════════════════════

print("\n=== Hypothesis Testing ===")

# Scenario: Did a new teaching method improve scores?
old_method = np.random.normal(72, 8, 50)  # Old method scores
new_method = np.random.normal(78, 8, 50)  # New method scores

# Null hypothesis (H0): No difference between methods
# Alternative (H1): New method is better

from scipy import stats
t_stat, p_value = stats.ttest_ind(new_method, old_method)

print(f"  Old method mean: {old_method.mean():.2f}")
print(f"  New method mean: {new_method.mean():.2f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant? {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

if p_value < 0.05:
    print("  → Reject H0: The new method significantly improved scores")
else:
    print("  → Cannot reject H0: No significant difference")

print("\n" + "=" * 50)
print("✓ Chapter 8 complete!")
print("=" * 50)
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Correlation ≠ causation.* Just because two variables are correlated doesn't mean one causes the other.
2. *Ignoring the p-value threshold.* The standard is p < 0.05, but this means 1 in 20 significant results could be by chance.
3. *Not understanding what "significant" means.* It means "unlikely to happen by chance," not "important" or "large effect."
4. *Using mean when median is better.* Mean is affected by outliers; median is robust.

---

## Mini-Project — Naive Bayes Spam Classifier

```python
"""
Mini-Project: Naive Bayes Spam Classifier
Uses Bayes' theorem to classify emails as spam or not spam.
"""
import numpy as np
from collections import Counter

# Training data
spam_emails = [
    "WIN free iPhone now",
    "Get free money click here",
    "Limited offer buy now discount",
    "FREE gift card claim now",
    "You won a prize click to claim",
]
ham_emails = [
    "Meeting tomorrow at 3pm",
    "Project update attached",
    "Lunch next week?",
    "Please review the document",
    "Thanks for the help yesterday",
]

def train_naive_bayes(spam, ham):
    """Train a simple Naive Bayes classifier."""
    # Tokenize all emails
    spam_words = " ".join(spam).lower().split()
    ham_words = " ".join(ham).lower().split()

    # Count word frequencies
    spam_counts = Counter(spam_words)
    ham_counts = Counter(ham_words)

    # Total words
    total_spam = sum(spam_counts.values())
    total_ham = sum(ham_counts.values())
    vocab = set(spam_words + ham_words)

    # Calculate probabilities with Laplace smoothing
    word_probs = {}
    for word in vocab:
        p_word_spam = (spam_counts.get(word, 0) + 1) / (total_spam + len(vocab))
        p_word_ham = (ham_counts.get(word, 0) + 1) / (total_ham + len(vocab))
        word_probs[word] = {"spam": p_word_spam, "ham": p_word_ham}

    p_spam = len(spam) / (len(spam) + len(ham))
    p_ham = len(ham) / (len(spam) + len(ham))

    return {"words": word_probs, "p_spam": p_spam, "p_ham": p_ham,
            "vocab": vocab}

def classify(email, model):
    """Classify an email as spam or ham."""
    words = email.lower().split()
    log_p_spam = np.log(model["p_spam"])
    log_p_ham = np.log(model["p_ham"])

    for word in words:
        if word in model["vocab"]:
            log_p_spam += np.log(model["words"][word]["spam"])
            log_p_ham += np.log(model["words"][word]["ham"])

    if log_p_spam > log_p_ham:
        return "SPAM", np.exp(log_p_spam) / (np.exp(log_p_spam) + np.exp(log_p_ham))
    else:
        return "HAM", np.exp(log_p_ham) / (np.exp(log_p_spam) + np.exp(log_p_ham))

# Train and test
model = train_naive_bayes(spam_emails, ham_emails)

test_emails = [
    "Get your free prize now",
    "Can we schedule a meeting",
    "Click here for free money",
    "Please send the project files",
]

print("=== Naive Bayes Spam Classifier ===\n")
for email in test_emails:
    label, prob = classify(email, model)
    print(f"  '{email}'")
    print(f"  → {label} (confidence: {prob:.1%})\n")
```

---

## Exercises

### Easy
1. Simulate flipping a fair coin 10,000 times. Verify P(heads) ≈ 0.5.
2. Calculate mean, median, and std of [10, 20, 30, 40, 50].
3. Create a scatter plot showing positive correlation between two variables.

### Medium
4. Implement Bayes' theorem from scratch and calculate P(disease | positive test) given P(disease)=0.01, P(positive|disease)=0.99, P(positive|no disease)=0.05.
5. Calculate the correlation between hours of sleep and test scores (generate your own data).
6. Perform a t-test comparing two groups of data.

### Advanced
7. Build a complete Naive Bayes text classifier that can be trained on any labeled text dataset.
8. Implement a confidence interval calculator for model accuracy scores.

---

## Chapter Summary

```
┌──────────────────────────────────────────────────────────┐
│                   CHAPTER 8 KEY TAKEAWAYS                │
│                                                          │
│  • Probability: P(A) between 0 and 1                     │
│  • Bayes: P(A|B) = P(B|A)×P(A) / P(B)                   │
│  • Normal dist: bell curve, mean ± std                  │
│  • Mean = average, Median = middle, Mode = most common │
│  • Variance/Std = how spread out data is                │
│  • Correlation: -1 to 1, measures linear relationship   │
│  • Correlation ≠ Causation                              │
│  • p < 0.05 = statistically significant                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Interview Questions

1. "Explain Bayes' theorem with an AI example."
2. "What's the difference between correlation and causation?"
3. "How would you handle class imbalance in a dataset?"
4. "What does a p-value tell you?"

---

## Knowledge Check

1. If P(A) = 0.3 and P(B) = 0.4, and A and B are independent, what is P(A and B)?
2. What does a standard deviation of 0 mean?
3. If two variables have a correlation of -0.9, what does that mean?
4. When would you use median instead of mean?
5. What is the null hypothesis in a hypothesis test?
