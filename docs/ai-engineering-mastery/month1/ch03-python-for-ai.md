# Chapter 3 — Python for AI: The Essentials

## Learning Objectives

By the end of this chapter, you will:
- Master the Python fundamentals needed for AI Engineering
- Understand data structures (lists, dicts, sets, tuples) used in AI
- Write functions, classes, and modules for AI applications
- Handle files and errors like a professional engineer

---

## Why This Matters

Python is the *lingua franca* of AI. Nearly every AI tool, library, and framework is built in or for Python. If your Python is weak, everything else in this book will be a struggle.

📌 **IMPORTANT:** You don't need to be a Python expert to start. This chapter teaches you the specific Python skills that AI Engineers use daily.

---

## Concept Explanation

### Python in the AI Stack

```
┌─────────────────────────────────────────────────────┐
│              YOUR AI APPLICATION                     │
├─────────────────────────────────────────────────────┤
│  FastAPI  │  RAG Pipeline  │  Agent Logic  │  Tests  │
├─────────────────────────────────────────────────────┤
│  LangChain │  Transformers │  PyTorch │  Scikit-learn│
├─────────────────────────────────────────────────────┤
│              NumPy │  Pandas │  Matplotlib            │
├─────────────────────────────────────────────────────┤
│                    PYTHON                            │
└─────────────────────────────────────────────────────┘

Everything above is built on Python. A strong Python foundation
supports everything else.
```

### Core Concepts You Need

1. Variables and types (int, float, str, bool)
2. Data structures (list, dict, set, tuple)
3. Control flow (if/elif/else, for, while)
4. Functions (def, args, kwargs, return)
5. Classes and objects (OOP basics)
6. File I/O (reading/writing files)
7. Error handling (try/except)
8. Modules and imports
9. List comprehensions
10. f-strings and string manipulation

---

## Visual Explanation — Data Structures in AI

```
┌──────────────────────────────────────────────────────────┐
│  HOW AI DATA STRUCTURES MAP TO PYTHON                     │
│                                                          │
│  AI CONCEPT              PYTHON TYPE     EXAMPLE          │
│  ─────────────────────────────────────────────────────── │
│  A single feature value  float           0.75           │
│  A label                 int / str        1 or "spam"    │
│  A feature vector        list/np.array    [0.1, 0.5, 0.9]│
│  A dataset row           dict/tuple       {"x": 5, "y": 3}│
│  A dataset               list of dicts    [{...}, {...}] │
│  Unique vocabulary       set             {"the", "cat"}  │
│  Model config            dict            {"lr": 0.01}   │
│  Training logs           list            [0.9, 0.8, ...]│
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Code — Complete Python AI Toolkit

```python
"""
Chapter 3 — Python for AI: The Essentials
Covers all Python fundamentals needed for AI Engineering.

No external libraries required — pure Python.
"""

# ════════════════════════════════════════════════════════
# SECTION 1: VARIABLES AND TYPES
# ════════════════════════════════════════════════════════

# In AI, we work with numbers and text constantly
accuracy = 0.95          # float — model accuracy
num_samples = 10000      # int — dataset size
model_name = "gpt-4"     # str — model identifier
is_trained = True        # bool — model status
embedding = [0.1, 0.5]   # list — a vector

# Type checking is important in AI code
print(f"Type of accuracy: {type(accuracy)}")
print(f"Type of model_name: {type(model_name)}")

# ─── Type conversion ───────────────────────────
# AI data often comes as strings and needs conversion
text_score = "0.87"
numeric_score = float(text_score)    # str → float
whole_number = int(3.99)             # float → int (becomes 3)
binary_label = bool(1)               # int → bool (becomes True)

print(f"Converted: {text_score} → {numeric_score} (type: {type(numeric_score)})")


# ════════════════════════════════════════════════════════
# SECTION 2: DATA STRUCTURES
# ════════════════════════════════════════════════════════

# ─── Lists: ordered collections (feature vectors, data rows) ───
features = [1.2, 3.4, 0.8, 2.1, 5.6]
print(f"\nFeatures: {features}")
print(f"First feature: {features[0]}")
print(f"Number of features: {len(features)}")

# Slicing — extracting parts of a list
print(f"First 3 features: {features[:3]}")
print(f"Last 2 features: {features[-2:]}")

# Adding and modifying
features.append(4.0)        # Add to end
features[0] = 1.5           # Modify by index
features.remove(4.0)       # Remove by value
print(f"Modified features: {features}")

# ─── Dictionaries: key-value pairs (configs, JSON, API responses) ───
model_config = {
    "name": "linear_regression",
    "learning_rate": 0.01,
    "max_iterations": 1000,
    "regularization": "l2",
    "verbose": True
}

print(f"\nModel config: {model_config}")
print(f"Learning rate: {model_config['learning_rate']}")
print(f"All keys: {list(model_config.keys())}")
print(f"All values: {list(model_config.values())}")

# Adding and updating
model_config["batch_size"] = 32      # Add new key
model_config["learning_rate"] = 0.001  # Update existing
print(f"Updated config: {model_config}")

# ─── Sets: unique elements (vocabularies, deduplication) ───
vocabulary = set()
for word in ["the", "cat", "sat", "the", "dog", "cat"]:
    vocabulary.add(word)

print(f"\nVocabulary (duplicates removed): {vocabulary}")
print(f"Vocabulary size: {len(vocabulary)}")

# Set operations — useful for comparing datasets
train_words = {"cat", "dog", "bird"}
test_words = {"dog", "bird", "fish"}
print(f"In train but not test: {train_words - test_words}")
print(f"In both: {train_words & test_words}")
print(f"In either: {train_words | test_words}")

# ─── Tuples: immutable sequences (coordinates, fixed records) ───
# Useful for things that shouldn't change, like data point coordinates
data_point = (3.5, "positive", 42)
x, label, id = data_point  # Unpacking
print(f"\nData point: x={x}, label={label}, id={id}")


# ════════════════════════════════════════════════════════
# SECTION 3: CONTROL FLOW
# ════════════════════════════════════════════════════════

# ─── if / elif / else ───────────────────────────
accuracy = 0.92

if accuracy >= 0.95:
    grade = "Excellent"
elif accuracy >= 0.85:
    grade = "Good"
elif accuracy >= 0.70:
    grade = "Fair"
else:
    grade = "Needs improvement"

print(f"\nModel accuracy {accuracy} → Grade: {grade}")

# ─── for loops ──────────────────────────────────
# Iterating over data is the most common AI operation
predictions = [0.9, 0.4, 0.7, 0.2, 0.85]
correct_labels = [1, 0, 1, 0, 1]

correct = 0
total = len(predictions)
for pred, label in zip(predictions, correct_labels):
    predicted_class = 1 if pred >= 0.5 else 0
    if predicted_class == label:
        correct += 1

accuracy = correct / total
print(f"Manual accuracy calculation: {accuracy:.2f}")

# ─── while loops ────────────────────────────────
# Used in training loops
loss = 1.0
iteration = 0
while loss > 0.1 and iteration < 1000:
    loss *= 0.95  # Simulate loss decreasing
    iteration += 1

print(f"\nTraining stopped at iteration {iteration}, loss = {loss:.4f}")


# ════════════════════════════════════════════════════════
# SECTION 4: FUNCTIONS
# ════════════════════════════════════════════════════════

# ─── Basic function ─────────────────────────────
def calculate_accuracy(predictions, labels, threshold=0.5):
    """Calculate classification accuracy.

    Args:
        predictions: list of predicted probabilities
        labels: list of true labels (0 or 1)
        threshold: decision threshold for classification

    Returns:
        float: accuracy between 0.0 and 1.0
    """
    correct = 0
    for pred, label in zip(predictions, labels):
        predicted_class = 1 if pred >= threshold else 0
        if predicted_class == label:
            correct += 1
    return correct / len(labels)

# Using the function
acc = calculate_accuracy(
    predictions=[0.9, 0.4, 0.7, 0.2, 0.85],
    labels=[1, 0, 1, 0, 1]
)
print(f"\nAccuracy: {acc:.2%}")

# ─── Function with multiple returns ─────────────
def evaluate_model(predictions, labels, threshold=0.5):
    """Returns multiple metrics."""
    tp = fp = tn = fn = 0

    for pred, label in zip(predictions, labels):
        predicted_class = 1 if pred >= threshold else 0
        if predicted_class == 1 and label == 1:
            tp += 1
        elif predicted_class == 1 and label == 0:
            fp += 1
        elif predicted_class == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    }

metrics = evaluate_model(
    predictions=[0.9, 0.4, 0.7, 0.2, 0.85, 0.6],
    labels=[1, 0, 1, 0, 1, 0]
)
print(f"\nModel Metrics:")
for metric, value in metrics.items():
    if metric == "confusion":
        print(f"  Confusion Matrix: {value}")
    else:
        print(f"  {metric}: {value:.4f}")


# ════════════════════════════════════════════════════════
# SECTION 5: CLASSES AND OBJECTS (OOP)
# ════════════════════════════════════════════════════════

class SimpleModel:
    """A simple model that learns the mean of training data.

    This demonstrates OOP concepts that you'll see in every
    AI library (scikit-learn, PyTorch, Hugging Face).
    """

    def __init__(self, name="simple_model"):
        """Constructor — called when model is created."""
        self.name = name
        self.is_trained = False
        self.prediction_value = None

    def fit(self, X):
        """Train the model — calculate the mean."""
        if len(X) == 0:
            raise ValueError("Cannot train on empty data")
        self.prediction_value = sum(X) / len(X)
        self.is_trained = True
        print(f"  [{self.name}] Trained on {len(X)} samples, "
              f"learned value = {self.prediction_value:.2f}")

    def predict(self, X):
        """Make predictions — always returns the learned mean."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return [self.prediction_value] * len(X)

    def __str__(self):
        """String representation."""
        status = "trained" if self.is_trained else "untrained"
        return f"SimpleModel(name='{self.name}', status={status})"


# Using the class
model = SimpleModel(name="mean_predictor")
print(f"\n{model}")

model.fit([3.5, 4.2, 3.8, 4.0, 3.9])
predictions = model.predict([1.0, 2.0, 3.0])
print(f"  Predictions: {predictions}")


# ════════════════════════════════════════════════════════
# SECTION 6: LIST COMPREHENSIONS
# ════════════════════════════════════════════════════════

# List comprehensions are Pythonic, fast, and heavily used in AI

# Traditional loop
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension (same result, cleaner)
squares = [x ** 2 for x in range(10)]

# With conditions (filtering)
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]

# AI example: convert probabilities to class labels
probabilities = [0.9, 0.3, 0.7, 0.2, 0.85]
classes = [1 if p >= 0.5 else 0 for p in probabilities]
print(f"\nProbabilities: {probabilities}")
print(f"Class labels:  {classes}")

# Dictionary comprehension (creating configs for multiple models)
models = {f"model_{i}": {"lr": 0.01 * i, "epochs": 100}
          for i in range(1, 4)}
print(f"\nModel configs: {models}")


# ════════════════════════════════════════════════════════
# SECTION 7: FILE I/O
# ════════════════════════════════════════════════════════

import json

# ─── Writing data to a JSON file ────────────────
training_log = {
    "model": "simple_classifier",
    "epochs": 100,
    "final_loss": 0.0234,
    "final_accuracy": 0.945,
    "history": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0234]
}

with open("training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)
print("\n✓ Saved training log to training_log.json")

# ─── Reading data from a JSON file ──────────────
with open("training_log.json", "r") as f:
    loaded_log = json.load(f)

print(f"  Loaded model: {loaded_log['model']}")
print(f"  Final accuracy: {loaded_log['final_accuracy']}")
print(f"  Training history: {loaded_log['history']}")

# ─── Writing to a text file (logs, predictions) ─
with open("predictions.txt", "w") as f:
    for i, pred in enumerate(probabilities):
        label = "positive" if pred >= 0.5 else "negative"
        f.write(f"Sample {i}: prob={pred:.2f}, label={label}\n")
print("✓ Saved predictions to predictions.txt")

# ─── Reading from a text file ───────────────────
with open("predictions.txt", "r") as f:
    lines = f.readlines()
print(f"  Read {len(lines)} predictions from file")


# ════════════════════════════════════════════════════════
# SECTION 8: ERROR HANDLING
# ════════════════════════════════════════════════════════

# AI code fails often — handle errors gracefully

def safe_predict(model, input_data):
    """Safely run prediction with error handling."""
    try:
        result = model.predict(input_data)
        return {"success": True, "result": result}
    except ValueError as e:
        return {"success": False, "error": f"Value error: {e}"}
    except TypeError as e:
        return {"success": False, "error": f"Type error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected: {e}"}

# Test with untrained model
untrained = SimpleModel("untrained")
result = safe_predict(untrained, [1.0, 2.0])
print(f"\nSafe predict (untrained): {result}")

# Test with trained model
trained = SimpleModel("trained")
trained.fit([1.0, 2.0, 3.0])
result = safe_predict(trained, [1.0, 2.0])
print(f"Safe predict (trained): {result}")


# ════════════════════════════════════════════════════════
# SECTION 9: MODULES AND IMPORTS
# ════════════════════════════════════════════════════════

# You'll import many libraries. Understanding how imports work is key.

# Standard library imports
import math
import os
import sys
import json
import random
import datetime

# Third-party imports (need pip install)
# import numpy as np          # Convention: alias as np
# import pandas as pd         # Convention: alias as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# Using standard library
print(f"\nπ (pi) = {math.pi:.6f}")
print(f"e = {math.e:.6f}")
print(f"Square root of 16 = {math.sqrt(16)}")
print(f"Current directory: {os.getcwd()}")


# ════════════════════════════════════════════════════════
# SECTION 10: f-STRINGS AND STRING MANIPULATION
# ════════════════════════════════════════════════════════

# f-strings are the Pythonic way to format strings
model_name = "GPT-4"
accuracy = 0.945
latency_ms = 234

print(f"\nModel: {model_name}")
print(f"Accuracy: {accuracy:.2%}")        # Percentage format
print(f"Latency: {latency_ms}ms ({latency_ms/1000:.2f}s)")

# String methods used in text processing for AI
text = "  The Quick Brown Fox Jumps Over The Lazy Dog  "

print(f"\nOriginal: '{text}'")
print(f"Stripped: '{text.strip()}'")
print(f"Lower:    '{text.lower().strip()}'")
print(f"Upper:    '{text.upper().strip()}'")
print(f"Split:    {text.split()}")
print(f"Replace:  '{text.replace('Fox', 'AI').strip()}'")

# Joining (common in text processing for RAG)
words = ["The", "AI", "model", "works"]
sentence = " ".join(words)
print(f"Joined:   '{sentence}'")

print("\n" + "=" * 50)
print("✓ Chapter 3 Python toolkit complete!")
print("=" * 50)
```

---

## Real-World Example

```
┌──────────────────────────────────────────────────────────────┐
│  PYTHON IN A REAL AI PIPELINE                                │
│                                                              │
│  # File: pipeline.py                                         │
│  import json                                                 │
│  from openai import OpenAI    # API call                     │
│  import chromadb              # Vector DB                    │
│  import pandas as pd          # Data handling                │
│                                                              │
│  def process_document(file_path):           # File I/O       │
│      with open(file_path, 'r') as f:                          │
│          text = f.read()                                      │
│      chunks = chunk_text(text)              # Function        │
│      embeddings = [embed(c) for c in chunks] # List comp     │
│      store_in_vectordb(embeddings)           # Function       │
│      return {"status": "success", "chunks": len(chunks)}     │
│                                                              │
│  Every line uses the Python skills from this chapter.         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

⚠️ **WARNING:**

1. *Mutating lists while iterating.* Never add/remove items from a list while looping over it. Make a copy first.
2. *Using `==` for floats.* Floating point numbers aren't exact. Use `abs(a - b) < 1e-9` instead of `a == b`.
3. *Not using list comprehensions.* They're faster and more readable. Practice them.
4. *Ignoring error handling.* AI code deals with external APIs, files, and models. Things *will* fail. Always use try/except.
5. *Confusing `=` and `==`.* `=` assigns, `==` compares. In `if` statements, always use `==`.

---

## Engineering Perspective

🧠 **THINK LIKE AN ENGINEER:**

Professional AI code follows these conventions:

```
1. Type hints     → def predict(self, X: np.ndarray) -> np.ndarray
2. Docstrings     → Explain what every function does
3. Error handling → try/except around all external calls
4. Logging        → print() is for notebooks, logging is for production
5. Config files   → Don't hardcode values; use config dicts/JSON
6. Modularity     → Split code into files: models.py, data.py, api.py
```

---

## Mini-Project — Student Grade Tracker

**Goal:** Build a complete Python program using everything from this chapter.

```python
"""
Project 1: Student Grade Tracker
Demonstrates: classes, dicts, file I/O, error handling, functions

project/
├── grade_tracker.py   ← This file
└── grades.json         ← Created by the program
"""

import json
import os

class Student:
    """Represents a student with grades."""
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.grades = {}  # subject → grade

    def add_grade(self, subject, grade):
        """Add or update a grade for a subject."""
        if not (0 <= grade <= 100):
            raise ValueError(f"Grade must be 0-100, got {grade}")
        self.grades[subject] = grade

    def average(self):
        """Calculate the average grade."""
        if not self.grades:
            return 0
        return sum(self.grades.values()) / len(self.grades)

    def to_dict(self):
        """Convert to dictionary for JSON storage."""
        return {
            "name": self.name,
            "id": self.student_id,
            "grades": self.grades,
            "average": round(self.average(), 2)
        }

    def __str__(self):
        return f"Student({self.name}, avg={self.average():.1f})"


class GradeTracker:
    """Manages multiple students and persists to file."""
    def __init__(self, data_file="grades.json"):
        self.data_file = data_file
        self.students = {}
        self.load()

    def add_student(self, name, student_id):
        """Add a new student."""
        if student_id in self.students:
            print(f"  Student {student_id} already exists")
            return
        self.students[student_id] = Student(name, student_id)
        print(f"  ✓ Added student: {name} (ID: {student_id})")

    def add_grade(self, student_id, subject, grade):
        """Add a grade to a student."""
        if student_id not in self.students:
            print(f"  ✗ Student {student_id} not found")
            return
        try:
            self.students[student_id].add_grade(subject, grade)
            print(f"  ✓ {subject}: {grade} for {self.students[student_id].name}")
        except ValueError as e:
            print(f"  ✗ Error: {e}")

    def report(self):
        """Print a full report."""
        print("\n" + "=" * 50)
        print("  GRADE TRACKER REPORT")
        print("=" * 50)
        for sid, student in self.students.items():
            print(f"\n  {student.name} (ID: {sid})")
            print(f"  Average: {student.average():.1f}")
            for subject, grade in student.grades.items():
                print(f"    {subject}: {grade}")
        print("=" * 50)

    def save(self):
        """Save all students to JSON file."""
        data = {sid: s.to_dict() for sid, s in self.students.items()}
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Saved to {self.data_file}")

    def load(self):
        """Load students from JSON file if it exists."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            for sid, info in data.items():
                student = Student(info["name"], sid)
                student.grades = info["grades"]
                self.students[sid] = student
            print(f"  ✓ Loaded {len(self.students)} students from {self.data_file}")


# ─── Run the tracker ──────────────────────────
if __name__ == "__main__":
    tracker = GradeTracker()

    # Add students
    tracker.add_student("Alice", "S001")
    tracker.add_student("Bob", "S002")

    # Add grades
    tracker.add_grade("S001", "Math", 95)
    tracker.add_grade("S001", "Science", 88)
    tracker.add_grade("S001", "English", 92)
    tracker.add_grade("S002", "Math", 78)
    tracker.add_grade("S002", "Science", 85)
    tracker.add_grade("S002", "English", 90)

    # Print report
    tracker.report()

    # Save to file
    tracker.save()

    # Run again to test loading
    print("\n  --- Reloading from file ---")
    tracker2 = GradeTracker()
    tracker2.report()
```

---

## Exercises

### Easy
1. Create a list of 10 numbers. Use a list comprehension to create a new list with only the even numbers.
2. Write a function `is_spam(text)` that returns True if the text contains "free" or "win" (case insensitive).
3. Create a dictionary for a model configuration with 5 keys. Print each key-value pair.

### Medium
4. Write a `Dataset` class that stores samples and labels, with methods to add samples, get statistics, and split into train/test sets.
5. Write a function that reads a JSON file containing a list of predictions and calculates accuracy.
6. Create a text-processing function that: reads a file, splits into sentences, removes punctuation, and counts words.

### Advanced
7. Build a simple `Pipeline` class that chains multiple data transformations (e.g., lowercase → tokenize → remove stopwords) and applies them in sequence.
8. Implement a simple logging system that writes to both console and a file, with timestamps.

---

## Chapter Summary

```
┌──────────────────────────────────────────────────────────┐
│                   CHAPTER 3 KEY TAKEAWAYS                │
│                                                          │
│  • Python is the foundation of AI Engineering            │
│  • Key data structures: lists, dicts, sets, tuples      │
│  • List comprehensions are fast and Pythonic             │
│  • Classes model real-world AI components                 │
│  • File I/O: JSON for configs, text for logs             │
│  • Error handling is essential for production AI code     │
│  • f-strings for clean, readable output                   │
│  • Type hints and docstrings make code professional      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Interview Questions

1. "What's the difference between a list and a tuple? When would you use each?"
2. "Write a function that takes a list of numbers and returns the mean and standard deviation."
3. "How would you handle a JSON file that might be corrupted?"
4. "What are list comprehensions and why are they preferred over loops?"

---

## Knowledge Check

1. What's the output of `[x**2 for x in range(5) if x > 1]`?
2. How do you add a new key to an existing dictionary?
3. What does `zip()` do and why is it useful in AI code?
4. Write a try/except block that catches division by zero.
5. What's the difference between `==` and `is` in Python?
