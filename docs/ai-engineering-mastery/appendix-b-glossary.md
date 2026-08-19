# Appendix B — Complete Glossary

```
┌──────────────────────────────────────────────────────────────┐
│  AI ENGINEERING GLOSSARY                                     │
│                                                              │
│  A                                                           │
│  Accuracy — Fraction of correct predictions (TP+TN)/total  │
│  Activation Function — Non-linear function applied to       │
│    neuron output (ReLU, sigmoid, tanh, softmax)            │
│  Agent — AI system that uses tools and reasoning to        │
│    accomplish multi-step tasks                               │
│  API — Application Programming Interface; how software     │
│    communicates with other software                          │
│  Attention — Mechanism that focuses on relevant parts       │
│    of input (core of Transformers)                           │
│  Autograd — PyTorch's automatic gradient computation        │
│  Autoregressive — Generating output one token at a time    │
│    based on previous tokens                                  │
│                                                              │
│  B                                                           │
│  Backpropagation — Algorithm for computing gradients        │
│    in neural networks using the chain rule                  │
│  Bagging — Bootstrap aggregating; training multiple        │
│    models on random subsets                                  │
│  Batch — A group of samples processed together              │
│  Batch Size — Number of samples per training step           │
│  BPE — Byte Pair Encoding; subword tokenization             │
│    used by GPT models                                        │
│  Broadcasting — NumPy's feature for operating on            │
│    arrays of different shapes                                │
│  Bias — 1) Model's tendency to consistently predict        │
│    wrong (underfitting)  2) The 'b' term in y = Wx + b     │
│                                                              │
│  C                                                           │
│  CNN — Convolutional Neural Network; uses filters to       │
│    process images                                            │
│  Cross-Entropy — Loss function for classification           │
│  Cross-Validation — Splitting data K times to get          │
│    honest performance estimate                               │
│  Chain Rule — Calculus rule: d(f(g(x)))/dx = f'(g(x))*g'(x)│
│    Used in backpropagation                                   │
│  Chunking — Splitting documents into smaller pieces        │
│    for RAG retrieval                                        │
│  Citation — Reference to the source of information          │
│    in a RAG answer                                           │
│                                                              │
│  D                                                           │
│  DataFrame — Pandas 2D data structure (like a table)       │
│  Data Leakage — When test data information leaks           │
│    into training (causes overoptimistic results)            │
│  Deep Learning — ML using multi-layer neural networks     │
│  Dense Layer — Fully connected layer; every neuron         │
│    connects to every neuron in previous layer               │
│  Dropout — Randomly disabling neurons during training      │
│    to prevent overfitting                                    │
│  Dot Product — Element-wise multiply and sum; measures     │
│    vector similarity                                         │
│  Docker — Containerization tool for packaging apps         │
│    with dependencies                                         │
│                                                              │
│  E                                                           │
│  Embedding — Dense vector representation of text/image     │
│  Epoch — One full pass through the training data            │
│  Evaluation — Measuring model performance on test data     │
│  Eigenvalue — "Importance" of a direction in data           │
│    (used in PCA)                                             │
│                                                              │
│  F                                                           │
│  FAISS — Facebook AI Similarity Search; fast vector        │
│    search library                                            │
│  Feature — An input variable for a model                    │
│  Feature Engineering — Creating new features from          │
│    existing data to improve model performance               │
│  Few-Shot — Prompting with a few examples to guide          │
│    the LLM                                                   │
│  F1 Score — Harmonic mean of precision and recall           │
│  Fine-Tuning — Training a pre-trained model on your         │
│    specific data                                             │
│  FP (False Positive) — Model predicted positive,           │
│    but actual is negative                                    │
│  FN (False Negative) — Model predicted negative,            │
│    but actual is positive                                    │
│                                                              │
│  G                                                           │
│  Generative AI — AI that creates new content               │
│    (text, images, audio)                                     │
│  Gini Impurity — Measure used in decision trees to         │
│    decide splits                                            │
│  GPU — Graphics Processing Unit; thousands of cores       │
│    for parallel computation                                 │
│  Gradient — Vector of partial derivatives; points          │
│    in direction of steepest increase                        │
│  Gradient Descent — Optimization algorithm: move           │
│    opposite to gradient to minimize loss                    │
│                                                              │
│  H                                                           │
│  Hallucination — LLM generating plausible but false        │
│    information                                               │
│  Hidden Layer — Neural network layer between input          │
│    and output                                                │
│  Hugging Face — Open-source AI platform with 500k+        │
│    models, datasets, and tools                              │
│  Hyperparameter — Configuration set by engineer            │
│    (learning rate, batch size, etc.)                        │
│                                                              │
│  I                                                           │
│  Inference — Running a model to get predictions             │
│    (after training)                                          │
│  INT4/INT8 — Quantization methods reducing model           │
│    memory usage                                              │
│                                                              │
│  K                                                           │
│  K-Fold — Cross-validation with K splits                    │
│  K-NN — k-Nearest Neighbors; classifies by majority         │
│    vote of k closest points                                  │
│  KV Cache — Key-value cache in transformers that            │
│    stores previous token computations                       │
│                                                              │
│  L                                                           │
│  Label — The target/answer for a training example           │
│  Latency — Time from request to response                    │
│  Layer — A set of neurons in a neural network               │
│  Learning Rate — Step size for gradient descent             │
│  Linear Regression — Predicting a number using a            │
│    linear combination of features                           │
│  LSTM — Long Short-Term Memory; RNN with gates             │
│    to control memory                                         │
│  LLM — Large Language Model; transformer-based model       │
│    trained on massive text data                             │
│  LoRA — Low-Rank Adaptation; fine-tuning only a            │
│    small set of parameters                                   │
│  Loss Function — Measures how wrong predictions are       │
│                                                              │
│  M                                                           │
│  MSE — Mean Squared Error; common loss for regression     │
│  ML — Machine Learning; learning patterns from data        │
│  Model — A trained algorithm that makes predictions        │
│  Multi-Head Attention — Multiple attention mechanisms      │
│    running in parallel                                       │
│                                                              │
│  N                                                           │
│  Neuron — Basic unit of a neural network; computes         │
│    weighted sum + activation                                 │
│  Normalization — Scaling data to a standard range          │
│  NumPy — Python library for numerical computing            │
│                                                              │
│  O                                                           │
│  One-Hot Encoding — Converting categories to binary        │
│    vectors                                                   │
│  Optimizer — Algorithm that updates model weights          │
│    (SGD, Adam)                                               │
│  Ollama — Tool for running LLMs locally for free           │
│  Overfitting — Model learns training data too well,        │
│    fails on new data                                         │
│                                                              │
│  P                                                           │
│  Pandas — Python library for data manipulation             │
│  Parameters — Learnable weights in a model                  │
│  PCA — Principal Component Analysis; reduces               │
│    dimensionality while keeping variance                    │
│  PEFT — Parameter-Efficient Fine-Tuning                    │
│  Perceptron — Simplest neural network (single neuron)      │
│  Precision — TP / (TP + FP); of predicted positive,       │
│    how many are correct                                      │
│  Pre-training — Initial training on large data             │
│  Prompt Engineering — Designing inputs to get best        │
│    LLM outputs                                               │
│                                                              │
│  Q                                                           │
│  Q/K/V — Query, Key, Value in attention mechanism          │
│  Quantization — Reducing precision to save memory         │
│    (FP32 → INT4)                                             │
│                                                              │
│  R                                                           │
│  RAG — Retrieval-Augmented Generation; grounding           │
│    LLM answers in retrieved documents                       │
│  Recall — TP / (TP + FN); of actual positive,             │
│    how many were found                                       │
│  ReLU — Rectified Linear Unit; f(x) = max(0, x)           │
│  RNN — Recurrent Neural Network; processes sequences      │
│  ROC-AUC — Area under ROC curve; measures ranking         │
│    quality                                                   │
│                                                              │
│  S                                                           │
│  Scikit-learn — Python ML library                           │
│  Self-Attention — Each token attends to all other         │
│    tokens in the input                                       │
│  Sigmoid — f(x) = 1/(1+e^-x); outputs [0,1]               │
│  Softmax — Converts logits to probabilities (sum to 1)   │
│  Supervised Learning — Learning from labeled examples     │
│                                                              │
│  T                                                           │
│  Tensor — Multi-dimensional array (PyTorch's core)        │
│  Token — Unit of text (word or subword) for LLMs           │
│  Tokenization — Converting text to tokens                  │
│  TP (True Positive) — Correctly predicted positive        │
│  TN (True Negative) — Correctly predicted negative        │
│  Transformer — Neural network using attention;             │
│    foundation of modern LLMs                                 │
│                                                              │
│  U                                                           │
│  Underfitting — Model too simple; poor on both             │
│    train and test                                            │
│  Unsupervised Learning — Finding patterns without         │
│    labels                                                    │
│                                                              │
│  V                                                           │
│  Validation — Data used to tune model without               │
│    touching test data                                        │
│  Vanishing Gradient — Gradients become too small          │
│    in deep networks (solved by LSTMs, ReLU)                │
│  Vector DB — Database optimized for storing and           │
│    searching embeddings                                      │
│                                                              │
│  W                                                           │
│  Weights — Learnable parameters in a neural network        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
