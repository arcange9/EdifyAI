# Appendix C — Final Assessment

## Overview

This assessment covers all material from the 90-day curriculum. It is divided into four sections matching the three months plus a capstone evaluation.

Passing score: 75% overall. Each section is weighted equally.

---

## Section 1: Foundations (Month 1) — 25 Questions

### Multiple Choice (1-15)

```
1. What is the key difference between AI, ML, and DL?
   a) AI is robots, ML is statistics, DL is neural networks
   b) AI is the broad field, ML is learning from data, DL is multi-layer neural networks
   c) They are the same thing
   d) DL is broader than AI

   Answer: b

2. What does NumPy's broadcasting do?
   a) Sends arrays over the network
   b) Allows operations on arrays of different shapes
   c) Converts arrays to lists
   d) Increases array dimensions automatically

   Answer: b

3. Which Pandas method groups data and computes statistics?
   a) df.aggregate()
   b) df.group()
   c) df.groupby()
   d) df.combine()

   Answer: c

4. What is the dot product of [1, 2, 3] and [4, 5, 6]?
   a) [4, 10, 18]
   b) 32
   c) [5, 7, 9]
   d) 6

   Answer: b  (1×4 + 2×5 + 3×6 = 4+10+18 = 32)

5. In a standard normal distribution, what percentage falls within 2 standard deviations?
   a) 68%
   b) 95%
   c) 99.7%
   d) 50%

   Answer: b

6. What does gradient descent do?
   a) Classifies data points
   b) Minimizes loss by moving opposite to the gradient
   c) Finds the maximum of a function
   d) Normalizes data

   Answer: b

7. Which metric is best for imbalanced classification?
   a) Accuracy
   b) F1 Score
   c) Mean Squared Error
   d) R²

   Answer: b

8. What is overfitting?
   a) Model performs poorly on training data
   b) Model memorizes training data but fails on new data
   c) Model has too few parameters
   d) Model uses too little data

   Answer: b

9. What does StandardScaler do?
   a) Scales features to [0, 1]
   b) Centers to mean 0 and scales to std 1
   c) Removes outliers
   d) Converts to categorical

   Answer: b

10. Which algorithm builds an ensemble of decision trees?
    a) Linear Regression
    b) K-Means
    c) Random Forest
    d) PCA

    Answer: c

11. What is the purpose of train_test_split?
    a) To increase dataset size
    b) To evaluate model on unseen data
    c) To normalize features
    d) To remove duplicates

    Answer: b

12. A confusion matrix has TP=80, TN=50, FP=10, FN=20. What is accuracy?
    a) 0.80
    b) 0.75
    c) 0.85
    d) 0.65

    Answer: a  ((80+50)/(80+50+10+20) = 130/160 = 0.8125 ≈ 0.80)

13. What is a residual in regression?
    a) The predicted value
    b) The actual value
    c) The difference between actual and predicted: y - ŷ
    d) The slope of the line

    Answer: c

14. Which is a categorical encoding method?
    a) StandardScaler
    b) One-Hot Encoding
    c) MinMaxScaler
    d) PCA

    Answer: b

15. What does k-fold cross-validation do?
    a) Splits data into k parts, trains on k-1, tests on 1, repeats k times
    b) Trains k models simultaneously
    c) Splits data into k categories
    d) Reduces features by k

    Answer: a
```

### Short Answer (16-25)

```
16. Explain what eigenvalues and eigenvectors represent.
    → Eigenvalues measure the "importance" or "scale" of a direction.
    Eigenvectors are the directions. Used in PCA to find the most
    important directions in data.

17. What is the difference between supervised and unsupervised learning?
    → Supervised: labeled data (has answers). Unsupervised: no labels,
    find patterns/structure (clustering, PCA).

18. Why do we normalize features before training?
    → Features on different scales cause some to dominate. Normalization
    puts all features on a similar scale so the model treats them equally.

19. What is the bias-variance tradeoff?
    → Bias: underfitting (too simple). Variance: overfitting (too complex).
    The tradeoff is finding the sweet spot that generalizes well.

20. Write Python code to compute the mean and standard deviation of a list.
    → import numpy as np
      data = [1, 2, 3, 4, 5]
      print(np.mean(data))   # 3.0
      print(np.std(data))    # 1.414...

21. What is a confusion matrix and why is it useful?
    → A table showing TP, TN, FP, FN. Useful because it shows exactly
    what the model gets right and wrong, not just a single number.

22. What is the purpose of the validation set?
    → To tune hyperparameters and select models without touching the
    test set, ensuring an honest final evaluation.

23. Explain what a decision tree's "split" does.
    → It divides data based on a feature threshold (e.g., age < 30)
    to maximize purity (minimize Gini impurity) in each child node.

24. What is feature engineering and why does it matter?
    → Creating new input features from existing data (e.g., extracting
    "day of week" from a date). Better features often improve model
    performance more than algorithm choice.

25. What is one-hot encoding and when would you use it?
    → Converting categorical values to binary vectors (e.g., "red" → [1,0,0]).
    Used when categorical data has no natural ordering.
```

---

## Section 2: Deep Learning (Month 2) — 25 Questions

### Multiple Choice (26-40)

```
26. What does ReLU activation do?
    a) Squashes to [0, 1]
    b) Returns max(0, x)
    c) Returns 1/(1+e^-x)
    d) Returns tanh(x)

    Answer: b

27. What is backpropagation?
    a) Forward pass through the network
    b) Computing gradients using the chain rule
    c) Initializing weights
    d) Applying dropout

    Answer: b

28. Which layer type is used for image processing?
    a) LSTM
    b) Dense
    c) Convolutional
    d) Embedding

    Answer: c

29. What is the attention mechanism?
    a) A regularization technique
    b) A way for each token to focus on relevant parts of input
    c) A loss function
    d) An optimizer

    Answer: b

30. What does LoRA do?
    a) Trains a model from scratch
    b) Fine-tunes only a small set of adapter parameters
    c) Quantizes a model to INT4
    d) Prunes model weights

    Answer: b

31. Which is NOT a tokenization method?
    a) BPE
    b) WordPiece
    c) SentencePiece
    d) Backpropagation

    Answer: d

32. What is a transformer's key advantage over RNNs?
    a) Smaller models
    b) Parallel processing of all tokens
    c) Better for images
    d) No need for training data

    Answer: b

33. What does quantization do?
    a) Adds more parameters
    b) Reduces precision (e.g., FP32 → INT4) to save memory
    c) Increases model accuracy
    d) Speeds up tokenization

    Answer: b

34. What is a KV cache in transformers?
    a) Stores training data
    b) Stores computed keys and values for previous tokens to avoid recomputation
    c) Caches API responses
    d) Stores model weights

    Answer: b

35. In PyTorch, what does loss.backward() do?
    a) Reverses the forward pass
    b) Computes gradients of the loss w.r.t. all parameters
    c) Updates model weights
    d) Saves the model

    Answer: b

36. What problem do LSTMs solve that vanilla RNNs cannot?
    a) Image classification
    b) Vanishing gradient problem in long sequences
    c) Overfitting
    d) Slow training

    Answer: b

37. What is the purpose of the softmax function?
    a) Activate neurons
    b) Convert logits to a probability distribution (sums to 1)
    c) Normalize input data
    d) Apply dropout

    Answer: b

38. What is dropout?
    a) Removing the last layer
    b) Randomly disabling neurons during training to prevent overfitting
    c) Dropping training examples
    d) Reducing learning rate

    Answer: b

39. What does the Hugging Face transformers library primarily provide?
    a) Only training scripts
    b) Pre-trained models, tokenizers, and pipelines for NLP tasks
    c) A database
    d) A web framework

    Answer: b

40. What is the embedding dimension in an LLM?
    a) The number of layers
    b) The size of the vector representing each token
    c) The vocabulary size
    d) The context window length

    Answer: b
```

### Short Answer (41-50)

```
41. Explain the chain rule and why it matters for neural networks.
    → The chain rule computes derivatives of composed functions.
    Neural networks are layers of composed functions, so backpropagation
    uses the chain rule to compute gradients for every weight.

42. What is the vanishing gradient problem?
    → Gradients shrink as they propagate backward through deep layers,
    causing early layers to barely learn. Solved by ReLU, residual
    connections, and LSTMs.

43. Why are transformers better than RNNs for long sequences?
    → Transformers use self-attention to process all tokens in parallel
    and can attend to any token regardless of distance, while RNNs must
    process sequentially and forget distant information.

44. What is the difference between pre-training and fine-tuning?
    → Pre-training: training on massive data to learn general patterns.
    Fine-tuning: adapting the pre-trained model to a specific task with
    smaller, labeled data.

45. Write a PyTorch training loop (pseudocode is fine).
    → for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch.x)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

46. What is multi-head attention and why use multiple heads?
    → Multiple attention mechanisms running in parallel, each focusing
    on different relationships (syntax, semantics, etc.). Captures richer
    patterns than single-head attention.

47. What does model.eval() do in PyTorch?
    → Sets the model to evaluation mode, disabling dropout and batch
    norm updates. Used before inference/testing.

48. Explain what an embedding is in the context of LLMs.
    → A dense vector (e.g., 768 dimensions) representing a token's meaning.
    Similar tokens have similar vectors. Learned during pre-training.

49. What is the difference between greedy decoding and temperature sampling?
    → Greedy: always pick the highest probability token (deterministic).
    Temperature sampling: samples from a softened distribution (more
    creative, less predictable).

50. Why would you use LoRA instead of full fine-tuning?
    → LoRA trains only small adapter matrices (0.1-1% of parameters),
    making it much faster, cheaper, and requiring less GPU memory
    while achieving near-full-finetuning quality.
```

---

## Section 3: Production AI (Month 3) — 25 Questions

### Multiple Choice (51-65)

```
51. What is RAG?
    a) A type of neural network
    b) Retrieval-Augmented Generation; grounding LLM answers in retrieved documents
    c) A regularization technique
    d) A loss function

    Answer: b

52. What is the purpose of a vector database?
    a) Store SQL tables
    b) Store and search embeddings by similarity
    c) Cache API responses
    d) Store model weights

    Answer: b

53. Which is a common vector database?
    a) PostgreSQL
    b) ChromaDB
    c) Redis
    d) MongoDB

    Answer: b

54. What is chunking in RAG?
    a) Splitting the model
    b) Splitting documents into smaller pieces for retrieval
    c) Dividing the training data
    d) Compressing vectors

    Answer: b

55. What does an AI agent do?
    a) Only answers questions
    b) Uses tools and multi-step reasoning to accomplish tasks
    c) Trains models
    d) Stores embeddings

    Answer: b

56. What is a hallucination in LLMs?
    a) A memory error
    b) The LLM generating plausible but false information
    c) A GPU failure
    d) A training artifact

    Answer: b

57. What is structured output in LLM APIs?
    a) Formatting the response as HTML
    b) Forcing the LLM to return JSON matching a schema
    c) Sorting the output alphabetically
    d) Caching the response

    Answer: b

58. What is the purpose of FastAPI?
    a) Train ML models
    b) Build fast web APIs in Python
    c) Store vectors
    d) Generate images

    Answer: b

59. What does Docker do?
    a) Trains models faster
    b) Packages applications with dependencies in containers
    c) Hosts vector databases
    d) Optimizes prompts

    Answer: b

60. What is cosine similarity used for?
    a) Measuring model accuracy
    b) Measuring similarity between vectors
    c) Optimizing loss
    d) Normalizing data

    Answer: b

61. What is the purpose of a guardrail in AI systems?
    a) Speed up inference
    b) Prevent unsafe or out-of-scope model outputs
    c) Compress the model
    d) Cache responses

    Answer: b

62. Which is NOT a strategy to reduce LLM costs?
    a) Using smaller models for simple tasks
    b) Caching responses
    c) Adding more GPU layers
    d) Batching requests

    Answer: c

63. What is the role of an orchestrator in multi-agent systems?
    a) Trains all agents
    b) Coordinates which agent handles which task
    c) Stores agent memory
    d) Generates embeddings

    Answer: b

64. What does the /docs endpoint provide in FastAPI?
    a) Source code
    b) Interactive API documentation (Swagger UI)
    c) Model training logs
    d) Database schema

    Answer: b

65. What is the main purpose of monitoring in production AI?
    a) To increase model size
    b) To track performance, detect issues, and ensure quality over time
    c) To train models
    d) To generate embeddings

    Answer: b
```

### Short Answer (66-75)

```
66. Describe the RAG pipeline step by step.
    → 1. Chunk documents into pieces.
    2. Embed each chunk.
    3. Store embeddings in a vector database.
    4. When a question comes in, embed the question.
    5. Retrieve the most similar chunks (cosine similarity).
    6. Pass chunks as context to the LLM.
    7. LLM generates an answer grounded in the context.

67. What is the difference between semantic search and keyword search?
    → Keyword search matches exact words. Semantic search matches meaning
    using embeddings, so it finds relevant results even without exact
    word matches.

68. Why is prompt engineering important?
    → The quality of LLM output depends heavily on input quality. Good
    prompts include context, clear instructions, examples, and
    constraints to get reliable, useful responses.

69. What is the difference between an AI agent and a chatbot?
    → A chatbot answers questions. An agent uses tools, plans multi-step
    actions, reasons about results, and accomplishes tasks autonomously.

70. Name three strategies to reduce hallucinations.
    → 1. RAG (ground answers in real documents)
    2. Ask the model to say "I don't know" when uncertain
    3. Use structured outputs and validation
    4. Lower temperature for factual tasks
    5. Citation requirements

71. What is the purpose of Docker in AI deployment?
    → Packages the model, dependencies, and runtime into a portable
    container that runs identically on any machine, eliminating
    "works on my machine" issues.

72. Explain what an embedding model does.
    → Converts text (or images) into dense numeric vectors where
    semantically similar items have similar vectors. Enables
    similarity search, clustering, and RAG.

73. What is the role of an API key in LLM applications?
    → Authenticates requests to the LLM provider. Must be stored
    securely (environment variables, secret managers) and never
    committed to version control.

74. What are three metrics you should monitor in a production AI system?
    → 1. Latency (response time)
    2. Token usage / cost per request
    3. Error rate
    4. User satisfaction / feedback
    5. Hallucination rate
    6. Cache hit rate

75. Describe one advanced RAG technique and its benefit.
    → Re-ranking: Retrieve more chunks (e.g., 20), then use a
    cross-encoder to re-rank them by relevance and keep the top 5.
    Benefit: better precision in what's sent to the LLM.
    OR: Query expansion: generate multiple variations of the query
    and retrieve for each, then combine results. Benefit: catches
    more relevant documents.
```

---

## Section 4: Capstone Evaluation — Practical

```
┌──────────────────────────────────────────────────────────────┐
│  CAPSTONE: Build a Production AI Application               │
│                                                              │
│  TASK: Build an AI-powered application that:               │
│                                                              │
│  1. Accepts document uploads (PDF or text)                │
│  2. Uses RAG to answer questions about documents           │
│  3. Generates study materials (notes or flashcards)       │
│  4. Exposes a REST API using FastAPI                       │
│  5. Is containerized with Docker                            │
│  6. Includes a README with setup instructions             │
│                                                              │
│  EVALUATION RUBRIC (100 points):                           │
│                                                              │
│  RAG Pipeline (20 pts)                                    │
│    20: Accurate retrieval, citations, no hallucination    │
│    15: Good retrieval, minor issues                        │
│    10: Basic retrieval, some hallucination                 │
│     5: Poor retrieval, frequent hallucination              │
│     0: No RAG pipeline                                     │
│                                                              │
│  API Design (15 pts)                                      │
│    15: RESTful, documented, proper error handling         │
│    10: Functional but rough                                │
│     5: Some endpoints work                                 │
│     0: No API                                              │
│                                                              │
│  Code Quality (15 pts)                                   │
│    15: Clean, typed, documented, modular, tested          │
│    10: Readable but lacks tests or types                  │
│     5: Messy but functional                                │
│     0: Unreadable or broken                                │
│                                                              │
│  Deployment (15 pts)                                     │
│    15: Dockerized, deployed, health checks work          │
│    10: Dockerized but not deployed                       │
│     5: Dockerfile exists but doesn't build               │
│     0: No deployment setup                                │
│                                                              │
│  AI Features (15 pts)                                    │
│    15: RAG + generation + agent all functional            │
│    10: RAG + one other feature                           │
│     5: Only basic RAG                                     │
│     0: No AI features                                     │
│                                                              │
│  Documentation (10 pts)                                  │
│    10: README, API docs, architecture diagram            │
│     7: README only                                        │
│     3: Minimal docs                                       │
│     0: No docs                                            │
│                                                              │
│  Testing (10 pts)                                       │
│    10: Unit tests, integration tests, RAG eval           │
│     7: Some tests                                        │
│     3: Minimal tests                                      │
│     0: No tests                                           │
│                                                              │
│  PASSING: 75/100                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Scoring Summary

```
┌──────────────────────────────────────────────────────────────┐
│  FINAL ASSESSMENT SCORING                                │
│                                                              │
│  Section                    Questions    Weight    Passing   │
│  ──────────────────────────────────────────────────────────  │
│  1. Foundations             25          25%       18/25     │
│  2. Deep Learning           25          25%       18/25     │
│  3. Production AI           25          25%       18/25     │
│  4. Capstone (practical)    1 project   25%       75/100   │
│                                                              │
│  TOTAL                                   100%      75%      │
│                                                              │
│  GRADE SCALE:                                             │
│    90-100% = A (Expert)                                   │
│    80-89%  = B (Proficient)                               │
│    75-79%  = C (Pass)                                     │
│    <75%    = Review weak areas and retake                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Self-Assessment Checklist

After completing the 90-day journey, you should be able to:

```
FOUNDATIONS
  ☐ Explain AI vs ML vs DL vs GenAI
  ☐ Write Python code for data processing
  ☐ Use NumPy for array operations
  ☐ Use Pandas for data analysis
  ☐ Create visualizations with Matplotlib
  ☐ Understand vectors, matrices, dot products
  ☐ Understand probability distributions
  ☐ Understand gradients and optimization
  ☐ Preprocess data (scaling, encoding, missing values)
  ☐ Train and evaluate ML models (regression, classification)
  ☐ Use cross-validation and grid search

DEEP LEARNING
  ☐ Explain how a neural network works
  ☐ Implement backpropagation conceptually
  ☐ Build a neural network from scratch
  ☐ Use PyTorch to train models
  ☐ Understand CNNs for images
  ☐ Understand RNNs/LSTMs for sequences
  ☐ Explain self-attention and transformers
  ☐ Understand tokenization and embeddings
  ☐ Explain how GPT models work
  ☐ Write effective prompts (few-shot, chain-of-thought)
  ☐ Use Hugging Face for inference
  ☐ Fine-tune a model with LoRA
  ☐ Explain quantization

PRODUCTION AI
  ☐ Call LLM APIs (chat, embeddings, structured output)
  ☐ Implement tool calling
  ☐ Use a vector database for semantic search
  ☐ Build a RAG pipeline
  ☐ Evaluate RAG quality
  ☐ Build an AI agent with tools
  ☐ Design a multi-agent system
  ☐ Apply safety guardrails
  ☐ Build a FastAPI backend
  ☐ Containerize with Docker
  ☐ Monitor an AI system in production
  ☐ Optimize LLM costs
  ☐ Design production AI architecture

If you can check every box, you are an AI Engineer.
Welcome to the future. Now go build something amazing.
```

---

*End of Assessment. End of AI Engineering Mastery: A 90-Day Practical Journey.*
