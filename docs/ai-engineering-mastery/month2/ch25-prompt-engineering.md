# Chapter 25 — Prompt Engineering

## Learning Objectives

- Master prompt patterns that get better LLM outputs
- Know system prompts, few-shot, chain-of-thought, and structured prompts
- Understand why prompt engineering matters for production AI
- Build reusable prompt templates

---

## Why This Matters

How you ask an LLM determines what you get. The same model can produce a brilliant answer or a terrible one depending entirely on your prompt. In production AI systems, prompts are part of the codebase — they're tested, versioned, and optimized.

---

## Prompt Patterns

```
┌──────────────────────────────────────────────────────────────┐
│  PROMPT ENGINEERING PATTERNS                                │
│                                                              │
│  1. ZERO-SHOT: Just ask directly                              │
│     "Classify this email as spam or not: 'WIN FREE...'"     │
│                                                              │
│  2. FEW-SHOT: Show examples                                   │
│     "Classify emails:                                        │
│      'WIN FREE' → spam                                       │
│      'Meeting tomorrow' → not spam                            │
│      'Buy now' → spam                                         │
│      Now classify: 'Project update'"                         │
│                                                              │
│  3. CHAIN-OF-THOUGHT: Ask for reasoning                       │
│     "Think step by step: If a train travels..."             │
│     → Model shows reasoning → better accuracy                │
│                                                              │
│  4. ROLE-PLAYING: Give the model a role                      │
│     "You are an expert Python engineer. Review this code:"  │
│                                                              │
│  5. STRUCTURED OUTPUT: Ask for JSON/format                   │
│     "Return as JSON: {"sentiment": "...", "score": 0.8}"    │
│                                                              │
│  6. SYSTEM PROMPT: Set behavior rules                        │
│     "You are a helpful assistant. Only answer based on      │
│      the provided context. If the answer isn't in the       │
│      context, say 'I don't know.'"                          │
│                                                              │
│  7. INSTRUCTION + CONTEXT + QUESTION:                        │
│     [System: You are a tutor]                                │
│     [Context: Here is the lesson material...]                │
│     [Question: Explain photosynthesis]                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Prompt Templates

```python
"""
Chapter 25 — Prompt Engineering
Reusable prompt templates for AI applications.
"""

# ─── 1. System Prompts ─────────────────────────
SYSTEM_PROMPTS = {
    "tutor": "You are an expert AI tutor. Explain concepts simply, "
             "use examples, and check understanding. Adapt your "
             "explanation level to the student's knowledge.",

    "code_reviewer": "You are a senior software engineer reviewing code. "
                     "Check for bugs, security issues, performance, and "
                     "best practices. Be specific and constructive.",

    "summarizer": "You are a professional summarizer. Create concise, "
                  "accurate summaries that capture key points. "
                  "Never add information not in the source.",

    "classifier": "You are a text classifier. Only respond with the "
                  "category name. Categories: {categories}. "
                  "If unsure, respond 'unknown'.",
}

# ─── 2. Few-Shot Template ──────────────────────
def few_shot_prompt(task_description, examples, query):
    """Build a few-shot prompt from examples."""
    prompt = f"Task: {task_description}\n\n"
    prompt += "Examples:\n"
    for inp, out in examples:
        prompt += f"Input: {inp}\nOutput: {out}\n\n"
    prompt += f"Input: {query}\nOutput:"
    return prompt

# Example: sentiment classification
sentiment_prompt = few_shot_prompt(
    "Classify the sentiment as positive, negative, or neutral.",
    [("This product is amazing!", "positive"),
     ("Worst purchase ever.", "negative"),
     ("It's okay, nothing special.", "neutral")],
    "The delivery was on time and quality is decent."
)
print("=== Few-Shot Prompt ===")
print(sentiment_prompt)

# ─── 3. Chain-of-Thought Template ──────────────
def cot_prompt(question):
    """Chain-of-thought: ask model to reason step by step."""
    return f"""Question: {question}

Let's think step by step to find the answer.

1. First, identify what we know:
2. Then, determine what we need to find:
3. Next, apply the relevant formula or logic:
4. Finally, calculate the answer:

Answer:"""

print("\n=== Chain-of-Thought Prompt ===")
print(cot_prompt("If a store sells 15 items at $12 each and gives "
                 "a 10% discount, what's the total revenue?"))

# ─── 4. Structured Output Template ─────────────
def structured_prompt(text, schema):
    """Ask for structured JSON output."""
    return f"""Analyze the following text and return the result as JSON.

Text: "{text}"

Return JSON with this exact structure:
{schema}

Important:
- Return ONLY valid JSON, no additional text
- Use null for missing information
- Be concise and accurate"""

print("\n=== Structured Output Prompt ===")
print(structured_prompt(
    "Apple announced the iPhone 15 with a titanium body and 48MP camera, "
    "priced at $799, releasing September 22.",
    '{"product": "...", "company": "...", "price": ..., "features": [...], "release_date": "..."}'
))

# ─── 5. RAG Prompt Template ────────────────────
def rag_prompt(question, context_chunks):
    """RAG: answer based ONLY on provided context."""
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer using only the information in the context above
2. If the answer is not in the context, say "I don't know based on the provided information"
3. Cite which part of the context supports your answer
4. Be concise and accurate

Answer:"""

print("\n=== RAG Prompt ===")
print(rag_prompt(
    "What is the return policy?",
    ["Our return policy allows returns within 30 days of purchase.",
     "Items must be in original condition with receipt."]
))

# ─── 6. Agent Prompt (tool-using) ──────────────
def agent_prompt(task, tools, constraints=None):
    """Prompt for an AI agent that can use tools."""
    tool_descriptions = "\n".join(
        f"- {t['name']}: {t['description']}\n  Usage: {t['usage']}"
        for t in tools
    )
    return f"""You are an AI agent. Complete the task using available tools.

Task: {task}

Available tools:
{tool_descriptions}

Process:
1. Think about what you need to do
2. Choose a tool to use
3. Call the tool with the right parameters
4. Observe the result
5. Repeat until the task is complete

{f'Constraints: {constraints}' if constraints else ''}

Think:"""

tools = [
    {"name": "search", "description": "Search the web",
     "usage": "search(query: str) -> results"},
    {"name": "calculate", "description": "Evaluate math expression",
     "usage": "calculate(expression: str) -> number"},
    {"name": "summarize", "description": "Summarize text",
     "usage": "summarize(text: str) -> summary"},
]

print("\n=== Agent Prompt ===")
print(agent_prompt(
    "Find the population of Rwanda and calculate what percentage "
    "it is of the world population.",
    tools,
    "Use only factual data from reliable sources"
))

print("\n✓ Chapter 25 complete!")
```

---

## Best Practices

```
┌──────────────────────────────────────────────────────────────┐
│  PROMPT ENGINEERING BEST PRACTICES                           │
│                                                              │
│  1. BE SPECIFIC: "Write a 3-sentence summary" beats "summarize"│
│                                                              │
│  2. USE EXAMPLES: Few-shot > zero-shot for complex tasks      │
│                                                              │
│  3. SET CONSTRAINTS: "Answer in exactly 2 sentences"         │
│                                                              │
│  4. DEFINE ROLES: "You are an expert..." sets behavior        │
│                                                              │
│  5. ASK FOR REASONING: "Think step by step" for complex logic │
│                                                              │
│  6. REQUEST FORMAT: "Return as JSON" for structured output     │
│                                                              │
│  7. SAY WHAT NOT TO DO: "Do not include personal opinions"    │
│                                                              │
│  8. ITERATE: Test, measure, improve. Prompts are code.       │
│                                                              │
│  9. VERSION CONTROL: Track prompt changes and their effects   │
│                                                              │
│  10. TEST EDGE CASES: Empty input, long input, adversarial   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Chapter Summary

```
• Prompt patterns: zero-shot, few-shot, chain-of-thought, role
• System prompt sets behavior rules for the conversation
• Few-shot: show examples → better accuracy
• CoT: "think step by step" → better reasoning
• Structured: ask for JSON for programmatic use
• RAG: "answer based ONLY on provided context"
• Agent: "choose tools, call them, observe results"
• Prompts are code: version, test, iterate
```

---

## Interview Questions

1. "How would you improve LLM output quality through prompting?"
2. "What's chain-of-thought prompting and when is it useful?"
3. "How do you get an LLM to return structured JSON?"
4. "What's the difference between a system prompt and a user prompt?"
