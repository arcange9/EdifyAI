# Chapter 29 — LLM Application Architecture

## Learning Objectives

- Understand LLM application architecture patterns
- Know the components of a production LLM system
- Design an LLM application from scratch
- Choose between different architecture approaches

---

## Why This Matters

Calling an LLM API is 5 lines of code. Building a *reliable, scalable* LLM application requires architecture. This chapter bridges the gap between "I can call an API" and "I can build a production AI system."

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  PRODUCTION LLM APPLICATION ARCHITECTURE                   │
│                                                              │
│  ┌──────────┐                                                │
│  │  USER     │                                                │
│  │  INTERFACE│                                                │
│  └────┬─────┘                                                │
│       │                                                       │
│       ▼                                                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                   │
│  │  API     │──→│  LLM     │──→│  Response│                   │
│  │  Layer   │   │  Service │   │  Formatter│                  │
│  └──────────┘   └──────────┘   └──────────┘                   │
│       │              │                                         │
│       │         ┌────┴────┐                                    │
│       │         │  Prompt │                                    │
│       │         │  Builder│                                    │
│       │         └────┬────┘                                    │
│       │              │                                         │
│       │    ┌─────────┼─────────┐                               │
│       │    ▼         ▼         ▼                               │
│       │ ┌──────┐ ┌──────┐ ┌──────┐                           │
│       │ │ RAG  │ │Tools │ │Memory│                           │
│       │ │Pipeline│ │Engine│ │Store │                           │
│       │ └──┬───┘ └──┬───┘ └──┬───┘                           │
│       │    │        │        │                                │
│       │    ▼        │        │                                │
│       │ ┌──────┐    │        │                                │
│       │ │Vector │    │        │                                │
│       │ │ DB   │    │        │                                │
│       │ └──────┘    │        │                                │
│       │             ▼        │                                 │
│       │        ┌──────┐      │                                 │
│       │        │External│    │                                 │
│       │        │APIs  │      │                                 │
│       │        └──────┘      │                                 │
│       │                      ▼                                 │
│       │                ┌──────┐                                │
│       │                │ Chat │                                │
│       │                │History│                               │
│       │                │DB   │                                │
│       └────────────────┘                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Common Architecture Patterns

```
┌──────────────────────────────────────────────────────────────┐
│  LLM APP PATTERNS                                           │
│                                                              │
│  1. DIRECT API CALL (simplest)                              │
│     User → API → LLM → Response                             │
│     Use: simple chatbot, Q&A without context               │
│                                                              │
│  2. RAG (Retrieval-Augmented)                               │
│     User → Retrieve docs → Augment prompt → LLM → Response │
│     Use: document Q&A, knowledge base, customer support    │
│                                                              │
│  3. AGENT (tool-using)                                      │
│     User → Agent → Think → Use tool → Observe → LLM → Done │
│     Use: complex tasks, web search, calculations            │
│                                                              │
│  4. MULTI-AGENT                                              │
│     User → Coordinator → Agent1, Agent2, Agent3 → Result   │
│     Use: complex workflows, research, code generation      │
│                                                              │
│  5. STREAMING                                               │
│     User → API → LLM (token by token) → Live display       │
│     Use: chat interfaces, real-time UX                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — LLM Application Components

```python
"""
Chapter 29 — LLM Application Architecture
Shows the structure of a production LLM application.
pip install openai numpy
"""
import os
import json
from typing import List, Dict, Optional

# ─── Core abstraction: LLM Client ──────────────
class LLMClient:
    """Abstract LLM client — works with any provider."""
    def __init__(self, model="gpt-4o-mini", api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "demo")

    def chat(self, messages: List[Dict], temperature=0.7, max_tokens=500) -> str:
        """Send a chat request to the LLM."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except ImportError:
            # Demo mode (no API key)
            return f"[Demo mode] Would call {self.model} with: {messages[-1]['content'][:50]}..."

# ─── Prompt Builder ────────────────────────────
class PromptBuilder:
    """Builds structured prompts for the LLM."""
    @staticmethod
    def system_prompt(role="assistant", constraints=None):
        prompt = f"You are a helpful {role}. "
        if constraints:
            prompt += f" {constraints}"
        return {"role": "system", "content": prompt}

    @staticmethod
    def rag_prompt(question, context_chunks):
        context = "\n\n".join(context_chunks)
        return {
            "role": "user",
            "content": f"""Context: {context}

Question: {question}

Answer based ONLY on the context above. If the answer isn't in the context, say 'I don't know'."""
        }

# ─── Complete LLM Application ──────────────────
class LLMApplication:
    """A complete LLM application with conversation history."""
    def __init__(self, model="gpt-4o-mini"):
        self.llm = LLMClient(model=model)
        self.history: List[Dict] = []

    def chat(self, user_message: str, system_role="assistant") -> str:
        messages = [PromptBuilder.system_prompt(system_role)]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        response = self.llm.chat(messages)

        # Save to history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        self.history = []

# ─── Demo ──────────────────────────────────────
app = LLMApplication()
print("=== LLM Application Demo ===\n")
print("(Running in demo mode — set OPENAI_API_KEY for real responses)\n")

response = app.chat("What is AI engineering?")
print(f"Q: What is AI engineering?")
print(f"A: {response}\n")

response = app.chat("How is it different from data science?")
print(f"Q: How is it different from data science?")
print(f"A: {response}\n")

print(f"History length: {len(app.history)} messages")

print("\n✓ Chapter 29 complete!")
```

---

## Chapter Summary

```
• LLM app = API layer + LLM service + prompt builder + data stores
• Patterns: direct, RAG, agent, multi-agent, streaming
• Conversation history enables multi-turn chat
• System prompt sets behavior, user prompt is the query
• Production needs: error handling, rate limiting, caching
```

---

## Interview Questions

1. "Describe the architecture of a production LLM application."
2. "When would you use RAG vs a direct LLM call?"
3. "How do you manage conversation history?"
4. "What components does a production AI system need?"
