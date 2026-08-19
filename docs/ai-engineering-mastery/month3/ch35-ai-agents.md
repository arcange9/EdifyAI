# Chapter 35 — AI Agents: Architecture & Tool-Using Agents

## Learning Objectives

- Understand agent architecture and the reasoning loop
- Build agents that use tools to accomplish real tasks
- Implement ReAct and Plan-and-Execute patterns
- Know when to use agents vs simple RAG

---

## Why This Matters

Agents are the next evolution beyond RAG. While RAG retrieves and answers, agents *act* — they can search the web, run code, query databases, and chain multiple tools to solve complex problems. Agent frameworks (LangChain, CrewAI, AutoGPT) are exploding in popularity.

---

## Agent Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  AI AGENT ARCHITECTURE                                       │
│                                                              │
│  ┌──────────┐                                                │
│  │  USER     │ "Find the weather in Kigali and               │
│  │           │  suggest what to wear"                        │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────┐            │
│  │  AGENT LOOP (LLM as brain)                   │            │
│  │                                               │            │
│  │  1. PERCEIVE: understand the task              │            │
│  │  2. THINK: decide what to do next              │            │
│  │  3. ACT: call a tool                           │            │
│  │  4. OBSERVE: read the result                   │            │
│  │  5. REPEAT until done                         │            │
│  │  6. RESPOND: give final answer                 │            │
│  │                                               │            │
│  └──────┬──────┬──────┬──────┬──────┐             │            │
│         │      │      │      │      │             │            │
│         ▼      ▼      ▼      ▼      ▼             │            │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐        │            │
│  │Search││Calc  ││DB   ││Code ││Email│        │            │
│  │Tool  ││Tool  ││Query││Exec ││Tool │        │            │
│  │      ││      ││Tool ││Tool ││      │        │            │
│  └──────┘└──────┘└──────┘└──────┘└──────┘        │            │
│  TOOLS (the agent's hands and eyes)               │            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Agent Patterns

```
┌──────────────────────────────────────────────────────────────┐
│  AGENT PATTERNS                                              │
│                                                              │
│  ReAct (Reasoning + Acting):                                │
│  Think → Act → Observe → Think → Act → ... → Final          │
│  Simple, effective, most common                             │
│                                                              │
│  Plan-and-Execute:                                          │
│  1. Plan all steps upfront                                   │
│  2. Execute each step                                        │
│  3. Replan if something fails                               │
│  Better for complex, multi-step tasks                        │
│                                                              │
│  Reflexion:                                                 │
│  1. Attempt task                                             │
│  2. Evaluate result                                          │
│  3. If failed, reflect on why                               │
│  4. Try again with reflection                               │
│  Self-improving agent                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Code — Tool-Using Agent

```python
"""
Chapter 35 — AI Agent with Tools
Builds a complete agent that can use tools to solve tasks.
"""
import json
from typing import List, Dict, Callable

class Tool:
    """Represents a tool the agent can use."""
    def __init__(self, name: str, description: str, func: Callable, args_schema: str):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema

    def run(self, **kwargs):
        """Execute the tool."""
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"Error: {e}"

# ─── Define tools ──────────────────────────────
def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed = "0123456789+-*/()., "
    if all(c in allowed for c in expression):
        return str(eval(expression))
    return "Error: Invalid expression"

def search_knowledge(query: str) -> str:
    """Search a knowledge base (simulated)."""
    knowledge = {
        "python": "Python is a programming language great for AI.",
        "rag": "RAG combines retrieval with LLM generation.",
        "neural": "Neural networks are inspired by the human brain.",
        "transformer": "Transformers use attention mechanisms.",
    }
    for key, val in knowledge.items():
        if key in query.lower():
            return val
    return f"No results found for '{query}'"

def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    return f"File '{filename}' saved with {len(content)} characters."

tools = [
    Tool("calculator", "Evaluate math expressions", calculator, "expression: str"),
    Tool("search", "Search knowledge base", search_knowledge, "query: str"),
    Tool("write_file", "Write content to a file", write_file, "filename: str, content: str"),
]

# ─── Agent ─────────────────────────────────────
class Agent:
    """AI Agent using the ReAct pattern."""
    def __init__(self, tools: List[Tool], llm_chat=None):
        self.tools = {t.name: t for t in tools}
        self.llm_chat = llm_chat
        self.max_steps = 10

    def build_system_prompt(self) -> str:
        tool_descs = "\n".join(
            f"- {t.name}: {t.description}\n  Args: {t.args_schema}"
            for t in self.tools.values()
        )
        return f"""You are an AI agent. Use tools to complete tasks.

Available tools:
{tool_descs}

Format:
THOUGHT: [your reasoning about what to do]
ACTION: {{"tool": "tool_name", "args": {{"param": "value"}}}}
OBSERVATION: [tool result will appear here]
... (repeat as needed)
FINAL: [your final answer to the user]

Always use the format above. Think step by step."""

    def run(self, task: str) -> str:
        """Execute a task using tools."""
        print(f"\n{'='*50}")
        print(f"TASK: {task}")
        print(f"{'='*50}\n")

        system_prompt = self.build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]

        # Simulated agent execution (in production, use real LLM)
        # Demo: search for RAG, then explain
        if self.llm_chat:
            for step in range(self.max_steps):
                response = self.llm_chat(messages)
                if "FINAL:" in response:
                    return response.split("FINAL:")[-1].strip()
                # Parse and execute tools...
                messages.append({"role": "assistant", "content": response})
        else:
            # Demo without LLM
            print("DEMO MODE (set LLM for real execution):\n")

            # Step 1: Search
            print("THOUGHT: I need to search for information first")
            result = self.tools["search"].run(query="rag")
            print(f"ACTION: search(query='rag')")
            print(f"OBSERVATION: {result}\n")

            # Step 2: Write summary
            print("THOUGHT: Now I'll save the information")
            result = self.tools["write_file"].run(
                filename="output.txt", content=result)
            print(f"ACTION: write_file(filename='output.txt', content='...')")
            print(f"OBSERVATION: {result}\n")

            print("FINAL: Task completed! Found information about RAG and saved it.")
            return "Task completed!"

# ─── Run the agent ──────────────────────────────
agent = Agent(tools)
agent.run("What is RAG and save the answer to a file")

print("\n✓ Chapter 35 complete!")
```

---

## Mini-Project — AI Agent with Tools

Build an agent that can:
1. Search a knowledge base
2. Calculate math
3. Write files
4. Answer multi-step questions by chaining tools

---

## Chapter Summary

```
• Agent = LLM + tools + reasoning loop
• ReAct: Think → Act → Observe → Repeat → Final
• Tools are functions the agent can call
• Agent decides WHICH tool to use and HOW
• RAG answers questions; agents accomplish tasks
• Production: LangChain, CrewAI, or custom framework
```

---

## Interview Questions

1. "What's the difference between RAG and an AI agent?"
2. "Explain the ReAct pattern."
3. "How do you prevent agents from looping forever?"
4. "When would you use an agent vs a simple API call?"
