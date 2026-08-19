# Chapter 31 — Structured Outputs & Function/Tool Calling

## Learning Objectives

- Get reliable JSON output from LLMs
- Implement function/tool calling for AI agents
- Build an AI that can use tools to accomplish tasks
- Understand the agent reasoning loop

---

## Why This Matters

Raw LLM output is unstructured text. Production AI systems need structured data (JSON) to pass to databases, APIs, and other code. Tool calling is what turns an LLM from a chatbot into an *agent* that can take actions.

---

## Code — Structured Outputs & Tools

```python
"""
Chapter 31 — Structured Outputs & Function/Tool Calling
pip install openai json
"""
import json
import os

# ─── 1. Structured Output (JSON) ───────────────
def get_structured_response(client, prompt, schema):
    """Get structured JSON from an LLM."""
    system = f"""You must respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Return ONLY valid JSON, no markdown, no explanation."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    response = client.chat(messages, temperature=0)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Retry with stricter instructions
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content":
            "That was not valid JSON. Return ONLY JSON, no other text."})
        return json.loads(client.chat(messages, temperature=0))

# ─── 2. Tool Definition ────────────────────────
TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {"expression": "str"},
        "function": lambda expression: str(eval(expression))
    },
    {
        "name": "search_knowledge",
        "description": "Search a knowledge base for information",
        "parameters": {"query": "str"},
        "function": lambda query: f"Results for '{query}': Python is great for AI."
    },
    {
        "name": "save_note",
        "description": "Save a note to the database",
        "parameters": {"title": "str", "content": "str"},
        "function": lambda title, content: f"Saved note '{title}'"
    },
]

# ─── 3. Agent Loop (ReAct pattern) ──────────────
class SimpleAgent:
    """An AI agent that can use tools to complete tasks."""

    def __init__(self, client, tools):
        self.llm = client
        self.tools = {t["name"]: t for t in tools}

    def run(self, task, max_steps=5):
        """Execute a task using tools."""
        print(f"\nTask: {task}\n")

        system_prompt = f"""You are an AI agent with tools.
Available tools: {list(self.tools.keys())}

Process:
1. Think about what to do
2. Call a tool: TOOL_CALL: tool_name(arg1=val1, arg2=val2)
3. Observe the result
4. Repeat or give final answer

Always use this format:
THOUGHT: [your reasoning]
TOOL_CALL: [tool_name]([arguments])
OBSERVE: [tool result]
... (repeat as needed)
FINAL: [your final answer]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]

        for step in range(max_steps):
            response = self.llm.chat(messages, temperature=0)
            print(f"  Step {step+1}:\n  {response[:200]}...\n")

            if "FINAL:" in response:
                final = response.split("FINAL:")[-1].strip()
                return final

            # Parse tool calls (simplified)
            if "TOOL_CALL:" in response:
                tool_line = [l for l in response.split("\n")
                            if l.strip().startswith("TOOL_CALL:")][0]
                tool_call = tool_line.split("TOOL_CALL:")[-1].strip()

                # Parse: tool_name(arg1=val1, arg2=val2)
                tool_name = tool_call.split("(")[0].strip()
                args_str = tool_call.split("(")[-1].rstrip(")")

                if tool_name in self.tools:
                    # Simple arg parsing
                    try:
                        if args_str.startswith("'") or args_str.startswith('"'):
                            args = [args_str.strip("'\"")]
                        else:
                            args = [a.strip().strip("'\"") for a in args_str.split(",")]

                        result = self.tools[tool_name]["function"](*args)
                        print(f"  OBSERVE: {result}\n")

                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content":
                            f"OBSERVE: {result}"})
                    except Exception as e:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content":
                            f"Tool error: {e}"})

        return "Max steps reached without final answer."

# ─── Demo ──────────────────────────────────────
print("=== Structured Outputs & Agent Demo ===\n")

# Demo structured output schema
schema = {
    "sentiment": "string (positive/negative/neutral)",
    "confidence": "number (0-1)",
    "key_topics": ["array of strings"]
}

# Demo agent
from month3.ch30_ai_apis import UniversalLLMClient  # (simulated)

class DemoClient:
    """Simulated LLM client for demo purposes."""
    def chat(self, messages, temperature=0.7, max_tokens=500):
        return '''THOUGHT: I need to calculate the total
TOOL_CALL: calculate('15 * 12 * 0.9')
OBSERVE: 162.0
FINAL: 15 items at $12 each with 10% discount = $162'''

agent = SimpleAgent(DemoClient(), TOOLS)
result = agent.run("Calculate: 15 items at $12 each with 10% discount")
print(f"Final answer: {result}")

print("\n✓ Chapter 31 complete!")
```

---

## Chapter Summary

```
• Structured output: prompt the LLM to return valid JSON
• Tools: give the LLM functions it can call
• Agent loop: Think → Call tool → Observe → Repeat → Final answer
• ReAct pattern: Reasoning + Acting interleaved
• Temperature 0 for structured/tool-calling responses
```

---

## Interview Questions

1. "How do you get reliable JSON from an LLM?"
2. "What is function/tool calling and how does it work?"
3. "Explain the ReAct agent pattern."
4. "How would you build an AI agent that can use multiple tools?"
