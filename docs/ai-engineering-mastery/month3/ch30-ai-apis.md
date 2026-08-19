# Chapter 30 — AI APIs & OpenAI-Compatible APIs

## Learning Objectives

- Call LLM APIs from Python
- Understand the OpenAI API format (the industry standard)
- Build a provider-agnostic client that works with multiple APIs
- Handle streaming, errors, and rate limits

---

## Why This Matters

APIs are how you integrate LLMs into applications. The OpenAI API format has become the industry standard — OpenRouter, Groq, local models via Ollama, and many others all use the same format. Learn one API, use any provider.

---

## Code — Universal LLM Client

```python
"""
Chapter 30 — AI APIs & OpenAI-Compatible APIs
A universal client that works with any OpenAI-compatible provider.
pip install openai
"""
import os
from typing import Optional, List, Dict

class UniversalLLMClient:
    """Works with OpenAI, OpenRouter, Groq, Ollama, and any compatible API."""

    PROVIDERS = {
        "openai": {"base_url": None, "env_key": "OPENAI_API_KEY"},
        "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                       "env_key": "OPENROUTER_API_KEY"},
        "groq": {"base_url": "https://api.groq.com/openai/v1",
                 "env_key": "GROQ_API_KEY"},
        "ollama": {"base_url": "http://localhost:11434/v1",
                   "env_key": None},  # Local, no key needed
    }

    def __init__(self, provider="openai", model=None, api_key=None):
        config = self.PROVIDERS.get(provider, self.PROVIDERS["openai"])
        self.base_url = config["base_url"]
        self.api_key = api_key or os.environ.get(config["env_key"], "demo")
        self.model = model or "gpt-4o-mini"
        self.provider = provider

    def chat(self, messages, temperature=0.7, max_tokens=500, stream=False):
        """Send chat messages to the LLM."""
        try:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            client = OpenAI(**kwargs)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            if stream:
                result = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        result += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print()
                return result
            else:
                return response.choices[0].message.content
        except Exception as e:
            return f"[API Error: {e}. Set your API key to use this.]"

    def stream_chat(self, messages, **kwargs):
        """Stream response token by token."""
        return self.chat(messages, stream=True, **kwargs)

# ─── Demo ──────────────────────────────────────
print("=== Universal LLM Client ===\n")

# Same interface, different providers:
providers = [
    ("openai", "gpt-4o-mini"),
    ("groq", "llama-3.1-8b-instant"),
    ("ollama", "llama3.2"),
]

messages = [
    {"role": "system", "content": "You are a helpful AI tutor."},
    {"role": "user", "content": "Explain neural networks in one sentence."}
]

for provider, model in providers:
    client = UniversalLLMClient(provider=provider, model=model)
    print(f"\n{provider} ({model}):")
    response = client.chat(messages, temperature=0.7, max_tokens=100)
    print(f"  {response[:80]}...")

# ─── Error handling ─────────────────────────────
print("\n=== Error Handling ===\n")

def safe_chat(client, messages, retries=3):
    """Chat with retry logic for production."""
    for attempt in range(retries):
        try:
            response = client.chat(messages)
            return response
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                return f"Error after {retries} attempts: {e}"
    return None

# ─── Cost estimation ────────────────────────────
print("\n=== Cost Estimation ===\n")

PRICING = {  # per 1M tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}

def estimate_cost(model, input_tokens, output_tokens):
    if model not in PRICING:
        return "Unknown pricing"
    cost = (input_tokens * PRICING[model]["input"] +
            output_tokens * PRICING[model]["output"]) / 1_000_000
    return f"${cost:.4f}"

for model in PRICING:
    cost = estimate_cost(model, 1000, 500)
    print(f"  {model}: 1K input + 500 output = {cost}")

print("\n✓ Chapter 30 complete!")
```

---

## Chapter Summary

```
• OpenAI API format = industry standard (one interface, any provider)
• Providers: OpenAI, OpenRouter, Groq, Ollama (local)
• Streaming: token-by-token for real-time UX
• Error handling: retries, rate limiting, fallbacks
• Cost: track input + output tokens × per-token pricing
• Free option: Ollama (local, no API key, no cost)
```
