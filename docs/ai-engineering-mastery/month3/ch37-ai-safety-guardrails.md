# Chapter 37 — AI Safety, Guardrails & Hallucination Reduction

## Learning Objectives

- Understand hallucination causes and prevention
- Implement guardrails for AI applications
- Know AI safety best practices for production

---

## Why This Matters

AI systems can produce harmful, false, or biased outputs. In production, you're responsible for what your AI says. Guardrails, monitoring, and safety measures are not optional — they're engineering requirements.

---

## Hallucination Reduction

```
┌──────────────────────────────────────────────────────────────┐
│  HALLUCINATION REDUCTION STRATEGIES                         │
│                                                              │
│  1. RAG: Ground answers in retrieved documents               │
│  2. System prompt: "Say 'I don't know' if unsure"          │
│  3. Low temperature: 0.0-0.3 for factual tasks               │
│  4. Citations: Require source attribution                    │
│  5. Confidence scoring: Ask model to rate confidence        │
│  6. Self-checking: Ask model to verify its own answer       │
│  7. Human review: Flag low-confidence answers                │
│  8. Content filtering: Block certain topics                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Guardrails

```
┌──────────────────────────────────────────────────────────────┐
│  GUARDRAILS                                                 │
│                                                              │
│  INPUT GUARDRAILS (before LLM):                              │
│  • Prompt injection detection                               │
│  • PII (personal info) redaction                            │
│  • Content filtering (block harmful requests)              │
│  • Rate limiting                                            │
│  • Input length limits                                      │
│                                                              │
│  OUTPUT GUARDRAILS (after LLM):                              │
│  • Hallucination detection                                  │
│  • Toxicity filtering                                       │
│  • PII detection in responses                              │
│  • Content policy checking                                  │
│  • Fact-checking against knowledge base                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Chapter Summary

```
• Hallucination: RAG, low temperature, citations, self-check
• Input guardrails: injection detection, PII redaction, filtering
• Output guardrails: toxicity, fact-check, content policy
• Safety is an engineering requirement, not optional
```
