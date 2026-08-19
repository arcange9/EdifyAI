# Chapter 40 — Monitoring, Cost Optimization & Security

## Learning Objectives

- Monitor AI applications in production (latency, quality, errors)
- Optimize API costs and model usage
- Implement security best practices for AI systems

---

## Monitoring

```
┌──────────────────────────────────────────────────────────────┐
│  WHAT TO MONITOR IN PRODUCTION AI                           │
│                                                              │
│  PERFORMANCE:                                                │
│  • Latency: time to first token, total response time       │
│  • Throughput: requests per second                         │
│  • Error rate: 4xx, 5xx, timeouts                          │
│                                                              │
│  QUALITY:                                                    │
│  • User feedback (thumbs up/down)                          │
│  • Hallucination rate (sampled audits)                     │
│  • RAG retrieval relevance scores                          │
│                                                              │
│  COST:                                                       │
│  • Tokens consumed per request                             │
│  • API cost per user per day                               │
│  • Model comparison (cheaper model vs quality tradeoff)    │
│                                                              │
│  SAFETY:                                                     │
│  • Guardrail trigger rate                                   │
│  • Prompt injection attempts                               │
│  • Toxic output detection                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Cost Optimization

```
┌──────────────────────────────────────────────────────────────┐
│  COST OPTIMIZATION STRATEGIES                              │
│                                                              │
│  1. MODEL TIERING                                           │
│     Simple questions → small model (gpt-4o-mini, $0.15/M)  │
│     Complex reasoning → large model (gpt-4o, $2.50/M)     │
│     → Route based on question complexity                    │
│                                                              │
│  2. CACHING                                                  │
│     Cache identical questions → return cached answer       │
│     TTL: 1 hour for general, permanent for factual          │
│                                                              │
│  3. COMPRESSION                                              │
│     Shorter prompts → fewer input tokens → lower cost      │
│     Summarize long context before sending to LLM           │
│                                                              │
│  4. LOCAL MODELS                                             │
│     Use Ollama for common tasks (free!)                     │
│     Cloud API only when local model quality isn't enough  │
│                                                              │
│  5. BATCH PROCESSING                                         │
│     Batch API (50% cheaper than real-time)                 │
│     For non-urgent tasks like document processing          │
│                                                              │
│  6. TOKEN OPTIMIZATION                                       │
│     Reduce max_tokens, use concise system prompts          │
│     Monitor and set token budgets per user                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Security

```
┌──────────────────────────────────────────────────────────────┐
│  AI SECURITY CHECKLIST                                      │
│                                                              │
│  1. API KEYS: Never hardcode. Use env vars / secret manager│
│  2. PROMPT INJECTION: Detect and block malicious prompts    │
│  3. PII PROTECTION: Redact personal info before/after LLM   │
│  4. RATE LIMITING: Prevent abuse, control costs            │
│  5. OUTPUT FILTERING: Block harmful/toxic content           │
│  6. AUTHENTICATION: Require auth for API access            │
│  7. AUDIT LOGGING: Log all requests and responses          │
│  8. DATA ENCRYPTION: Encrypt at rest and in transit        │
│  9. MODEL VERSIONING: Track which model produced output    │
│  10. FALLBACK PLAN: What happens if API is down?           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Chapter Summary

```
• Monitor: latency, error rate, quality, cost, safety
• Cost: model tiering, caching, local models, batch, token limits
• Security: keys in env vars, injection detection, PII, rate limits
• Always have a fallback when the AI API goes down
```
