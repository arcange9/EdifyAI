# Chapter 36 — Multi-Agent Systems & Memory

## Learning Objectives

- Understand multi-agent coordination and delegation
- Implement short-term and long-term memory for agents
- Know when multi-agent beats single-agent

---

## Multi-Agent Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  MULTI-AGENT SYSTEM                                          │
│                                                              │
│  ┌──────────┐                                                │
│  │ COORDINATOR│ ← receives task, delegates to specialists    │
│  │  AGENT    │                                                │
│  └──┬──┬──┬─┘                                                │
│     │  │  │                                                   │
│     ▼  ▼  ▼                                                   │
│  ┌────┐┌────┐┌────┐                                         │
│  │RE  ││CODE││TEST│ ← each agent has a specialty             │
│  │AGT ││AGT ││AGT │                                         │
│  └────┘└────┘└────┘                                         │
│     │  │  │                                                   │
│     └──┼──┘                                                   │
│        ▼                                                      │
│  ┌──────────┐                                                │
│  │COORDINATOR│ ← combines results, sends to user            │
│  └──────────┘                                                │
│                                                              │
│  Use cases:                                                  │
│  • Research (search agent + analysis agent + writer)       │
│  • Software (coder + tester + reviewer)                     │
│  • Customer service (triage + resolver + escalator)         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Memory Types

```
┌──────────────────────────────────────────────────────────────┐
│  AGENT MEMORY TYPES                                         │
│                                                              │
│  SHORT-TERM (working memory):                               │
│  Current conversation context, recent actions                │
│  Stored in: message list / context window                   │
│  Lost when: session ends                                    │
│                                                              │
│  LONG-TERM (persistent):                                    │
│  Past interactions, user preferences, learned facts        │
│  Stored in: vector database / key-value store              │
│  Retrieved: by semantic similarity to current context      │
│                                                              │
│  EPISODIC:                                                  │
│  Record of past events and actions                          │
│  "Last time user asked about Python, I explained X"        │
│  Stored in: conversation history DB                          │
│                                                              │
│  SEMANTIC:                                                  │
│  General knowledge and facts                                 │
│  "User prefers Python over Java"                            │
│  Stored in: knowledge graph / vector DB                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Chapter Summary

```
• Multi-agent: coordinator delegates to specialist agents
• Each agent has a role (research, code, review)
• Short-term memory: current context (message list)
• Long-term memory: vector DB of past interactions
• Episodic: what happened; Semantic: what we learned
• Use multi-agent when task is complex enough to benefit from specialization
```

## Interview Questions
1. "When would you use multi-agent instead of single-agent?"
2. "How do you implement memory for an AI agent?"
3. "What are the types of agent memory?"
