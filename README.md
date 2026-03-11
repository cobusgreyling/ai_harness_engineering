# AI Harness Engineering

A minimal agent harness implementation demonstrating six core harness components using NVIDIA Nemotron 3 Super.

## Summary

A **harness** is the software system that governs how an AI Agent operates. It manages tools, memory, retries, context engineering and verification so the model can focus on reasoning.

This repo contains two things:

1. **`harness-demo.py`** — A single-file Python demo implementing all six harness components as a working CLI agent
2. **`harness-engineering-blog.md`** — The full blog post on harness engineering

## The Six Harness Components

| Component | Class | Role |
|---|---|---|
| Tool Integration | `ToolRegistry` | Register and execute tools (calculator, get_current_time, reverse_string) |
| Memory & State | `MemoryManager` | Key-value working memory that persists results across steps |
| Context Engineering | `ContextEngine` | Dynamically assembles system prompt + tool descriptions + relevant memory per step |
| Planning | `Planner` | LLM call to decompose a complex task into discrete steps |
| Verification | `Verifier` | Rule-based checks (numeric, contains-values, single-sentence) |
| Modularity | `HarnessConfig` | Toggle any component ON/OFF independently |

## Architecture

```
User Task → Planner decomposes → For each step:
  ContextEngine builds prompt → LLM call → Tool calls executed →
  Verifier checks → MemoryManager stores result
→ Final answer with stats
```

## Running the Demo

```bash
export NVIDIA_API_KEY="your-key"
python3 harness-demo.py
```

Requires: `openai` Python package, NVIDIA NIM API access for Nemotron 3 Super.

## Sample Output

```
══════════════════════════════════════════════════
  HARNESS DEMO — Nemotron 3 Super
══════════════════════════════════════════════════

[CONFIG    ] Tools: ON | Memory: ON | Planner: ON | Verifier: ON
[TASK      ] "What is 47 * 83? Then find the square root..."

[PLANNER   ] Decomposed into 3 steps:
              1. Calculate 47 * 83
              2. Compute the square root of the product
              3. Summarise both results

[STEP 1/3  ] Calculate 47 * 83
[TOOL      ] calculator("47 * 83") = 3901
[VERIFY    ] ✓ Result is numeric
[MEMORY    ] Stored: multiplication_result = 3901

[STEP 2/3  ] Compute the square root of the product
[TOOL      ] calculator("3901**0.5") = 62.457985878508765
[VERIFY    ] ✓ Result is numeric
[MEMORY    ] Stored: sqrt_result = 62.457985878508765

[STEP 3/3  ] Summarise both results
[VERIFY    ] ✓ Contains all expected values
[VERIFY    ] ✓ Is a single sentence

══════════════════════════════════════════════════
  FINAL ANSWER
══════════════════════════════════════════════════
  The multiplication result is 3901 and the square root
  result is approximately 62.458.

  Steps: 3/3 | Tool calls: 2 | Memory: 2 | Verifications: 4/4
```

## Author

Cobus Greyling — [Chief AI Evangelist @ Kore.ai](https://www.linkedin.com/in/cobusgreyling/)
