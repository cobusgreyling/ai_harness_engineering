# AI Harness Engineering

![Agent Harness Engineering](images/header.svg)

A **harness** is the software system that governs how an AI Agent operates. It manages tools, memory, retries, context engineering and verification so the model can focus on reasoning.

This repo is a complete, working implementation of all six harness components — both as a CLI demo and a Gradio GUI — powered by **NVIDIA Nemotron Super 49B**.

---

## What's Inside

| File | Description |
|------|-------------|
| `harness-demo.py` | Full-featured CLI agent with 6 tools, retry loop, token tracking, audit log, and comparison mode |
| `harness-gui.py` | Gradio GUI with three tabs: Run Task, Compare Configs, and Audit Log |
| `harness-engineering-blog.md` | The full blog post on harness engineering |

---

## The Six Harness Components

| # | Component | Class | Role |
|---|-----------|-------|------|
| 1 | Tool Integration | `ToolRegistry` | Register and execute 6 tools (calculator, time, reverse, web search, file read, word count) |
| 2 | Memory & State | `MemoryManager` | Key-value working memory that persists results across steps |
| 3 | Context Engineering | `ContextEngine` | Dynamically assembles system prompt + tool descriptions + memory + retry feedback |
| 4 | Planning | `Planner` | LLM call to decompose a complex task into discrete steps |
| 5 | Verification | `Verifier` | Rule-based checks (numeric, contains, non-empty, single-sentence) with retry on failure |
| 6 | Modularity | `HarnessConfig` | Toggle any component ON/OFF independently |

---

## Features

- **6 Tools** — calculator, get_current_time, reverse_string, web_search, read_file, word_count
- **5 Diverse Example Tasks** — multi-step math, web search + calculation, file analysis, multi-tool chains, research + memory recall
- **Retry / Self-Correction** — failed verification triggers automatic retry with feedback injection (max 2 retries)
- **Token Tracking** — input/output token counts per LLM call with running totals
- **Structured Audit Log** — every LLM call, tool execution, verification, memory store, and retry recorded as timestamped JSON
- **Comparison Mode** — run the same task with 4 different harness configs (All ON, No Planner, No Verifier, No Memory) and compare side-by-side

---

## Architecture

```
User Task → Planner decomposes → For each step:
  ContextEngine builds prompt (system + tools + memory + retry feedback)
    → LLM call → Tool calls executed
      → Verifier checks → retry if failed (with feedback injection)
        → MemoryManager stores result
  → TokenTracker records usage
  → AuditLog records event
→ Final answer with stats + token breakdown + audit trail
```

---

## Quick Start

```bash
export NVIDIA_API_KEY="your-key"
pip install -r requirements.txt

# CLI demo
python3 harness-demo.py                # Run default task
python3 harness-demo.py --task 2       # Run task 2 (file analysis)
python3 harness-demo.py --all          # Run all 5 tasks
python3 harness-demo.py --compare      # Compare harness configs
python3 harness-demo.py --compare 1    # Compare on a specific task

# GUI
python3 harness-gui.py
```

---

## CLI Output

```
============================================================
  HARNESS DEMO — Nemotron Super 49B
============================================================

[CONFIG    ] Tools: ON | Memory: ON | Planner: ON | Verifier: ON | Retry: ON
[TASK      ] "What is 47 * 83? Then find the square root..."
[TOOLS     ] 6 registered: calculator, get_current_time, reverse_string, web_search, read_file, word_count

[PLANNER   ] Decomposed into 3 steps:
              1. Calculate 47 * 83
              2. Compute the square root of the product
              3. Summarise both results

[STEP 1/3  ] Calculate 47 * 83
[TOOL      ] calculator("47 * 83") = 3901
[VERIFY    ] ✓ Non-empty response
[MEMORY    ] Stored: step_1_result = 3901

[STEP 2/3  ] Compute the square root of the product
[TOOL      ] calculator("3901**0.5") = 62.457985878508765
[VERIFY    ] ✓ Non-empty response
[MEMORY    ] Stored: step_2_result = 62.457985878508765

[STEP 3/3  ] Summarise both results
[VERIFY    ] ✓ Non-empty response

============================================================
  FINAL ANSWER
============================================================
  47 × 83 = 3901, and √3901 ≈ 62.46.

============================================================
  TOKEN USAGE
============================================================
  Call 1 (planner): 85 in + 42 out = 127
  Call 2 (step_1): 310 in + 28 out = 338
  Call 3 (step_2): 335 in + 31 out = 366
  Call 4 (step_3): 380 in + 35 out = 415
  TOTAL: 1110 in + 136 out = 1246 tokens across 4 calls
```

---

## GUI

The Gradio interface has three tabs:

1. **Run Task** — configure harness components, select an example task or write your own, view execution log + final answer + memory + stats + token usage
2. **Compare Configs** — run the same task across 4 configurations and see a summary table
3. **Audit Log** — generate and view the full structured JSON audit trail

Launch with `python3 harness-gui.py` and open the URL shown in terminal.

---

## Author

Cobus Greyling
