# The Rise of Harness Engineering

**AI Agents needed SDKs, then Frameworks, then Scaffolding. Now they need a Harness.**

I've written about the three architectural approaches to building AI Agents — [SDKs, Frameworks, and Scaffolding](https://cobusgreyling.substack.com/p/architecting-agentic-ai-how-sdks). Each sits at a different point on the [flexibility-versus-structure spectrum](https://cobusgreyling.substack.com/p/architecting-agentic-ai-sdks-vs-frameworks).

A fourth pattern has emerged in 2026 that sits above all three. It's called a **Harness**.

Both [OpenAI](https://openai.com/index/harness-engineering/) and [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) are now using the term formally. [Martin Fowler](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html) has written about it. An [arXiv paper](https://arxiv.org/abs/2603.05344) formalises it. This is not a buzzword — it's the missing architectural layer that determines whether AI Agents actually work in production.

## Bottom Line

A harness is not the agent. It's the software system that governs how the agent operates. It manages the full lifecycle — tools, memory, retries, human approvals, context engineering, sub-agents — so the model can focus on reasoning.

[Philipp Schmid](https://www.philschmid.de/agent-harness-2026) put it best with a computer analogy:

```
┌─────────────────────────────────────────────┐
│                                             │
│   Model          =  CPU                     │
│   Context Window  =  RAM                    │
│   Agent Harness   =  Operating System       │
│   Agent           =  Application            │
│                                             │
└─────────────────────────────────────────────┘
```

The model is raw processing capability. The context window is limited working memory. The harness is the operating system — managing context, initialisation sequences, and standard tool drivers. The agent is the application that runs on top.

## Where a Harness Fits in the Architecture Stack

I previously covered [three architectural approaches](https://cobusgreyling.substack.com/p/architecting-agentic-ai-how-sdks) for building AI Agents. Here is how a harness relates to each.

```
┌─────────────────────────────────────────────────┐
│              How You BUILD                       │
│                                                  │
│   SDK ──────── Scaffolding ──────── Framework    │
│   (flexible)   (templates)     (opinionated)     │
│                                                  │
├─────────────────────────────────────────────────┤
│              How The Agent RUNS                   │
│                                                  │
│   ┌─────────────────────────────────────────┐   │
│   │            HARNESS                       │   │
│   │                                          │   │
│   │  Context Engineering                     │   │
│   │  Tool Orchestration                      │   │
│   │  Memory & State Management               │   │
│   │  Planning & Decomposition                │   │
│   │  Verification & Guardrails               │   │
│   │  Lifecycle Management                    │   │
│   │                                          │   │
│   └─────────────────────────────────────────┘   │
│                                                  │
└─────────────────────────────────────────────────┘
```

SDKs, Scaffolding, and Frameworks answer the question of **how you build** an AI Agent. A Harness answers a different question entirely — **how the agent runs**.

You can build a harness using any of the three. The harness is not a replacement for them. It's a layer above.

## Four Approaches Compared

| | SDK | Scaffolding | Framework | Harness |
|---|---|---|---|---|
| **Purpose** | Building blocks | Project templates | App architecture | Agent runtime |
| **Who controls flow** | Developer | Developer | Framework | Model |
| **Scope** | API access | Boilerplate setup | Full app lifecycle | Agent lifecycle |
| **Coupling** | Loose | Moderate | Tight | Wraps everything |
| **Focus** | "Give me the tools" | "Give me a starting point" | "Give me the structure" | "Keep the agent on track" |
| **Example** | OpenAI SDK, Anthropic SDK | create-next-app templates | LangChain, CrewAI | Claude Code, OpenAI Codex |

## Six Components of a Harness

The [parallel.ai](https://parallel.ai/articles/what-is-an-agent-harness) team identified six core components. This aligns with what both [OpenAI](https://openai.com/index/harness-engineering/) and [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) have published.

```
┌──────────────────────────────────────────────────────┐
│                    AGENT HARNESS                      │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ 1. Tool      │  │ 2. Memory &  │  │ 3. Context │ │
│  │ Integration  │  │ State Mgmt   │  │ Engineering│ │
│  │              │  │              │  │            │ │
│  │ APIs, DBs,   │  │ Working ctx, │  │ Dynamic    │ │
│  │ code exec,   │  │ session      │  │ prompt     │ │
│  │ filesystem   │  │ state, long- │  │ curation,  │ │
│  │              │  │ term memory  │  │ retrieval  │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ 4. Planning  │  │ 5. Verify &  │  │ 6. Module  │ │
│  │ & Decompose  │  │ Guardrails   │  │ Extension  │ │
│  │              │  │              │  │            │ │
│  │ Structured   │  │ Validation,  │  │ Pluggable  │ │
│  │ task         │  │ safety       │  │ components │ │
│  │ sequences    │  │ filters,     │  │ enable/    │ │
│  │              │  │ format check │  │ disable    │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**1. Tool Integration Layer** — Connects the model to external APIs, databases, code execution environments, and custom tools via defined protocols.

**2. Memory and State Management** — Multi-layered memory (working context, session state, long-term memory) that persists beyond a single context window. [Anthropic's approach](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) uses progress files and git history to bridge sessions.

**3. Context Engineering and Prompt Management** — Dynamically curates what information appears in each model invocation. Not static prompt templates — active context selection based on the current task state.

**4. Planning and Decomposition** — Guides models through structured task sequences rather than attempting everything in one pass.

**5. Verification and Guardrails** — Validation checks, format verification, safety filters. The self-correcting loop. When the agent struggles, the harness treats it as a signal to identify what's missing.

**6. Modularity and Extensibility** — Pluggable components that can be enabled, disabled, or replaced independently.

## Real Harnesses in Production

**Claude Code** is a harness. It reads entire codebases, manages filesystem access, spawns sub-agents, handles tool orchestration, maintains memory across sessions, and implements guardrails. Developers focus on the task. The harness manages everything else.

**[OpenAI Codex](https://openai.com/index/harness-engineering/)** uses harness engineering. Their team built a codebase of over 1 million lines with "no manually typed code at all" — treating the harness as the primary interface. When the agent struggles, they feed improvements back into the repository. Context engineering, architectural constraints, and periodic cleanup agents form the core.

**[OpenAI's CUA Sample App](https://github.com/openai/openai-cua-sample-app)** is a harness for computer use. The runner manages the screenshot → actions → verify → repeat loop. The model decides what to do. The harness executes it safely.

## The Framework Layer is Collapsing into the Harness

In my recent piece on [the disappearing framework layer](https://cobusgreyling.substack.com/p/when-the-ai-framework-layer-disappears), I argued that models are absorbing capabilities traditionally handled by multi-agent frameworks. Agent definition, message routing, task lifecycle, dependency management, spawning workers — roughly 80% of what developers use a framework for — the model now handles natively.

The remaining 20% — persistence, deterministic replay, cost control, observability, error recovery — is exactly what a harness provides.

```
┌─────────────────────────────────────────────────────┐
│         WHAT FRAMEWORKS USED TO DO                   │
│                                                      │
│   ┌────────────────────────┐ ┌────────────────────┐ │
│   │ Now handled by MODEL   │ │ Now handled by     │ │
│   │ (~80%)                 │ │ HARNESS (~20%)     │ │
│   │                        │ │                    │ │
│   │ - Define agents        │ │ - Persistence      │ │
│   │ - Route messages       │ │ - Replay           │ │
│   │ - Manage task lifecycle│ │ - Cost control     │ │
│   │ - Handle dependencies  │ │ - Observability    │ │
│   │ - Spawn/terminate      │ │ - Error recovery   │ │
│   └────────────────────────┘ └────────────────────┘ │
│                                                      │
└─────────────────────────────────────────────────────┘
```

The framework layer isn't just disappearing. It's splitting. The intelligence moves into the model. The infrastructure moves into the harness.

## Harness vs Framework — The Key Distinction

A framework tells the developer how to structure an application. A harness tells the agent how to operate safely.

With a framework, the developer writes the orchestration logic. With a harness, the model makes the plan. The harness keeps it on track.

|                          | Framework                  | Harness                    |
|--------------------------|----------------------------|----------------------------|
| **Controls**             | Application architecture   | Agent runtime behaviour    |
| **User**                 | Developer                  | The model itself           |
| **Flow**                 | Framework dictates flow    | Model dictates flow        |
| **Value**                | Productivity via convention| Reliability via guardrails |
| **When things go wrong** | Developer debugs           | Harness self-corrects      |

## Practical Implications

For teams building AI Agents today, the question is shifting.

It's no longer "which framework should we use?" It's "what does our harness look like?"

The harness determines whether an agent succeeds or fails. Great harnesses manage human approvals, filesystem access, tool orchestration, sub-agents, prompts, and lifecycle — intervening minimally but preventing catastrophic failures.

Start simple. Build robust atomic tools. Let the model make the plan. Add guardrails, retries, and verification. That's harness engineering.

---

*I'm passionate about exploring the intersection of AI and language. Chief Evangelist @ Kore.ai*

---

**References**

1. Martin Fowler — [Harness Engineering](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)
2. OpenAI — [Harness Engineering: Codex Agents](https://openai.com/index/harness-engineering/)
3. Anthropic — [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
4. Philipp Schmid — [The Importance of Agent Harness in 2026](https://www.philschmid.de/agent-harness-2026)
5. Parallel.ai — [What is an Agent Harness?](https://parallel.ai/articles/what-is-an-agent-harness)
6. arXiv — [Building AI Coding Agents: Scaffolding, Harness, Context Engineering](https://arxiv.org/abs/2603.05344)
7. Aakash Gupta — [2025 Was Agents. 2026 Is Agent Harnesses.](https://aakashgupta.medium.com/2025-was-agents-2026-is-agent-harnesses-heres-why-that-changes-everything-073e9877655e)

**Previous writings referenced**

8. Cobus Greyling — [Architecting Agentic AI: How SDKs, Scaffolding & Frameworks Are Different](https://cobusgreyling.substack.com/p/architecting-agentic-ai-how-sdks)
9. Cobus Greyling — [Architecting Agentic AI — SDKs vs. Frameworks](https://cobusgreyling.substack.com/p/architecting-agentic-ai-sdks-vs-frameworks)
10. Cobus Greyling — [When The AI Framework Layer Disappears](https://cobusgreyling.substack.com/p/when-the-ai-framework-layer-disappears)
