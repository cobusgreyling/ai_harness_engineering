# The Rise of AI Harness Engineering

**AI Agents needed SDKs, then Frameworks, then Scaffolding. Now they need a Harness.**

I've written about the three architectural approaches to building AI Agents: [SDKs, Frameworks and Scaffolding](https://cobusgreyling.substack.com/p/architecting-agentic-ai-how-sdks).

Each one sits at a different point on the [flexibility-versus-structure](https://cobusgreyling.substack.com/p/architecting-agentic-ai-sdks-vs-frameworks) spectrum.

> A fourth pattern has emerged in 2026 that sits above all three. It's called a Harness.

Both [OpenAI](https://openai.com/index/harness-engineering/) and [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) are now using the term formally.

[Martin Fowler](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html) has written about it. An [arXiv paper](https://arxiv.org/abs/2603.05344) formalises it.

This is not a ***buzzword***, it's the missing architectural layer that determines whether AI Agents actually work in production.

> Harness Engineering is the missing architectural layer that determines whether AI Agents actually work in production.

## Bottom line

A harness is not the agent.

It's the software system that governs how the agent operates.

It manages the full lifecycle…tools, memory, retries, human approvals, context engineering, sub-agents…so the model can focus on reasoning.

[Philipp Schmid](https://www.philschmid.de/agent-harness-2026) put it best with a computer analogy…

The model is raw processing capability.

The context window is limited working memory.

The harness is the operating system…managing context, initialisation sequences and standard tool drivers.

The agent is the application that runs on top.

## Where a harness fits in the architecture stack

I previously covered [three architectural approaches](https://cobusgreyling.substack.com/p/architecting-agentic-ai-how-sdks) for building AI Agents.

Here is how a harness relates to each.

SDKs, Scaffolding and Frameworks answer the question of ***how you build*** an AI Agent.

A Harness answers a different question entirely, *how the agent runs*.

You can build a harness using any of the three. The harness is not a replacement for them. It's a layer above.

## Four approaches compared

## Six components of a harness

The [parallel.ai](https://parallel.ai/articles/what-is-an-agent-harness) team identified six core components…

This aligns with what both [OpenAI](https://openai.com/index/harness-engineering/) and [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) have published.

### Tool Integration Layer

Connects the model to external APIs, databases, code execution environments, and custom tools via defined protocols.

### Memory and State Management

Multi-layered memory (working context, session state, long-term memory) that persists beyond a single context window.

[Anthropic's approach](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) uses progress files and git history to bridge sessions.

### Context Engineering and Prompt Management

Dynamically curates what information appears in each model invocation.

Not static prompt templates, active context selection based on the current task state.

### Planning and Decomposition

Guides models through structured task sequences rather than attempting everything in one pass.

### Verification and Guardrails

Validation checks, format verification, safety filters. The self-correcting loop. When the agent struggles, the harness treats it as a signal to identify what's missing.

### Modularity and Extensibility

Pluggable components that can be enabled, disabled, or replaced independently.

## Real harnesses in production

Claude Code is a harness.

It reads entire codebases, manages filesystem access, spawns sub-agents, handles tool orchestration, maintains memory across sessions and implements guardrails.

> Developers focus on the task. The harness manages everything else.

[OpenAI Codex](https://openai.com/index/harness-engineering) uses harness engineering.

Their team built a codebase of over 1 million lines with ***no manually typed code at all***, treating the harness as the primary interface.

When the agent struggles, they feed improvements back into the repository. Context engineering, architectural constraints, and periodic cleanup agents form the core.

[OpenAI's CUA Sample App](https://github.com/openai/openai-cua-sample-app) is a harness for computer use.

The runner manages the screenshot → actions → verify → repeat loop.

The model decides what to do. The harness executes it safely.

## The Framework Layer is Collapsing into the Harness

In my recent piece on [the disappearing framework layer](https://cobusgreyling.substack.com/p/when-the-ai-framework-layer-disappears), I argued that models are absorbing capabilities traditionally handled by multi-agent frameworks.

Agent definition, message routing, task lifecycle, dependency management, spawning workers…roughly 80% of what developers use a framework for, the model now handles natively.

The remaining 20%: persistence, deterministic replay, cost control, observability, error recovery — is exactly what a harness provides.

The framework layer isn't just disappearing. It's splitting. The intelligence moves into the model. The infrastructure moves into the harness.

## Harness vs Framework

A framework tells the developer how to structure an application.

A harness tells the agent how to operate safely.

With a framework, the developer writes the orchestration logic.

With a harness, the model makes the plan. The harness keeps it on track.

## Practical Implications

For teams building AI Agents today, the question is shifting.

It's no longer "which framework should we use?" It's "what does our harness look like?"

The harness determines whether an agent succeeds or fails.

Great harnesses manage human approvals, filesystem access, tool orchestration, sub-agents, prompts, and lifecycle — intervening minimally but preventing catastrophic failures.

Start simple.

Build robust atomic tools. Let the model make the plan.

Add guardrails, retries, and verification.

That's harness engineering.

## Lastly

**Markdown/prompt harness** (like Anthropic's CLAUDE.md skills) embeds the orchestration instructions directly in the system prompt or structured markdown files.

The LLM itself becomes the loop controller — it reads the harness rules and follows them.

Best when the LLM is capable enough to self-direct and you want rapid iteration without code changes.

---

*[Chief AI Evangelist](https://www.linkedin.com/in/cobusgreyling/) @ [Kore.ai](https://blog.kore.ai/cobus-greyling/the-shifting-vocabulary-of-ai/) | I'm passionate about exploring the intersection of AI and language. From Language Models, AI Agents to Agentic Applications, Development Frameworks & Data-Centric Productivity Tools, I share insights and ideas on how these technologies are shaping the future.*
