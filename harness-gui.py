"""
Agent Harness GUI — Gradio frontend for the Agent Harness
Uses shared components from harness_core.py.

Features:
  - Run Task tab with all config toggles (original + new)
  - Compare Configs tab (6 configs including guardrails & budget)
  - Audit Log tab with JSON viewer
  - Observability Dashboard with live stats
  - Replay Viewer for step-by-step execution trace
"""

import json
import gradio as gr

from harness_core import (
    NVIDIA_API_KEY,
    EXAMPLE_TASKS,
    COMPARISON_CONFIGS,
    Harness,
    HarnessConfig,
    TokenBudgetExceeded,
    register_tools,
    load_yaml_configs,
)


# ---------------------------------------------------------------------------
# GUI task descriptions (short form for example buttons)
# ---------------------------------------------------------------------------

GUI_EXAMPLE_TASKS = [t["task"] for t in EXAMPLE_TASKS]


# ---------------------------------------------------------------------------
# Run handlers
# ---------------------------------------------------------------------------

def run_harness(task, tools_on, memory_on, planner_on, verifier_on, retry_on,
                guardrails_on, persistent_mem, token_budget, human_approval, sub_agents):
    if not NVIDIA_API_KEY:
        err = "ERROR: NVIDIA_API_KEY environment variable not set.\nRun: export NVIDIA_API_KEY='your-key'\nOr create a .env file."
        return err, "", "", "", "", ""
    if not task.strip():
        return "ERROR: Please enter a task.", "", "", "", "", ""

    config = HarnessConfig(
        tools_enabled=tools_on,
        memory_enabled=memory_on,
        planner_enabled=planner_on,
        verifier_enabled=verifier_on,
        retry_enabled=retry_on,
        guardrails_enabled=guardrails_on,
        persistent_memory=persistent_mem,
        token_budget=int(token_budget) if token_budget else 0,
        human_approval=human_approval,
        sub_agents_enabled=sub_agents,
    )

    harness = Harness(config)
    register_tools(harness)

    try:
        return harness.run(task)
    except TokenBudgetExceeded as exc:
        log = "\n".join(harness.log_lines)
        return log + f"\n\nBUDGET EXCEEDED: {exc}", "", harness.memory.summary(), "", harness.tokens.summary(), ""
    except Exception as exc:
        return f"ERROR: {exc}", "", "", "", "", ""


def run_comparison(task):
    if not NVIDIA_API_KEY:
        return "ERROR: NVIDIA_API_KEY not set.", ""
    if not task.strip():
        return "ERROR: Please enter a task.", ""

    all_logs = []
    rows = []

    for name, config in COMPARISON_CONFIGS:
        all_logs.append(f"{'=' * 50}")
        all_logs.append(f"  CONFIG: {name}")
        all_logs.append(f"  {config.status_line()}")
        all_logs.append(f"{'=' * 50}")

        harness = Harness(config)
        register_tools(harness)

        try:
            log, answer, mem, stats, tokens, audit = harness.run(task)
            all_logs.append(log)
            all_logs.append(f"\nFINAL ANSWER: {answer[:150]}")
            all_logs.append("")

            rows.append({
                "Config": name,
                "Tool Calls": harness.stats["tool_calls"],
                "LLM Calls": harness.stats["llm_calls"],
                "Memory": harness.memory.count,
                "Verify": f"{harness.stats['verify_passes']}/{harness.stats['verify_total']}",
                "Retries": harness.stats["retries"],
                "Tokens": harness.tokens.total,
                "Guardrails": harness.stats.get("guardrail_checks", 0),
                "Answer": answer[:100],
            })
        except Exception as exc:
            all_logs.append(f"ERROR: {exc}\n")
            rows.append({"Config": name, "Tool Calls": "-", "LLM Calls": "-", "Memory": "-",
                         "Verify": "-", "Retries": "-", "Tokens": "-", "Guardrails": "-",
                         "Answer": f"Error: {exc}"})

    # Build summary table
    summary_lines = [f"{'Config':<18} {'Tools':>6} {'LLM':>5} {'Mem':>5} {'Verify':>8} {'Retry':>6} {'Guards':>7} {'Tokens':>8}"]
    summary_lines.append(f"{'-'*18} {'-'*6} {'-'*5} {'-'*5} {'-'*8} {'-'*6} {'-'*7} {'-'*8}")
    for r in rows:
        summary_lines.append(
            f"{r['Config']:<18} {r['Tool Calls']:>6} {r['LLM Calls']:>5} {r['Memory']:>5} "
            f"{r['Verify']:>8} {r['Retries']:>6} {r['Guardrails']:>7} {r['Tokens']:>8}"
        )
    summary_lines.append("")
    for r in rows:
        summary_lines.append(f"[{r['Config']}] {r['Answer']}")

    return "\n".join(all_logs), "\n".join(summary_lines)


def run_audit(task):
    if not NVIDIA_API_KEY:
        return "ERROR: NVIDIA_API_KEY not set."
    if not task.strip():
        return "ERROR: Enter a task."
    harness = Harness()
    register_tools(harness)
    try:
        _, _, _, _, _, audit_json = harness.run(task)
        return audit_json
    except Exception as exc:
        return f"ERROR: {exc}"


def run_observability(task):
    """Run a task and return observability dashboard data."""
    if not NVIDIA_API_KEY:
        return "ERROR: NVIDIA_API_KEY not set.", "", "", ""
    if not task.strip():
        return "ERROR: Enter a task.", "", "", ""

    harness = Harness(HarnessConfig(guardrails_enabled=True))
    register_tools(harness)

    try:
        log, answer, mem, stats, tokens, audit_json = harness.run(task)
    except Exception as exc:
        return f"ERROR: {exc}", "", "", ""

    # Build observability summary
    obs_lines = [
        "EXECUTION OVERVIEW",
        "=" * 40,
        f"Task: {task[:80]}",
        "",
        stats,
        "",
        "TOKEN BREAKDOWN",
        "=" * 40,
        tokens,
        "",
    ]

    if harness.tokens.budget > 0:
        obs_lines.append(f"Budget Usage: {harness.tokens.budget_pct:.1f}% ({harness.tokens.total}/{harness.tokens.budget})")
        obs_lines.append(f"Remaining: {harness.tokens.budget_remaining} tokens")

    # Guardrail summary
    if harness.stats.get("guardrail_checks", 0) > 0:
        obs_lines.append("")
        obs_lines.append("GUARDRAIL RESULTS")
        obs_lines.append("=" * 40)
        obs_lines.append(f"Checks run: {harness.stats['guardrail_checks']}")
        obs_lines.append(f"Blocks: {harness.stats['guardrail_blocks']}")

    obs_summary = "\n".join(obs_lines)

    # Token per-call breakdown
    token_detail = tokens

    # Timeline from audit
    try:
        entries = json.loads(audit_json)
        timeline_lines = []
        for entry in entries:
            ts = entry.get("timestamp", "")
            if "T" in ts:
                ts = ts.split("T")[1][:8]
            event = entry.get("event", "")
            detail = ""
            if event == "llm_call":
                detail = f"  {entry.get('call_type', '')} ({entry.get('input_tokens', 0)}+{entry.get('output_tokens', 0)} tokens)"
            elif event == "tool_call":
                detail = f"  {entry.get('tool', '')}({json.dumps(entry.get('args', {}))[:40]})"
            elif event == "verification":
                symbol = "PASS" if entry.get("passed") else "FAIL"
                detail = f"  {symbol}: {entry.get('reason', '')}"
            elif event == "guardrail":
                symbol = "PASS" if entry.get("passed") else "BLOCK"
                detail = f"  [{entry.get('check', '')}] {symbol}"
            elif event == "memory_store":
                detail = f"  {entry.get('key', '')} = {str(entry.get('value', ''))[:40]}"
            timeline_lines.append(f"[{ts}] {event}{detail}")
        timeline = "\n".join(timeline_lines)
    except Exception:
        timeline = audit_json

    return obs_summary, token_detail, timeline, answer


def run_replay(task):
    """Run a task and return replay trace."""
    if not NVIDIA_API_KEY:
        return "ERROR: NVIDIA_API_KEY not set."
    if not task.strip():
        return "ERROR: Enter a task."

    harness = Harness()
    register_tools(harness)

    try:
        harness.run(task)
    except Exception as exc:
        return f"ERROR: {exc}"

    replay_events = harness.audit.get_replay_events()
    lines = []
    for i, evt in enumerate(replay_events, 1):
        ts = evt["time"]
        if "T" in ts:
            ts = ts.split("T")[1][:8]
        icon = {
            "start": ">>", "plan": "[]", "llm": "**", "tool": "->",
            "verify": "??", "guardrail": "!!", "memory": "<<", "retry": "~~",
            "approval": "^^", "budget": "$$", "subagent": "@@", "complete": "##",
        }.get(evt["type"], "--")
        lines.append(f"  {i:>3}. [{ts}] {icon} {evt['step']:<12} {evt['detail']}")

    header = f"REPLAY TRACE ({len(replay_events)} events)\n{'=' * 60}\n"
    return header + "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

CSS = """
.log-box textarea { font-family: 'SF Mono', 'Fira Code', monospace !important; font-size: 13px !important; }
.answer-box textarea { font-size: 15px !important; font-weight: 500 !important; }
.stats-box textarea { font-family: 'SF Mono', 'Fira Code', monospace !important; font-size: 13px !important; }
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(title="Agent Harness Engineering", theme=THEME, css=CSS) as demo:

    gr.Markdown("# Agent Harness Engineering")
    gr.Markdown(
        "Visual interface for an agent harness using **NVIDIA Nemotron Super 49B**. "
        "Nine tools, six harness components, guardrails, budget enforcement, sub-agents, "
        "token tracking, and full audit logging."
    )

    with gr.Tabs():
        # ============================================================
        # TAB 1: Single Run
        # ============================================================
        with gr.Tab("Run Task"):
            with gr.Row():
                # --- Left Column: Controls ---
                with gr.Column(scale=1):
                    gr.Markdown("### Core Components")
                    tools_on = gr.Checkbox(label="Tool Integration", value=True, info="9 tools (calculator, time, reverse, search, file, word count, json extract, summarize, code execute)")
                    memory_on = gr.Checkbox(label="Memory & State", value=True, info="Store intermediate results in working memory")
                    planner_on = gr.Checkbox(label="Planning", value=True, info="Decompose complex tasks into steps via LLM")
                    verifier_on = gr.Checkbox(label="Verification", value=True, info="Validate outputs with rule-based checks")
                    retry_on = gr.Checkbox(label="Retry / Self-Correction", value=True, info="Retry failed steps with feedback injection")

                    gr.Markdown("### Advanced Features")
                    guardrails_on = gr.Checkbox(label="Guardrails", value=False, info="PII detection, toxicity filter, code injection checks")
                    persistent_mem = gr.Checkbox(label="Persistent Memory (SQLite)", value=False, info="Store memory across sessions")
                    human_approval = gr.Checkbox(label="Human-in-the-Loop", value=False, info="Require approval for high-risk tools")
                    sub_agents = gr.Checkbox(label="Sub-Agents", value=False, info="Enable researcher, analyst, writer sub-agents")
                    token_budget = gr.Number(label="Token Budget (0 = unlimited)", value=0, precision=0, info="Max tokens before stopping")

                    gr.Markdown("### Task")
                    task_input = gr.Textbox(
                        label="Enter your task",
                        placeholder="What is 47 * 83? Then find the square root...",
                        lines=3,
                    )
                    run_btn = gr.Button("Run Harness", variant="primary", size="lg")

                    gr.Markdown("### Examples")
                    for ex in GUI_EXAMPLE_TASKS[:6]:
                        gr.Button(ex[:65] + "..." if len(ex) > 65 else ex, size="sm").click(
                            fn=lambda e=ex: e, outputs=task_input
                        )

                # --- Centre Column: Execution Log ---
                with gr.Column(scale=2):
                    gr.Markdown("### Execution Log")
                    log_output = gr.Textbox(label="", lines=22, max_lines=50, interactive=False, elem_classes=["log-box"])

                    gr.Markdown("### Final Answer")
                    answer_output = gr.Textbox(label="", lines=3, interactive=False, elem_classes=["answer-box"])

                # --- Right Column: State ---
                with gr.Column(scale=1):
                    gr.Markdown("### Working Memory")
                    memory_output = gr.Textbox(label="", lines=6, interactive=False, elem_classes=["stats-box"])

                    gr.Markdown("### Harness Stats")
                    stats_output = gr.Textbox(label="", lines=10, interactive=False, elem_classes=["stats-box"])

                    gr.Markdown("### Token Usage")
                    token_output = gr.Textbox(label="", lines=8, interactive=False, elem_classes=["stats-box"])

                    gr.Markdown("### Registered Tools")
                    gr.Markdown(
                        "- **calculator**(expression) — math evaluation\n"
                        "- **get_current_time**() — current date/time\n"
                        "- **reverse_string**(text) — reverse a string\n"
                        "- **web_search**(query) — search the web\n"
                        "- **read_file**(filename) — read file contents\n"
                        "- **word_count**(text) — count words/chars/lines\n"
                        "- **json_extract**(json, key) — extract from JSON\n"
                        "- **text_summarize**(text) — AI summarization\n"
                        "- **code_execute**(code) — sandboxed Python"
                    )

            run_btn.click(
                fn=run_harness,
                inputs=[task_input, tools_on, memory_on, planner_on, verifier_on, retry_on,
                        guardrails_on, persistent_mem, token_budget, human_approval, sub_agents],
                outputs=[log_output, answer_output, memory_output, stats_output, token_output, gr.Textbox(visible=False)],
            )

        # ============================================================
        # TAB 2: Comparison Mode
        # ============================================================
        with gr.Tab("Compare Configs"):
            gr.Markdown(
                "### Comparison Mode\n"
                "Run the same task with 6 different harness configurations and compare results.\n"
                "Configs: **All ON**, **No Planner**, **No Verifier**, **No Memory**, **With Guardrails**, **Budget Limited**."
            )

            comp_task = gr.Textbox(
                label="Task to compare",
                placeholder="Enter a task to run across all configs...",
                value=GUI_EXAMPLE_TASKS[0],
                lines=2,
            )
            comp_btn = gr.Button("Run Comparison", variant="primary", size="lg")

            gr.Markdown("### Summary Table")
            comp_summary = gr.Textbox(label="", lines=12, interactive=False, elem_classes=["stats-box"])

            gr.Markdown("### Full Execution Logs")
            comp_logs = gr.Textbox(label="", lines=30, max_lines=80, interactive=False, elem_classes=["log-box"])

            comp_btn.click(
                fn=run_comparison,
                inputs=[comp_task],
                outputs=[comp_logs, comp_summary],
            )

        # ============================================================
        # TAB 3: Audit Log
        # ============================================================
        with gr.Tab("Audit Log"):
            gr.Markdown(
                "### Structured Audit Log\n"
                "Run a task and view the structured JSON audit trail. "
                "Every LLM call, tool execution, verification, guardrail check, memory store, "
                "and retry is recorded with timestamps."
            )

            audit_task = gr.Textbox(label="Task", value=GUI_EXAMPLE_TASKS[0], lines=2)
            audit_btn = gr.Button("Run & Generate Audit Log", variant="primary")
            audit_display = gr.Textbox(label="Audit Log (JSON)", lines=30, max_lines=60, interactive=False, elem_classes=["log-box"])

            audit_btn.click(fn=run_audit, inputs=[audit_task], outputs=[audit_display])

        # ============================================================
        # TAB 4: Observability Dashboard
        # ============================================================
        with gr.Tab("Observability"):
            gr.Markdown(
                "### Observability Dashboard\n"
                "Run a task with guardrails enabled and view execution metrics, "
                "token breakdown, and event timeline."
            )

            obs_task = gr.Textbox(label="Task", value=GUI_EXAMPLE_TASKS[0], lines=2)
            obs_btn = gr.Button("Run & Observe", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Execution Summary")
                    obs_summary = gr.Textbox(label="", lines=18, interactive=False, elem_classes=["stats-box"])
                with gr.Column(scale=1):
                    gr.Markdown("### Token Breakdown")
                    obs_tokens = gr.Textbox(label="", lines=10, interactive=False, elem_classes=["stats-box"])

                    gr.Markdown("### Final Answer")
                    obs_answer = gr.Textbox(label="", lines=4, interactive=False, elem_classes=["answer-box"])

            gr.Markdown("### Event Timeline")
            obs_timeline = gr.Textbox(label="", lines=15, max_lines=40, interactive=False, elem_classes=["log-box"])

            obs_btn.click(
                fn=run_observability,
                inputs=[obs_task],
                outputs=[obs_summary, obs_tokens, obs_timeline, obs_answer],
            )

        # ============================================================
        # TAB 5: Replay Viewer
        # ============================================================
        with gr.Tab("Replay"):
            gr.Markdown(
                "### Replay / Time-Travel Debugging\n"
                "Run a task and view a step-by-step replay trace of every event "
                "that occurred during execution — LLM calls, tool invocations, "
                "verification checks, memory stores, retries, and more."
            )

            replay_task = gr.Textbox(label="Task", value=GUI_EXAMPLE_TASKS[0], lines=2)
            replay_btn = gr.Button("Run & Replay", variant="primary")
            replay_output = gr.Textbox(label="Replay Trace", lines=30, max_lines=60, interactive=False, elem_classes=["log-box"])

            replay_btn.click(fn=run_replay, inputs=[replay_task], outputs=[replay_output])

    gr.Markdown(
        "---\n"
        "*Agent Harness Engineering — Cobus Greyling*"
    )

if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css=CSS,
    )
