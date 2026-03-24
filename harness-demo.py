"""
Agent Harness CLI Demo — Nemotron Super 49B
============================================
Full-featured CLI agent with 9 tools, retry loop, token tracking,
budget enforcement, guardrails, sub-agents, and audit log.

Uses shared components from harness_core.py.
"""

import sys

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
# Comparison mode
# ---------------------------------------------------------------------------

def run_comparison(task: str, configs=None):
    """Run the same task with different harness configs and compare."""
    configs = configs or COMPARISON_CONFIGS
    results = []

    for name, config in configs:
        print(f"\n{'#' * 60}")
        print(f"  COMPARISON RUN: {name}")
        print(f"{'#' * 60}")

        harness = Harness(config)
        register_tools(harness)

        try:
            log, answer, mem, stats, tokens, audit = harness.run(task)
            # Print log to stdout
            for line in log.split("\n"):
                print(line)
            print(f"\n{'=' * 60}")
            print(f"  FINAL ANSWER")
            print(f"{'=' * 60}")
            print(f"  {answer}")
            print()
        except TokenBudgetExceeded as e:
            answer = f"Budget exceeded: {e}"
            print(f"  {answer}")

        results.append({
            "config": name,
            "answer": (answer or "")[:150],
            "tool_calls": harness.stats["tool_calls"],
            "llm_calls": harness.stats["llm_calls"],
            "memory_entries": harness.memory.count,
            "verify_pass_rate": f"{harness.stats['verify_passes']}/{harness.stats['verify_total']}",
            "retries": harness.stats["retries"],
            "total_tokens": harness.tokens.total,
        })

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Config':<18} {'Tools':>6} {'LLM':>5} {'Mem':>5} {'Verify':>8} {'Retry':>6} {'Tokens':>8}")
    print(f"  {'-' * 18} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 8} {'-' * 6} {'-' * 8}")
    for r in results:
        print(f"  {r['config']:<18} {r['tool_calls']:>6} {r['llm_calls']:>5} {r['memory_entries']:>5} {r['verify_pass_rate']:>8} {r['retries']:>6} {r['total_tokens']:>8}")
    print()
    for r in results:
        print(f"  [{r['config']}] {r['answer']}")
    print()


# ---------------------------------------------------------------------------
# Single run (CLI output)
# ---------------------------------------------------------------------------

def run_single(task: str, config: HarnessConfig | None = None):
    """Run a single task with CLI-formatted output."""
    harness = Harness(config or HarnessConfig())
    register_tools(harness)

    print(f"\n{'=' * 60}")
    print(f"  HARNESS DEMO — Nemotron Super 49B")
    print(f"{'=' * 60}\n")

    try:
        log, answer, mem, stats, tokens, audit = harness.run(task)
    except TokenBudgetExceeded as e:
        print(f"  Token budget exceeded: {e}")
        return ""

    # Print execution log
    for line in log.split("\n"):
        print(line)

    # Final answer
    print(f"{'=' * 60}")
    print(f"  FINAL ANSWER")
    print(f"{'=' * 60}")
    print(f"  {answer}")
    print()

    # Stats
    print(f"{'=' * 60}")
    print(f"  HARNESS STATS")
    print(f"{'=' * 60}")
    for line in stats.split("\n"):
        print(f"  {line}")
    print()

    # Token usage
    print(f"{'=' * 60}")
    print(f"  TOKEN USAGE")
    print(f"{'=' * 60}")
    for line in tokens.split("\n"):
        print(f"  {line}")
    print()

    # Export audit log
    audit_file = harness.audit.export("harness-audit.json")
    print(f"  Audit log exported: {audit_file}")

    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not NVIDIA_API_KEY:
        print("Error: set NVIDIA_API_KEY environment variable")
        print("  export NVIDIA_API_KEY='your-key'")
        print("  Or create a .env file with NVIDIA_API_KEY=your-key")
        return

    # Parse args
    mode = "demo"
    task_index = 0
    config_file = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--compare":
            mode = "compare"
            if i + 1 < len(args) and args[i + 1].isdigit():
                task_index = int(args[i + 1])
                i += 1
        elif args[i] == "--all":
            mode = "all"
        elif args[i] == "--task":
            mode = "single"
            if i + 1 < len(args):
                task_index = int(args[i + 1])
                i += 1
        elif args[i] == "--config":
            if i + 1 < len(args):
                config_file = args[i + 1]
                i += 1
        elif args[i] == "--list-configs":
            configs = load_yaml_configs()
            if configs:
                print("Available YAML configs:")
                for name, cfg in configs.items():
                    print(f"  {name}: {cfg.status_line()}")
            else:
                print("No YAML configs found in configs/ directory.")
            return
        elif args[i] == "--help":
            print("Usage:")
            print(f"  python3 harness-demo.py                  Run default task")
            print(f"  python3 harness-demo.py --task N          Run task N (0-{len(EXAMPLE_TASKS)-1})")
            print(f"  python3 harness-demo.py --all             Run all {len(EXAMPLE_TASKS)} tasks")
            print(f"  python3 harness-demo.py --compare         Compare harness configs on default task")
            print(f"  python3 harness-demo.py --compare N       Compare configs on task N")
            print(f"  python3 harness-demo.py --config FILE     Use a YAML config file")
            print(f"  python3 harness-demo.py --list-configs    List available YAML configs")
            print()
            print("Tasks:")
            for idx, t in enumerate(EXAMPLE_TASKS):
                print(f"  {idx}: {t['name']} — {t['task'][:60]}...")
            return
        i += 1

    # Load YAML config if specified
    config = None
    if config_file:
        config = HarnessConfig.from_yaml(config_file)
        print(f"Loaded config: {config.name} ({config_file})")

    if mode == "compare":
        task = EXAMPLE_TASKS[task_index]["task"]
        print(f"Comparison mode: {EXAMPLE_TASKS[task_index]['name']}")
        run_comparison(task)
        return

    if mode == "all":
        for idx, t in enumerate(EXAMPLE_TASKS):
            print(f"\n{'*' * 60}")
            print(f"  TASK {idx + 1}/{len(EXAMPLE_TASKS)}: {t['name']}")
            print(f"{'*' * 60}")
            run_single(t["task"], config)
        return

    # Single task
    task = EXAMPLE_TASKS[task_index]["task"]
    run_single(task, config)


if __name__ == "__main__":
    main()
