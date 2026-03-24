"""
Minimal Agent Harness — Nemotron 3 Super
=========================================
Six harness components + retry loop + token tracking + audit log
orchestrated in a single-file CLI demo.

Components:
  1. ToolRegistry — register and execute tools
  2. MemoryManager — working memory + session state
  3. ContextEngine — assembles prompt dynamically
  4. Planner — decomposes a complex task into steps
  5. Verifier — checks output against rules (with retry on failure)
  6. HarnessConfig — enable/disable components independently

Enhancements over basic demo:
  - 6 tools (calculator, time, reverse, web_search, read_file, word_count)
  - 5 diverse example tasks
  - Retry/self-correction loop on verification failure (max 2 retries)
  - Token tracking per LLM call + total spend
  - Structured JSON audit log (harness-audit.json)
  - Comparison mode: run same task with different configs
"""

import os
import re
import json
import math
import time as time_mod
from datetime import datetime
from dataclasses import dataclass, field
from openai import OpenAI

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

CLIENT = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
    default_headers={"NVCF-POLL-SECONDS": "1800"},
)

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# 1. ToolRegistry — register and execute tools
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, callable] = {}
        self._schemas: list[dict] = []

    def register(self, name: str, func: callable, description: str, parameters: dict):
        self._tools[name] = func
        self._schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })

    def execute(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        return str(self._tools[name](**arguments))

    @property
    def schemas(self) -> list[dict]:
        return self._schemas

    @property
    def descriptions(self) -> str:
        lines = []
        for s in self._schemas:
            f = s["function"]
            params = ", ".join(f["parameters"].get("properties", {}).keys())
            lines.append(f"- {f['name']}({params}): {f['description']}")
        return "\n".join(lines)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# 2. MemoryManager — working memory + session state
# ---------------------------------------------------------------------------

class MemoryManager:
    def __init__(self):
        self._store: dict[str, str] = {}

    def store(self, key: str, value: str):
        self._store[key] = value

    def recall(self, keys: list[str] | None = None) -> dict[str, str]:
        if keys is None:
            return dict(self._store)
        return {k: self._store[k] for k in keys if k in self._store}

    def summary(self) -> str:
        if not self._store:
            return "No stored memory."
        return "\n".join(f"  {k} = {v}" for k, v in self._store.items())

    def clear(self):
        self._store.clear()

    @property
    def count(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# 3. ContextEngine — assembles prompt dynamically
# ---------------------------------------------------------------------------

class ContextEngine:
    SYSTEM_BASE = (
        "You are a precise assistant. When you need to compute something, "
        "call the appropriate tool. When you need to look something up, "
        "use the web_search tool. When you need to read a file, use read_file. "
        "Return only the requested information. Be concise."
    )

    def build(self, task: str, memory: MemoryManager, tools: ToolRegistry,
              step_instruction: str | None = None,
              retry_feedback: str | None = None) -> list[dict]:
        parts = [self.SYSTEM_BASE]
        parts.append(f"\nAvailable tools:\n{tools.descriptions}")
        mem = memory.summary()
        if mem != "No stored memory.":
            parts.append(f"\nWorking memory:\n{mem}")
        if retry_feedback:
            parts.append(f"\nPREVIOUS ATTEMPT FAILED: {retry_feedback}")
            parts.append("Please try again, correcting the issue.")
        system = "\n".join(parts)
        user_content = step_instruction or task
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]


# ---------------------------------------------------------------------------
# 4. Planner — decomposes a complex task into steps
# ---------------------------------------------------------------------------

class Planner:
    def decompose(self, task: str) -> list[str]:
        messages = [
            {"role": "system", "content": (
                "You are a task planner. Break the user's task into numbered steps. "
                "Return ONLY a JSON array of strings, one per step. No other text."
            )},
            {"role": "user", "content": task},
        ]
        response = CLIENT.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.2, max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return [line.strip().lstrip("0123456789.) ") for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 5. Verifier — checks output against rules
# ---------------------------------------------------------------------------

class Verifier:
    def check_numeric(self, result: str) -> tuple[bool, str]:
        try:
            float(result.replace(",", "").strip())
            return True, "Result is numeric"
        except ValueError:
            return False, "Result is not numeric"

    def check_contains(self, text: str, values: list[str]) -> tuple[bool, str]:
        missing = [v for v in values if str(v) not in text]
        if missing:
            return False, f"Missing values: {missing}"
        return True, "Contains all expected values"

    def check_single_sentence(self, text: str) -> tuple[bool, str]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) <= 2:
            return True, "Single sentence"
        return False, f"Contains {len(sentences)} sentences, expected 1-2"

    def check_not_empty(self, text: str) -> tuple[bool, str]:
        if text and text.strip() and text.strip().lower() not in ("none", "n/a", "error"):
            return True, "Non-empty response"
        return False, "Empty or invalid response"

    def check_json_format(self, text: str) -> tuple[bool, str]:
        try:
            json.loads(text)
            return True, "Valid JSON"
        except (json.JSONDecodeError, ValueError):
            return False, "Not valid JSON"


# ---------------------------------------------------------------------------
# 6. HarnessConfig — enable/disable components
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    tools_enabled: bool = True
    memory_enabled: bool = True
    planner_enabled: bool = True
    verifier_enabled: bool = True
    retry_enabled: bool = True

    def status_line(self) -> str:
        flags = {
            "Tools": self.tools_enabled,
            "Memory": self.memory_enabled,
            "Planner": self.planner_enabled,
            "Verifier": self.verifier_enabled,
            "Retry": self.retry_enabled,
        }
        return " | ".join(f"{k}: {'ON' if v else 'OFF'}" for k, v in flags.items())


# ---------------------------------------------------------------------------
# Token Tracker
# ---------------------------------------------------------------------------

@dataclass
class TokenTracker:
    calls: list = field(default_factory=list)

    def record(self, call_type: str, input_tokens: int, output_tokens: int):
        self.calls.append({
            "type": call_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        })

    @property
    def total_input(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def total_output(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)

    @property
    def total(self) -> int:
        return sum(c["total_tokens"] for c in self.calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def summary(self) -> str:
        lines = []
        for i, c in enumerate(self.calls, 1):
            lines.append(f"  Call {i} ({c['type']}): {c['input_tokens']} in + {c['output_tokens']} out = {c['total_tokens']}")
        lines.append(f"  TOTAL: {self.total_input} in + {self.total_output} out = {self.total} tokens across {self.call_count} calls")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class AuditLog:
    def __init__(self):
        self.entries: list[dict] = []

    def log(self, event_type: str, detail: dict):
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **detail,
        })

    def export(self, filepath: str = "harness-audit.json"):
        with open(filepath, "w") as f:
            json.dump(self.entries, f, indent=2)
        return filepath


# ---------------------------------------------------------------------------
# Harness — orchestrates the agent loop
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self, config: HarnessConfig | None = None):
        self.config = config or HarnessConfig()
        self.tools = ToolRegistry()
        self.memory = MemoryManager()
        self.context = ContextEngine()
        self.planner = Planner()
        self.verifier = Verifier()
        self.tokens = TokenTracker()
        self.audit = AuditLog()
        self.stats = {
            "tool_calls": 0, "llm_calls": 0,
            "verify_passes": 0, "verify_total": 0,
            "retries": 0,
        }

    def _log(self, tag: str, msg: str):
        print(f"[{tag:<10}] {msg}")

    def _llm_call(self, messages: list[dict], use_tools: bool = False,
                   call_type: str = "step") -> object:
        kwargs = dict(model=MODEL, messages=messages, temperature=0.3, max_tokens=1024)
        if use_tools and self.config.tools_enabled and self.tools.schemas:
            kwargs["tools"] = self.tools.schemas
            kwargs["tool_choice"] = "auto"
        self.stats["llm_calls"] += 1
        response = CLIENT.chat.completions.create(**kwargs)

        # Track tokens
        usage = response.usage
        if usage:
            self.tokens.record(call_type, usage.prompt_tokens, usage.completion_tokens)
            self.audit.log("llm_call", {
                "call_type": call_type,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            })

        return response

    def _execute_tool_calls(self, response) -> list[tuple[str, str, str]]:
        results = []
        msg = response.choices[0].message
        if not msg.tool_calls:
            return results
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = self.tools.execute(name, args)
            args_str = ", ".join(f'"{v}"' for v in args.values())
            self._log("TOOL", f'{name}({args_str}) = {result}')
            self.stats["tool_calls"] += 1
            self.audit.log("tool_call", {"tool": name, "args": args, "result": result})
            results.append((name, args_str, result))
        return results

    def _verify(self, check_fn, *args) -> tuple[bool, str]:
        self.stats["verify_total"] += 1
        passed, reason = check_fn(*args)
        symbol = "\u2713" if passed else "\u2717"
        self._log("VERIFY", f"{symbol} {reason}")
        if passed:
            self.stats["verify_passes"] += 1
        self.audit.log("verification", {"passed": passed, "reason": reason})
        return passed, reason

    def _run_step_with_retry(self, task: str, step: str, step_num: int,
                              total_steps: int) -> str:
        retry_feedback = None

        for attempt in range(1 + (MAX_RETRIES if self.config.retry_enabled else 0)):
            if attempt > 0:
                self.stats["retries"] += 1
                self._log("RETRY", f"Attempt {attempt + 1}/{MAX_RETRIES + 1} for step {step_num}")
                self.audit.log("retry", {"step": step_num, "attempt": attempt + 1, "feedback": retry_feedback})

            messages = self.context.build(
                task, self.memory, self.tools,
                step_instruction=step,
                retry_feedback=retry_feedback,
            )

            response = self._llm_call(messages, use_tools=True, call_type=f"step_{step_num}")
            msg = response.choices[0].message

            tool_results = self._execute_tool_calls(response)

            if tool_results:
                _, _, result_value = tool_results[-1]
                step_result = result_value

                if self.config.verifier_enabled:
                    passed, reason = self._verify(self.verifier.check_not_empty, step_result)
                    if not passed:
                        retry_feedback = reason
                        continue

                if self.config.memory_enabled:
                    key = f"step_{step_num}_result"
                    self.memory.store(key, step_result)
                    self._log("MEMORY", f"Stored: {key} = {step_result}")
                    self.audit.log("memory_store", {"key": key, "value": step_result})

                return step_result
            else:
                step_result = msg.content.strip() if msg.content else ""
                self._log("LLM", f'\u2192 "{step_result[:120]}{"..." if len(step_result) > 120 else ""}"')

                if self.config.verifier_enabled:
                    passed, reason = self._verify(self.verifier.check_not_empty, step_result)
                    if not passed:
                        retry_feedback = reason
                        continue

                if self.config.memory_enabled:
                    key = f"step_{step_num}_result"
                    self.memory.store(key, step_result)
                    self.audit.log("memory_store", {"key": key, "value": step_result[:200]})

                return step_result

        self._log("RETRY", f"Max retries exceeded for step {step_num}")
        return step_result if 'step_result' in dir() else ""

    def run(self, task: str) -> str:
        t0 = time_mod.time()

        print(f"\n{'=' * 60}")
        print(f"  HARNESS DEMO — Nemotron Super 49B")
        print(f"{'=' * 60}\n")

        self._log("CONFIG", self.config.status_line())
        self._log("TASK", f'"{task}"')
        self._log("TOOLS", f'{len(self.tools.names)} registered: {", ".join(self.tools.names)}')
        self.audit.log("task_start", {"task": task, "config": self.config.status_line()})
        print()

        # --- Planning ---
        if self.config.planner_enabled:
            self._log("PLANNER", "Decomposing task...")
            steps = self.planner.decompose(task)
            self._log("PLANNER", f"Decomposed into {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"{'':>13} {i}. {step}")
            self.audit.log("plan", {"steps": steps})
        else:
            steps = [task]
            self._log("PLANNER", "Disabled — running as single step")
        print()

        # --- Execute each step ---
        final_answer = ""
        for i, step in enumerate(steps, 1):
            self._log(f"STEP {i}/{len(steps)}", step)
            result = self._run_step_with_retry(task, step, i, len(steps))
            if result:
                final_answer = result
            print()

        # --- Final answer ---
        if not final_answer:
            mem = self.memory.recall()
            parts = [f"{k} = {v}" for k, v in mem.items()]
            final_answer = "; ".join(parts) if parts else "No result produced."

        elapsed = time_mod.time() - t0

        print(f"{'=' * 60}")
        print(f"  FINAL ANSWER")
        print(f"{'=' * 60}")
        print(f"  {final_answer}")
        print()

        # --- Stats ---
        print(f"{'=' * 60}")
        print(f"  HARNESS STATS")
        print(f"{'=' * 60}")
        print(f"  Steps completed:      {len(steps)}")
        print(f"  Tool calls:           {self.stats['tool_calls']}")
        print(f"  Memory entries:       {self.memory.count}")
        print(f"  Verification:         {self.stats['verify_passes']}/{self.stats['verify_total']} passed")
        print(f"  Retries:              {self.stats['retries']}")
        print(f"  LLM calls:            {self.stats['llm_calls']}")
        print(f"  Elapsed:              {elapsed:.1f}s")
        print()

        # --- Token breakdown ---
        print(f"{'=' * 60}")
        print(f"  TOKEN USAGE")
        print(f"{'=' * 60}")
        print(self.tokens.summary())
        print()

        # --- Audit log ---
        self.audit.log("task_complete", {
            "final_answer": final_answer[:200],
            "elapsed_seconds": round(elapsed, 2),
            "total_tokens": self.tokens.total,
            "stats": dict(self.stats),
        })

        return final_answer


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() eE")
    clean = expression.replace("**", "POWER").replace("sqrt", "")
    if not all(c in allowed or c in "POWER" for c in clean):
        return "Error: invalid expression"
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
    except Exception as exc:
        return f"Error: {exc}"
    if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
        return str(int(result))
    return str(result)


def get_current_time() -> str:
    """Return current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


def web_search(query: str) -> str:
    """Simulate a web search and return results."""
    # Simulated search results for common queries
    results = {
        "population of france": "France has a population of approximately 68.4 million people (2025 estimate).",
        "tallest building": "The Burj Khalifa in Dubai is the tallest building at 828 meters (2,717 feet).",
        "speed of light": "The speed of light in vacuum is 299,792,458 meters per second.",
        "largest ocean": "The Pacific Ocean is the largest ocean, covering approximately 165.25 million square kilometers.",
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
    }
    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower or any(w in query_lower for w in key.split()):
            return value
    return f"Search result for '{query}': No specific result found. The query returned general information about the topic."


def read_file(filename: str) -> str:
    """Simulate reading a file and return its contents."""
    files = {
        "config.yaml": "app_name: MyAgent\nversion: 2.1.0\nmax_retries: 3\ntimeout: 30\nlog_level: INFO",
        "metrics.csv": "date,requests,errors,latency_ms\n2026-03-01,12450,23,145\n2026-03-02,13200,18,132\n2026-03-03,11800,31,167\n2026-03-04,14500,12,128\n2026-03-05,15200,8,119",
        "notes.txt": "Meeting notes from March 20:\n- Launch date confirmed for April 15\n- Budget approved: $250,000\n- Team size: 8 engineers\n- Key risk: third-party API dependency",
    }
    if filename in files:
        return files[filename]
    return f"Error: file '{filename}' not found. Available files: {', '.join(files.keys())}"


def word_count(text: str) -> str:
    """Count words in the given text."""
    words = text.split()
    chars = len(text)
    lines = text.count("\n") + 1
    return f"Words: {len(words)}, Characters: {chars}, Lines: {lines}"


# ---------------------------------------------------------------------------
# Register all tools on a harness
# ---------------------------------------------------------------------------

def register_tools(harness: Harness):
    harness.tools.register(
        "calculator", calculator,
        "Evaluate a mathematical expression. Supports +, -, *, /, **, parentheses.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}}, "required": ["expression"]},
    )
    harness.tools.register(
        "get_current_time", get_current_time,
        "Get the current date and time.",
        {"type": "object", "properties": {}},
    )
    harness.tools.register(
        "reverse_string", reverse_string,
        "Reverse a given string.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "Text to reverse"}}, "required": ["text"]},
    )
    harness.tools.register(
        "web_search", web_search,
        "Search the web for information on a topic. Returns a text summary.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]},
    )
    harness.tools.register(
        "read_file", read_file,
        "Read the contents of a file. Available files: config.yaml, metrics.csv, notes.txt",
        {"type": "object", "properties": {"filename": {"type": "string", "description": "Name of the file to read"}}, "required": ["filename"]},
    )
    harness.tools.register(
        "word_count", word_count,
        "Count words, characters, and lines in a text string.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "Text to count"}}, "required": ["text"]},
    )


# ---------------------------------------------------------------------------
# Example tasks
# ---------------------------------------------------------------------------

EXAMPLE_TASKS = [
    {
        "name": "Multi-step math",
        "task": "What is 47 * 83? Then find the square root of that result. Finally summarise both results in a single sentence.",
    },
    {
        "name": "Web search + calculation",
        "task": "Search for the speed of light. Then calculate how many kilometers light travels in one minute. Summarise the answer.",
    },
    {
        "name": "File analysis",
        "task": "Read the file metrics.csv. Find the day with the highest number of requests and the day with the lowest latency. Summarise both findings.",
    },
    {
        "name": "Multi-tool chain",
        "task": "What is the current date and time? Reverse the string 'Harness Engineering'. Count the words in the reversed result. Summarise all three results.",
    },
    {
        "name": "Research + memory recall",
        "task": "Search for the population of France. Search for the largest ocean. Calculate the population divided by 1000. Summarise all findings in a single sentence using the stored results.",
    },
]


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def run_comparison(task: str):
    """Run the same task with different harness configs and compare."""
    configs = [
        ("All ON", HarnessConfig(tools_enabled=True, memory_enabled=True, planner_enabled=True, verifier_enabled=True, retry_enabled=True)),
        ("No Planner", HarnessConfig(tools_enabled=True, memory_enabled=True, planner_enabled=False, verifier_enabled=True, retry_enabled=True)),
        ("No Verifier", HarnessConfig(tools_enabled=True, memory_enabled=True, planner_enabled=True, verifier_enabled=False, retry_enabled=False)),
        ("No Memory", HarnessConfig(tools_enabled=True, memory_enabled=False, planner_enabled=True, verifier_enabled=True, retry_enabled=True)),
    ]

    results = []
    for name, config in configs:
        print(f"\n{'#' * 60}")
        print(f"  COMPARISON RUN: {name}")
        print(f"{'#' * 60}")

        harness = Harness(config)
        register_tools(harness)
        answer = harness.run(task)
        results.append({
            "config": name,
            "answer": answer[:150],
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
    print(f"  {'Config':<16} {'Tools':>6} {'LLM':>5} {'Mem':>5} {'Verify':>8} {'Retry':>6} {'Tokens':>8}")
    print(f"  {'-' * 16} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 8} {'-' * 6} {'-' * 8}")
    for r in results:
        print(f"  {r['config']:<16} {r['tool_calls']:>6} {r['llm_calls']:>5} {r['memory_entries']:>5} {r['verify_pass_rate']:>8} {r['retries']:>6} {r['total_tokens']:>8}")
    print()

    # Export all audit logs
    for i, (name, _) in enumerate(configs):
        print(f"  [{name}] Answer: {results[i]['answer']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys

    if not NVIDIA_API_KEY:
        print("Error: set NVIDIA_API_KEY environment variable")
        return

    # Parse args
    mode = "demo"
    task_index = 0
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            mode = "compare"
            if len(sys.argv) > 2:
                task_index = int(sys.argv[2])
        elif sys.argv[1] == "--all":
            mode = "all"
        elif sys.argv[1] == "--task":
            mode = "single"
            task_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python3 harness-demo.py              Run default task")
            print("  python3 harness-demo.py --task N      Run task N (0-4)")
            print("  python3 harness-demo.py --all         Run all 5 tasks")
            print("  python3 harness-demo.py --compare     Compare harness configs on default task")
            print("  python3 harness-demo.py --compare N   Compare harness configs on task N")
            print()
            print("Tasks:")
            for i, t in enumerate(EXAMPLE_TASKS):
                print(f"  {i}: {t['name']} — {t['task'][:60]}...")
            return

    if mode == "compare":
        task = EXAMPLE_TASKS[task_index]["task"]
        print(f"Comparison mode: {EXAMPLE_TASKS[task_index]['name']}")
        run_comparison(task)
        return

    if mode == "all":
        for i, t in enumerate(EXAMPLE_TASKS):
            print(f"\n{'*' * 60}")
            print(f"  TASK {i + 1}/5: {t['name']}")
            print(f"{'*' * 60}")
            harness = Harness()
            register_tools(harness)
            harness.run(t["task"])
            audit_file = harness.audit.export(f"harness-audit-task{i+1}.json")
            print(f"  Audit log: {audit_file}")
        return

    # Single task
    task = EXAMPLE_TASKS[task_index]["task"]
    harness = Harness()
    register_tools(harness)
    harness.run(task)
    audit_file = harness.audit.export("harness-audit.json")
    print(f"  Audit log exported: {audit_file}")


if __name__ == "__main__":
    main()
