"""
Agent Harness GUI — Gradio frontend for the Agent Harness
Wraps all six harness components with a visual interface.
Features: 6 tools, retry/self-correction, token tracking, audit log export, comparison mode.
"""

import os
import re
import json
import math
import time as time_mod
from datetime import datetime
from dataclasses import dataclass, field
from openai import OpenAI
import gradio as gr

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

CLIENT = None

def get_client():
    global CLIENT
    if CLIENT is None:
        CLIENT = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
            default_headers={"NVCF-POLL-SECONDS": "1800"},
        )
    return CLIENT

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# 1. ToolRegistry
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
    def names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def descriptions(self) -> str:
        lines = []
        for s in self._schemas:
            f = s["function"]
            params = ", ".join(f["parameters"].get("properties", {}).keys())
            lines.append(f"- {f['name']}({params}): {f['description']}")
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# 2. MemoryManager
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
# 3. ContextEngine
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
# 4. Planner
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
        response = get_client().chat.completions.create(
            model=MODEL, messages=messages, temperature=0.2, max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return [line.strip().lstrip("0123456789.) ") for line in text.splitlines() if line.strip()]

# ---------------------------------------------------------------------------
# 5. Verifier
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

# ---------------------------------------------------------------------------
# 6. HarnessConfig
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
            lines.append(f"Call {i} ({c['type']}): {c['input_tokens']} in + {c['output_tokens']} out = {c['total_tokens']}")
        lines.append(f"TOTAL: {self.total_input} in + {self.total_output} out = {self.total} tokens ({self.call_count} calls)")
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

    def to_json(self) -> str:
        return json.dumps(self.entries, indent=2)

    def export(self, filepath: str = "harness-audit.json"):
        with open(filepath, "w") as f:
            json.dump(self.entries, f, indent=2)
        return filepath

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def reverse_string(text: str) -> str:
    return text[::-1]

def web_search(query: str) -> str:
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
    files = {
        "config.yaml": "app_name: MyAgent\nversion: 2.1.0\nmax_retries: 3\ntimeout: 30\nlog_level: INFO",
        "metrics.csv": "date,requests,errors,latency_ms\n2026-03-01,12450,23,145\n2026-03-02,13200,18,132\n2026-03-03,11800,31,167\n2026-03-04,14500,12,128\n2026-03-05,15200,8,119",
        "notes.txt": "Meeting notes from March 20:\n- Launch date confirmed for April 15\n- Budget approved: $250,000\n- Team size: 8 engineers\n- Key risk: third-party API dependency",
    }
    if filename in files:
        return files[filename]
    return f"Error: file '{filename}' not found. Available files: {', '.join(files.keys())}"

def word_count(text: str) -> str:
    words = text.split()
    chars = len(text)
    lines = text.count("\n") + 1
    return f"Words: {len(words)}, Characters: {chars}, Lines: {lines}"

# ---------------------------------------------------------------------------
# Register all tools
# ---------------------------------------------------------------------------

def register_tools(harness):
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
# Harness (GUI-adapted) — collects log lines instead of printing
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
        self.log_lines: list[str] = []

    def _log(self, tag: str, msg: str):
        self.log_lines.append(f"[{tag:<10}] {msg}")

    def _llm_call(self, messages: list[dict], use_tools: bool = False,
                   call_type: str = "step") -> object:
        kwargs = dict(model=MODEL, messages=messages, temperature=0.3, max_tokens=1024)
        if use_tools and self.config.tools_enabled and self.tools.schemas:
            kwargs["tools"] = self.tools.schemas
            kwargs["tool_choice"] = "auto"
        self.stats["llm_calls"] += 1
        response = get_client().chat.completions.create(**kwargs)

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
        step_result = ""

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
            self._log("CONTEXT", f"System prompt + tools + memory({self.memory.count} entries)")

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
        return step_result

    def run(self, task: str) -> tuple[str, str, str, str, str, str]:
        """Run the harness and return (log, final_answer, memory_state, stats, tokens, audit_json)."""
        t0 = time_mod.time()
        self.log_lines = []

        self._log("CONFIG", self.config.status_line())
        self._log("TASK", f'"{task}"')
        self._log("TOOLS", f'{len(self.tools.names)} registered: {", ".join(self.tools.names)}')
        self.audit.log("task_start", {"task": task, "config": self.config.status_line()})
        self._log("", "")

        # --- Planning ---
        if self.config.planner_enabled:
            self._log("PLANNER", "Decomposing task...")
            steps = self.planner.decompose(task)
            self._log("PLANNER", f"Decomposed into {len(steps)} steps")
            for i, step in enumerate(steps, 1):
                self._log("PLANNER", f"  {i}. {step}")
            self.audit.log("plan", {"steps": steps})
        else:
            steps = [task]
            self._log("PLANNER", "Disabled \u2014 running as single step")
        self._log("", "")

        # --- Execute each step ---
        final_answer = ""
        for i, step in enumerate(steps, 1):
            step_label = f"STEP {i}/{len(steps)}"
            self._log(step_label, step)
            result = self._run_step_with_retry(task, step, i, len(steps))
            if result:
                final_answer = result
            self._log("", "")

        if not final_answer:
            mem = self.memory.recall()
            parts = [f"{k} = {v}" for k, v in mem.items()]
            final_answer = "; ".join(parts) if parts else "No result produced."

        elapsed = time_mod.time() - t0

        self.audit.log("task_complete", {
            "final_answer": final_answer[:200],
            "elapsed_seconds": round(elapsed, 2),
            "total_tokens": self.tokens.total,
            "stats": dict(self.stats),
        })

        # Build outputs
        log_output = "\n".join(self.log_lines)
        memory_output = self.memory.summary()
        stats_output = (
            f"Steps completed:      {len(steps)}\n"
            f"Tool calls:           {self.stats['tool_calls']}\n"
            f"Memory entries:       {self.memory.count}\n"
            f"Verification:         {self.stats['verify_passes']}/{self.stats['verify_total']} passed\n"
            f"Retries:              {self.stats['retries']}\n"
            f"LLM calls:            {self.stats['llm_calls']}\n"
            f"Elapsed:              {elapsed:.1f}s"
        )
        token_output = self.tokens.summary()
        audit_output = self.audit.to_json()

        return log_output, final_answer, memory_output, stats_output, token_output, audit_output


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

EXAMPLE_TASKS = [
    "What is 47 * 83? Then find the square root of that result. Finally summarise both results in a single sentence.",
    "Search for the speed of light. Then calculate how many kilometers light travels in one minute.",
    "Read the file metrics.csv. Find the day with the highest requests and the day with lowest latency.",
    "What is the current date and time? Reverse the string 'Harness Engineering'. Count the words in the reversed result.",
    "Search for the population of France. Search for the largest ocean. Calculate the population divided by 1000.",
]

COMPARISON_CONFIGS = [
    ("All ON", HarnessConfig(tools_enabled=True, memory_enabled=True, planner_enabled=True, verifier_enabled=True, retry_enabled=True)),
    ("No Planner", HarnessConfig(tools_enabled=True, memory_enabled=True, planner_enabled=False, verifier_enabled=True, retry_enabled=True)),
    ("No Verifier", HarnessConfig(tools_enabled=True, memory_enabled=True, planner_enabled=True, verifier_enabled=False, retry_enabled=False)),
    ("No Memory", HarnessConfig(tools_enabled=True, memory_enabled=False, planner_enabled=True, verifier_enabled=True, retry_enabled=True)),
]


def run_harness(task, tools_on, memory_on, planner_on, verifier_on, retry_on):
    if not NVIDIA_API_KEY:
        err = "ERROR: NVIDIA_API_KEY environment variable not set.\nRun: export NVIDIA_API_KEY='your-key'"
        return err, "", "", "", "", ""
    if not task.strip():
        return "ERROR: Please enter a task.", "", "", "", "", ""

    config = HarnessConfig(
        tools_enabled=tools_on,
        memory_enabled=memory_on,
        planner_enabled=planner_on,
        verifier_enabled=verifier_on,
        retry_enabled=retry_on,
    )

    harness = Harness(config)
    register_tools(harness)

    try:
        return harness.run(task)
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
                "Answer": answer[:100],
            })
        except Exception as exc:
            all_logs.append(f"ERROR: {exc}\n")
            rows.append({"Config": name, "Tool Calls": "-", "LLM Calls": "-", "Memory": "-", "Verify": "-", "Retries": "-", "Tokens": "-", "Answer": f"Error: {exc}"})

    # Build summary table
    summary_lines = [f"{'Config':<16} {'Tools':>6} {'LLM':>5} {'Mem':>5} {'Verify':>8} {'Retry':>6} {'Tokens':>8}"]
    summary_lines.append(f"{'-'*16} {'-'*6} {'-'*5} {'-'*5} {'-'*8} {'-'*6} {'-'*8}")
    for r in rows:
        summary_lines.append(f"{r['Config']:<16} {r['Tool Calls']:>6} {r['LLM Calls']:>5} {r['Memory']:>5} {r['Verify']:>8} {r['Retries']:>6} {r['Tokens']:>8}")
    summary_lines.append("")
    for r in rows:
        summary_lines.append(f"[{r['Config']}] {r['Answer']}")

    return "\n".join(all_logs), "\n".join(summary_lines)


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
        "Visual interface for a minimal agent harness using **NVIDIA Nemotron Super 49B**. "
        "Six harness components with retry/self-correction, token tracking, and audit logging."
    )

    with gr.Tabs():
        # ============================================================
        # TAB 1: Single Run
        # ============================================================
        with gr.Tab("Run Task"):
            with gr.Row():
                # --- Left Column: Controls ---
                with gr.Column(scale=1):
                    gr.Markdown("### Harness Config")
                    tools_on = gr.Checkbox(label="Tool Integration", value=True, info="Enable 6 tools (calculator, time, reverse, search, file, word count)")
                    memory_on = gr.Checkbox(label="Memory & State", value=True, info="Store intermediate results in working memory")
                    planner_on = gr.Checkbox(label="Planning", value=True, info="Decompose complex tasks into steps via LLM")
                    verifier_on = gr.Checkbox(label="Verification", value=True, info="Validate outputs with rule-based checks")
                    retry_on = gr.Checkbox(label="Retry / Self-Correction", value=True, info="Retry failed steps with feedback injection (max 2)")

                    gr.Markdown("### Task")
                    task_input = gr.Textbox(
                        label="Enter your task",
                        placeholder="What is 47 * 83? Then find the square root...",
                        lines=3,
                    )
                    run_btn = gr.Button("Run Harness", variant="primary", size="lg")

                    gr.Markdown("### Examples")
                    for ex in EXAMPLE_TASKS:
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
                    stats_output = gr.Textbox(label="", lines=8, interactive=False, elem_classes=["stats-box"])

                    gr.Markdown("### Token Usage")
                    token_output = gr.Textbox(label="", lines=6, interactive=False, elem_classes=["stats-box"])

                    gr.Markdown("### Registered Tools")
                    gr.Markdown(
                        "- **calculator**(expression) \u2014 math evaluation\n"
                        "- **get_current_time**() \u2014 current date/time\n"
                        "- **reverse_string**(text) \u2014 reverse a string\n"
                        "- **web_search**(query) \u2014 search the web\n"
                        "- **read_file**(filename) \u2014 read file contents\n"
                        "- **word_count**(text) \u2014 count words/chars/lines"
                    )

            run_btn.click(
                fn=run_harness,
                inputs=[task_input, tools_on, memory_on, planner_on, verifier_on, retry_on],
                outputs=[log_output, answer_output, memory_output, stats_output, token_output, gr.Textbox(visible=False)],
            )

        # ============================================================
        # TAB 2: Comparison Mode
        # ============================================================
        with gr.Tab("Compare Configs"):
            gr.Markdown(
                "### Comparison Mode\n"
                "Run the same task with 4 different harness configurations and compare results. "
                "Configs: **All ON**, **No Planner**, **No Verifier**, **No Memory**."
            )

            comp_task = gr.Textbox(
                label="Task to compare",
                placeholder="Enter a task to run across all configs...",
                value=EXAMPLE_TASKS[0],
                lines=2,
            )
            comp_btn = gr.Button("Run Comparison", variant="primary", size="lg")

            gr.Markdown("### Summary Table")
            comp_summary = gr.Textbox(label="", lines=10, interactive=False, elem_classes=["stats-box"])

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
                "Run a task from the **Run Task** tab, then view the structured JSON audit trail here. "
                "Every LLM call, tool execution, verification, memory store, and retry is recorded with timestamps."
            )

            audit_task = gr.Textbox(
                label="Task",
                value=EXAMPLE_TASKS[0],
                lines=2,
            )
            audit_btn = gr.Button("Run & Generate Audit Log", variant="primary")
            audit_output = gr.Textbox(label="Audit Log (JSON)", lines=30, max_lines=60, interactive=False, elem_classes=["log-box"])

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

            audit_btn.click(fn=run_audit, inputs=[audit_task], outputs=[audit_output])

    gr.Markdown(
        "---\n"
        "*Agent Harness Engineering \u2014 Cobus Greyling*"
    )

if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css=CSS,
    )
