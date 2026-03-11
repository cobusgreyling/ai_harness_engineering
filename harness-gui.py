"""
Agent Harness GUI — Gradio frontend for the Minimal Agent Harness
Wraps the six harness components with a visual interface.
"""

import os
import re
import json
import math
from datetime import datetime
from dataclasses import dataclass
from openai import OpenAI
import gradio as gr

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "private/nvidia/nemotron-3-super-120b-a12b"
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

    @property
    def count(self) -> int:
        return len(self._store)

# ---------------------------------------------------------------------------
# 3. ContextEngine
# ---------------------------------------------------------------------------

class ContextEngine:
    SYSTEM_BASE = (
        "You are a precise assistant. When you need to compute something, "
        "call the appropriate tool. Return only the requested information."
    )

    def build(self, task: str, memory: MemoryManager, tools: ToolRegistry,
              step_instruction: str | None = None) -> list[dict]:
        parts = [self.SYSTEM_BASE]
        parts.append(f"\nAvailable tools:\n{tools.descriptions}")
        mem = memory.summary()
        if mem != "No stored memory.":
            parts.append(f"\nWorking memory:\n{mem}")
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
        missing = [v for v in values if v not in text]
        if missing:
            return False, f"Missing values: {missing}"
        return True, "Contains all expected values"

    def check_single_sentence(self, text: str) -> tuple[bool, str]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) <= 2:
            return True, "Is a single sentence"
        return False, f"Contains {len(sentences)} sentences, expected 1"

# ---------------------------------------------------------------------------
# 6. HarnessConfig
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    tools_enabled: bool = True
    memory_enabled: bool = True
    planner_enabled: bool = True
    verifier_enabled: bool = True

    def status_line(self) -> str:
        flags = {
            "Tools": self.tools_enabled,
            "Memory": self.memory_enabled,
            "Planner": self.planner_enabled,
            "Verifier": self.verifier_enabled,
        }
        return " | ".join(f"{k}: {'ON' if v else 'OFF'}" for k, v in flags.items())

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
        self.stats = {"tool_calls": 0, "llm_calls": 0, "verify_passes": 0, "verify_total": 0}
        self.log_lines: list[str] = []

    def _log(self, tag: str, msg: str):
        self.log_lines.append(f"[{tag:<10}] {msg}")

    def _llm_call(self, messages: list[dict], use_tools: bool = False) -> object:
        kwargs = dict(model=MODEL, messages=messages, temperature=0.3, max_tokens=1024)
        if use_tools and self.config.tools_enabled and self.tools.schemas:
            kwargs["tools"] = self.tools.schemas
            kwargs["tool_choice"] = "auto"
        self.stats["llm_calls"] += 1
        return get_client().chat.completions.create(**kwargs)

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
            results.append((name, args_str, result))
        return results

    def _verify(self, check_fn, *args) -> bool:
        self.stats["verify_total"] += 1
        passed, reason = check_fn(*args)
        symbol = "PASS" if passed else "FAIL"
        self._log("VERIFY", f"[{symbol}] {reason}")
        if passed:
            self.stats["verify_passes"] += 1
        return passed

    def run(self, task: str) -> tuple[str, str, str, str]:
        """Run the harness and return (log, final_answer, memory_state, stats)."""
        self.log_lines = []

        self._log("CONFIG", self.config.status_line())
        self._log("TASK", f'"{task}"')
        self._log("", "")

        # --- Planning ---
        if self.config.planner_enabled:
            self._log("PLANNER", "Decomposing task...")
            steps = self.planner.decompose(task)
            self._log("PLANNER", f"Decomposed into {len(steps)} steps")
            for i, step in enumerate(steps, 1):
                self._log("PLANNER", f"  {i}. {step}")
        else:
            steps = [task]
        self._log("", "")

        # --- Execute each step ---
        final_answer = ""
        for i, step in enumerate(steps, 1):
            step_label = f"STEP {i}/{len(steps)}"
            self._log(step_label, step)

            messages = self.context.build(task, self.memory, self.tools, step_instruction=step)
            self._log("CONTEXT", f"System prompt + tools + memory({self.memory.count} entries)")

            response = self._llm_call(messages, use_tools=True)
            msg = response.choices[0].message

            tool_results = self._execute_tool_calls(response)

            if tool_results:
                _, _, result_value = tool_results[-1]
                step_result = result_value

                if self.config.verifier_enabled:
                    self._verify(self.verifier.check_numeric, step_result)

                if self.config.memory_enabled:
                    key = f"step_{i}_result"
                    self.memory.store(key, step_result)
                    self._log("MEMORY", f"Stored: {key} = {step_result}")
            else:
                step_result = msg.content.strip() if msg.content else ""
                self._log("LLM", f'{step_result}')

                if self.config.verifier_enabled:
                    mem = self.memory.recall()
                    check_values = list(mem.values())[:2]
                    short_vals = []
                    for v in check_values:
                        try:
                            num = float(v)
                            short_vals.append(str(int(num)))
                        except ValueError:
                            short_vals.append(v)
                    if short_vals:
                        self._verify(self.verifier.check_contains, step_result, short_vals)
                    self._verify(self.verifier.check_single_sentence, step_result)

                final_answer = step_result

            self._log("", "")

        if not final_answer:
            mem = self.memory.recall()
            parts = [f"{k} = {v}" for k, v in mem.items()]
            final_answer = "; ".join(parts) if parts else "No result produced."

        # Build outputs
        log_output = "\n".join(self.log_lines)
        memory_output = self.memory.summary()
        stats_output = (
            f"Steps completed:      {len(steps)}/{len(steps)}\n"
            f"Tool calls:           {self.stats['tool_calls']}\n"
            f"Memory entries:       {self.memory.count}\n"
            f"Verification passes:  {self.stats['verify_passes']}/{self.stats['verify_total']}\n"
            f"Total LLM calls:      {self.stats['llm_calls']}"
        )

        return log_output, final_answer, memory_output, stats_output


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

def run_harness(task, tools_on, memory_on, planner_on, verifier_on):
    if not NVIDIA_API_KEY:
        return (
            "ERROR: NVIDIA_API_KEY environment variable not set.\n"
            "Run: export NVIDIA_API_KEY='your-key'",
            "", "", ""
        )

    if not task.strip():
        return "ERROR: Please enter a task.", "", "", ""

    config = HarnessConfig(
        tools_enabled=tools_on,
        memory_enabled=memory_on,
        planner_enabled=planner_on,
        verifier_enabled=verifier_on,
    )

    harness = Harness(config)

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

    try:
        log_output, final_answer, memory_output, stats_output = harness.run(task)
    except Exception as exc:
        return f"ERROR: {exc}", "", "", ""

    return log_output, final_answer, memory_output, stats_output


EXAMPLE_TASKS = [
    "What is 47 * 83? Then find the square root of that result. Finally summarise both results in a single sentence.",
    "What time is it right now?",
    "Reverse the string 'Hello World' and tell me the result.",
    "Calculate 2**10 and then multiply by 3. Summarise.",
]

CSS = """
.log-box textarea { font-family: 'SF Mono', 'Fira Code', monospace !important; font-size: 13px !important; }
.answer-box textarea { font-size: 15px !important; font-weight: 500 !important; }
.stats-box textarea { font-family: 'SF Mono', 'Fira Code', monospace !important; font-size: 13px !important; }
"""

with gr.Blocks(title="Agent Harness Demo") as demo:

    gr.Markdown("# Agent Harness Demo")
    gr.Markdown(
        "Visual interface for a minimal agent harness using NVIDIA Nemotron 3 Super. "
        "Toggle harness components on/off to see how each one affects execution."
    )

    with gr.Row():
        # --- Left Column: Controls ---
        with gr.Column(scale=1):
            gr.Markdown("### Harness Config")
            tools_on = gr.Checkbox(label="Tool Integration", value=True, info="Enable tool calling (calculator, time, reverse)")
            memory_on = gr.Checkbox(label="Memory & State", value=True, info="Store intermediate results in working memory")
            planner_on = gr.Checkbox(label="Planning", value=True, info="Decompose complex tasks into steps via LLM")
            verifier_on = gr.Checkbox(label="Verification", value=True, info="Validate outputs with rule-based checks")

            gr.Markdown("### Task")
            task_input = gr.Textbox(
                label="Enter your task",
                placeholder="What is 47 * 83? Then find the square root...",
                lines=3,
            )
            run_btn = gr.Button("Run Harness", variant="primary", size="lg")

            gr.Markdown("### Examples")
            for ex in EXAMPLE_TASKS:
                gr.Button(ex[:60] + "..." if len(ex) > 60 else ex, size="sm").click(
                    fn=lambda e=ex: e, outputs=task_input
                )

        # --- Centre Column: Execution Log ---
        with gr.Column(scale=2):
            gr.Markdown("### Execution Log")
            log_output = gr.Textbox(
                label="",
                lines=20,
                max_lines=40,
                interactive=False,
                elem_classes=["log-box"],
            )

            gr.Markdown("### Final Answer")
            answer_output = gr.Textbox(
                label="",
                lines=3,
                interactive=False,
                elem_classes=["answer-box"],
            )

        # --- Right Column: State ---
        with gr.Column(scale=1):
            gr.Markdown("### Working Memory")
            memory_output = gr.Textbox(
                label="",
                lines=8,
                interactive=False,
                elem_classes=["stats-box"],
            )

            gr.Markdown("### Harness Stats")
            stats_output = gr.Textbox(
                label="",
                lines=6,
                interactive=False,
                elem_classes=["stats-box"],
            )

            gr.Markdown("### Registered Tools")
            gr.Markdown(
                "- **calculator**(expression) — math evaluation\n"
                "- **get_current_time**() — current date/time\n"
                "- **reverse_string**(text) — reverse a string"
            )

    run_btn.click(
        fn=run_harness,
        inputs=[task_input, tools_on, memory_on, planner_on, verifier_on],
        outputs=[log_output, answer_output, memory_output, stats_output],
    )

    gr.Markdown(
        "---\n"
        "*Minimal Agent Harness — Cobus Greyling, Chief AI Evangelist @ Kore.ai*"
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
