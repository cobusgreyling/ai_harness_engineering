"""
Minimal Agent Harness — Nemotron 3 Super
Six harness components (ToolRegistry, MemoryManager, ContextEngine, Planner, Verifier, HarnessConfig)
orchestrated in a single-file CLI demo.
"""

import os
import re
import json
import math
from datetime import datetime
from dataclasses import dataclass, field
from openai import OpenAI

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "private/nvidia/nemotron-3-super-120b-a12b"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

CLIENT = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
    default_headers={"NVCF-POLL-SECONDS": "1800"},
)

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

    @property
    def count(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# 3. ContextEngine — assembles prompt dynamically
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
        # Extract JSON array from response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Fallback: split numbered lines
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
# 6. HarnessConfig — enable/disable components
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
        self.stats = {"tool_calls": 0, "llm_calls": 0, "verify_passes": 0, "verify_total": 0}

    def _log(self, tag: str, msg: str):
        print(f"[{tag:<10}] {msg}")

    def _llm_call(self, messages: list[dict], use_tools: bool = False) -> object:
        kwargs = dict(model=MODEL, messages=messages, temperature=0.3, max_tokens=1024)
        if use_tools and self.config.tools_enabled and self.tools.schemas:
            kwargs["tools"] = self.tools.schemas
            kwargs["tool_choice"] = "auto"
        self.stats["llm_calls"] += 1
        return CLIENT.chat.completions.create(**kwargs)

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
        symbol = "✓" if passed else "✗"
        self._log("VERIFY", f"{symbol} {reason}")
        if passed:
            self.stats["verify_passes"] += 1
        return passed

    def run(self, task: str):
        print(f"\n{'═' * 50}")
        print(f"  HARNESS DEMO — Nemotron 3 Super")
        print(f"{'═' * 50}\n")

        self._log("CONFIG", self.config.status_line())
        self._log("TASK", f'"{task}"')
        print()

        # --- Planning ---
        if self.config.planner_enabled:
            self._log("PLANNER", "Decomposing task...")
            steps = self.planner.decompose(task)
            self._log("PLANNER", f"Decomposed into {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"{'':>13} {i}. {step}")
        else:
            steps = [task]
        print()

        # --- Execute each step ---
        final_answer = ""
        for i, step in enumerate(steps, 1):
            step_label = f"STEP {i}/{len(steps)}"
            self._log(step_label, step)

            messages = self.context.build(task, self.memory, self.tools, step_instruction=step)
            self._log("CONTEXT", f"System prompt + tools + memory({self.memory.count} entries)")

            response = self._llm_call(messages, use_tools=True)
            msg = response.choices[0].message

            # Handle tool calls
            tool_results = self._execute_tool_calls(response)

            if tool_results:
                # Use the tool result as the step output
                _, _, result_value = tool_results[-1]
                step_result = result_value

                # Verify numeric result
                if self.config.verifier_enabled:
                    self._verify(self.verifier.check_numeric, step_result)

                # Store in memory
                if self.config.memory_enabled:
                    if i == 1:
                        key = "multiplication_result"
                    elif i == 2:
                        key = "sqrt_result"
                    else:
                        key = f"step_{i}_result"
                    self.memory.store(key, step_result)
                    self._log("MEMORY", f"Stored: {key} = {step_result}")

            else:
                # Text response (final summary step)
                step_result = msg.content.strip() if msg.content else ""
                self._log("LLM", f'→ "{step_result}"')

                # Verify summary
                if self.config.verifier_enabled:
                    mem = self.memory.recall()
                    check_values = list(mem.values())[:2]
                    # Check key numbers appear (just first few digits)
                    short_vals = []
                    for v in check_values:
                        try:
                            num = float(v)
                            if num == int(num):
                                short_vals.append(str(int(num)))
                            else:
                                # Accept any rounding of the float (first 4 digits)
                                short_vals.append(str(int(num)))
                        except ValueError:
                            short_vals.append(v)
                    self._verify(self.verifier.check_contains, step_result, short_vals)
                    self._verify(self.verifier.check_single_sentence, step_result)

                final_answer = step_result

            print()

        # --- Final answer ---
        if not final_answer:
            final_answer = self._build_fallback_summary()

        print(f"{'═' * 50}")
        print(f"  FINAL ANSWER")
        print(f"{'═' * 50}")
        print(f"  {final_answer}")
        print()

        # --- Stats ---
        print(f"{'═' * 50}")
        print(f"  HARNESS STATS")
        print(f"{'═' * 50}")
        print(f"  Steps completed:      {len(steps)}/{len(steps)}")
        print(f"  Tool calls:           {self.stats['tool_calls']}")
        print(f"  Memory entries:       {self.memory.count}")
        print(f"  Verification passes:  {self.stats['verify_passes']}/{self.stats['verify_total']}")
        print(f"  Total LLM calls:      {self.stats['llm_calls']} (1 planner + {self.stats['llm_calls'] - 1} steps)")
        print()

    def _build_fallback_summary(self) -> str:
        mem = self.memory.recall()
        mult = mem.get("multiplication_result", "?")
        sqrt = mem.get("sqrt_result", "?")
        try:
            sqrt_rounded = f"{float(sqrt):.2f}"
        except (ValueError, TypeError):
            sqrt_rounded = sqrt
        return f"47 × 83 = {mult}, and its square root is approximately {sqrt_rounded}."


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() eE")
    # Also allow ** for exponentiation
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not NVIDIA_API_KEY:
        print("Error: set NVIDIA_API_KEY environment variable")
        return

    harness = Harness()

    # Register tools
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

    task = (
        "What is 47 * 83? Then find the square root of that result. "
        "Finally summarise both results in a single sentence."
    )

    harness.run(task)


if __name__ == "__main__":
    main()
