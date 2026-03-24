"""
Harness Core — Shared components for all harness interfaces.
Extracts and enhances the six harness components + new features:
  - Token budget enforcement
  - SQLite persistent memory
  - Advanced guardrails (PII, length, JSON schema, rate limiting)
  - YAML config loading
  - Human-in-the-loop approval callbacks
  - Sub-agent spawning
  - Enhanced tools (http_fetch, code_execute, json_extract)
  - Replay trace recording
"""

import os
import re
import json
import math
import time as time_mod
import sqlite3
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

_CLIENT = None


def get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
            default_headers={"NVCF-POLL-SECONDS": "1800"},
        )
    return _CLIENT


MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# 1. ToolRegistry — register and execute tools
# ---------------------------------------------------------------------------


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, callable] = {}
        self._schemas: list[dict] = []
        self._risk_levels: dict[str, str] = {}  # tool_name -> "low"|"medium"|"high"

    def register(self, name: str, func: callable, description: str,
                 parameters: dict, risk: str = "low"):
        self._tools[name] = func
        self._risk_levels[name] = risk
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

    def get_risk(self, name: str) -> str:
        return self._risk_levels.get(name, "low")

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
# 2. MemoryManager — working memory + SQLite persistence
# ---------------------------------------------------------------------------


class MemoryManager:
    def __init__(self, persistent: bool = False, db_path: str = "harness_memory.db"):
        self._store: dict[str, str] = {}
        self._persistent = persistent
        self._db_path = db_path
        if persistent:
            self._init_db()
            self._load_from_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                session_id TEXT,
                key TEXT,
                value TEXT,
                created_at TEXT,
                PRIMARY KEY (session_id, key)
            )
        """)
        conn.commit()
        conn.close()

    def _load_from_db(self):
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute(
            "SELECT key, value FROM memory WHERE session_id = 'current' ORDER BY created_at"
        ).fetchall()
        conn.close()
        for key, value in rows:
            self._store[key] = value

    def store(self, key: str, value: str):
        self._store[key] = value
        if self._persistent:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT OR REPLACE INTO memory (session_id, key, value, created_at) VALUES (?, ?, ?, ?)",
                ("current", key, value, datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()

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
        if self._persistent:
            conn = sqlite3.connect(self._db_path)
            conn.execute("DELETE FROM memory WHERE session_id = 'current'")
            conn.commit()
            conn.close()

    def list_sessions(self) -> list[str]:
        if not self._persistent:
            return []
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT DISTINCT session_id FROM memory").fetchall()
        conn.close()
        return [r[0] for r in rows]

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
              retry_feedback: str | None = None,
              guardrail_instructions: str | None = None) -> list[dict]:
        parts = [self.SYSTEM_BASE]
        parts.append(f"\nAvailable tools:\n{tools.descriptions}")
        mem = memory.summary()
        if mem != "No stored memory.":
            parts.append(f"\nWorking memory:\n{mem}")
        if guardrail_instructions:
            parts.append(f"\nGUARDRAILS:\n{guardrail_instructions}")
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
        response = get_client().chat.completions.create(
            model=MODEL, messages=messages, temperature=0.2, max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return [line.strip().lstrip("0123456789.) ") for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 5. Verifier + Advanced Guardrails
# ---------------------------------------------------------------------------

# PII patterns
PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
}

TOXIC_KEYWORDS = [
    "kill", "hack into", "steal", "exploit vulnerability",
    "bypass security", "inject malicious", "phishing",
]


class Guardrails:
    """Advanced guardrail checks beyond basic verification."""

    @staticmethod
    def check_pii(text: str) -> tuple[bool, str]:
        found = []
        for pii_type, pattern in PII_PATTERNS.items():
            if re.search(pattern, text):
                found.append(pii_type)
        if found:
            return False, f"PII detected: {', '.join(found)}"
        return True, "No PII detected"

    @staticmethod
    def check_toxicity(text: str) -> tuple[bool, str]:
        text_lower = text.lower()
        found = [kw for kw in TOXIC_KEYWORDS if kw in text_lower]
        if found:
            return False, f"Potentially harmful content: {', '.join(found)}"
        return True, "No toxic content detected"

    @staticmethod
    def check_max_length(text: str, max_chars: int = 5000) -> tuple[bool, str]:
        if len(text) > max_chars:
            return False, f"Response too long: {len(text)} chars (max {max_chars})"
        return True, f"Length OK: {len(text)} chars"

    @staticmethod
    def check_json_schema(text: str) -> tuple[bool, str]:
        try:
            json.loads(text)
            return True, "Valid JSON"
        except (json.JSONDecodeError, ValueError):
            return False, "Not valid JSON"

    @staticmethod
    def check_no_code_injection(text: str) -> tuple[bool, str]:
        patterns = [r'<script', r'javascript:', r'eval\s*\(', r'exec\s*\(', r'__import__']
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Potential code injection detected"
        return True, "No code injection detected"


class Verifier:
    def __init__(self, guardrails_enabled: bool = False):
        self.guardrails_enabled = guardrails_enabled
        self.guardrails = Guardrails()

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

    def run_guardrails(self, text: str) -> list[tuple[bool, str]]:
        if not self.guardrails_enabled:
            return []
        results = [
            ("PII", self.guardrails.check_pii(text)),
            ("Toxicity", self.guardrails.check_toxicity(text)),
            ("Length", self.guardrails.check_max_length(text)),
            ("Injection", self.guardrails.check_no_code_injection(text)),
        ]
        return results


# ---------------------------------------------------------------------------
# 6. HarnessConfig — enable/disable components + YAML loading
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    name: str = "default"
    tools_enabled: bool = True
    memory_enabled: bool = True
    planner_enabled: bool = True
    verifier_enabled: bool = True
    retry_enabled: bool = True
    guardrails_enabled: bool = False
    persistent_memory: bool = False
    token_budget: int = 0  # 0 = unlimited
    human_approval: bool = False
    sub_agents_enabled: bool = False
    max_retries: int = 2

    def status_line(self) -> str:
        flags = {
            "Tools": self.tools_enabled,
            "Memory": self.memory_enabled,
            "Planner": self.planner_enabled,
            "Verifier": self.verifier_enabled,
            "Retry": self.retry_enabled,
            "Guardrails": self.guardrails_enabled,
            "Persistence": self.persistent_memory,
            "Approval": self.human_approval,
            "SubAgents": self.sub_agents_enabled,
        }
        parts = [f"{k}: {'ON' if v else 'OFF'}" for k, v in flags.items()]
        if self.token_budget > 0:
            parts.append(f"Budget: {self.token_budget}")
        return " | ".join(parts)

    @classmethod
    def from_yaml(cls, path: str) -> "HarnessConfig":
        """Load config from a YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, data: dict) -> "HarnessConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def load_yaml_configs(config_dir: str = "configs") -> dict[str, HarnessConfig]:
    """Load all YAML configs from a directory."""
    configs = {}
    p = Path(config_dir)
    if not p.exists():
        return configs
    try:
        import yaml
    except ImportError:
        return configs
    for f in sorted(p.glob("*.yaml")):
        try:
            with open(f) as fh:
                data = yaml.safe_load(fh)
            configs[f.stem] = HarnessConfig.from_dict(data)
        except Exception:
            pass
    return configs


# ---------------------------------------------------------------------------
# Token Tracker with Budget Enforcement
# ---------------------------------------------------------------------------


class TokenBudgetExceeded(Exception):
    pass


@dataclass
class TokenTracker:
    calls: list = field(default_factory=list)
    budget: int = 0  # 0 = unlimited

    def record(self, call_type: str, input_tokens: int, output_tokens: int):
        self.calls.append({
            "type": call_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "timestamp": datetime.now().isoformat(),
        })

    def check_budget(self):
        if self.budget > 0 and self.total >= self.budget:
            raise TokenBudgetExceeded(
                f"Token budget exhausted: {self.total}/{self.budget} tokens used"
            )

    @property
    def budget_remaining(self) -> int:
        if self.budget <= 0:
            return -1  # unlimited
        return max(0, self.budget - self.total)

    @property
    def budget_pct(self) -> float:
        if self.budget <= 0:
            return 0.0
        return min(100.0, (self.total / self.budget) * 100)

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
        budget_str = f" (budget: {self.budget})" if self.budget > 0 else ""
        lines.append(f"TOTAL: {self.total_input} in + {self.total_output} out = {self.total} tokens ({self.call_count} calls){budget_str}")
        if self.budget > 0:
            lines.append(f"Budget: {self.budget_pct:.1f}% used, {self.budget_remaining} remaining")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audit Log + Replay Trace
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

    def get_replay_events(self) -> list[dict]:
        """Return events formatted for step-by-step replay."""
        replay = []
        for entry in self.entries:
            event = entry["event"]
            ts = entry["timestamp"]
            if event == "task_start":
                replay.append({"step": "Start", "time": ts, "type": "start",
                               "detail": f"Task: {entry.get('task', '')[:100]}"})
            elif event == "plan":
                steps = entry.get("steps", [])
                replay.append({"step": "Plan", "time": ts, "type": "plan",
                               "detail": f"{len(steps)} steps: " + " → ".join(s[:40] for s in steps)})
            elif event == "llm_call":
                replay.append({"step": "LLM Call", "time": ts, "type": "llm",
                               "detail": f"{entry.get('call_type', '')}: {entry.get('input_tokens', 0)} in + {entry.get('output_tokens', 0)} out"})
            elif event == "tool_call":
                replay.append({"step": "Tool", "time": ts, "type": "tool",
                               "detail": f"{entry.get('tool', '')}({json.dumps(entry.get('args', {}))}) → {str(entry.get('result', ''))[:80]}"})
            elif event == "verification":
                symbol = "✓" if entry.get("passed") else "✗"
                replay.append({"step": "Verify", "time": ts, "type": "verify",
                               "detail": f"{symbol} {entry.get('reason', '')}"})
            elif event == "guardrail":
                symbol = "✓" if entry.get("passed") else "✗"
                replay.append({"step": "Guardrail", "time": ts, "type": "guardrail",
                               "detail": f"{symbol} [{entry.get('check', '')}] {entry.get('reason', '')}"})
            elif event == "memory_store":
                replay.append({"step": "Memory", "time": ts, "type": "memory",
                               "detail": f"Store: {entry.get('key', '')} = {str(entry.get('value', ''))[:60]}"})
            elif event == "retry":
                replay.append({"step": "Retry", "time": ts, "type": "retry",
                               "detail": f"Step {entry.get('step', '')} attempt {entry.get('attempt', '')}"})
            elif event == "approval_requested":
                replay.append({"step": "Approval", "time": ts, "type": "approval",
                               "detail": f"Tool: {entry.get('tool', '')} — {entry.get('status', '')}"})
            elif event == "budget_warning":
                replay.append({"step": "Budget", "time": ts, "type": "budget",
                               "detail": entry.get("message", "")})
            elif event == "sub_agent":
                replay.append({"step": "SubAgent", "time": ts, "type": "subagent",
                               "detail": f"{entry.get('agent_name', '')}: {entry.get('task', '')[:60]}"})
            elif event == "task_complete":
                replay.append({"step": "Complete", "time": ts, "type": "complete",
                               "detail": f"Answer: {entry.get('final_answer', '')[:80]} ({entry.get('total_tokens', 0)} tokens, {entry.get('elapsed_seconds', 0)}s)"})
        return replay


# ---------------------------------------------------------------------------
# Tool implementations (original + new)
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    safe_ns = {"__builtins__": {}, "sqrt": math.sqrt, "sin": math.sin,
               "cos": math.cos, "log": math.log, "pi": math.pi, "e": math.e,
               "abs": abs, "round": round, "pow": pow}
    try:
        result = eval(expression, safe_ns)
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
    """Simulated web search with expanded results."""
    results = {
        "population of france": "France has a population of approximately 68.4 million people (2025 estimate).",
        "tallest building": "The Burj Khalifa in Dubai is the tallest building at 828 meters (2,717 feet).",
        "speed of light": "The speed of light in vacuum is 299,792,458 meters per second.",
        "largest ocean": "The Pacific Ocean is the largest ocean, covering approximately 165.25 million square kilometers.",
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
        "earth circumference": "Earth's circumference is approximately 40,075 km at the equator.",
        "boiling point water": "Water boils at 100°C (212°F) at standard atmospheric pressure.",
        "distance earth moon": "The average distance from Earth to the Moon is 384,400 km.",
        "bitcoin creator": "Bitcoin was created by Satoshi Nakamoto, first described in a 2008 whitepaper.",
        "largest country": "Russia is the largest country by area at 17.1 million square kilometers.",
        "deepest ocean": "The Mariana Trench in the Pacific Ocean is the deepest point at 10,994 meters.",
        "fastest animal": "The peregrine falcon is the fastest animal, reaching speeds over 389 km/h in a dive.",
    }
    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower or any(w in query_lower for w in key.split()):
            return value
    return f"Search result for '{query}': No specific result found. The query returned general information about the topic."


def read_file(filename: str) -> str:
    """Read simulated or real files."""
    simulated = {
        "config.yaml": "app_name: MyAgent\nversion: 2.1.0\nmax_retries: 3\ntimeout: 30\nlog_level: INFO",
        "metrics.csv": "date,requests,errors,latency_ms\n2026-03-01,12450,23,145\n2026-03-02,13200,18,132\n2026-03-03,11800,31,167\n2026-03-04,14500,12,128\n2026-03-05,15200,8,119",
        "notes.txt": "Meeting notes from March 20:\n- Launch date confirmed for April 15\n- Budget approved: $250,000\n- Team size: 8 engineers\n- Key risk: third-party API dependency",
        "users.json": '[\n  {"name": "Alice", "role": "engineer", "projects": 5},\n  {"name": "Bob", "role": "designer", "projects": 3},\n  {"name": "Carol", "role": "PM", "projects": 8}\n]',
        "errors.log": "2026-03-20 10:15:32 ERROR DB connection timeout after 30s\n2026-03-20 10:16:01 WARN Retry attempt 1/3\n2026-03-20 10:16:15 ERROR DB connection timeout after 30s\n2026-03-20 10:17:00 INFO Connection restored\n2026-03-20 14:22:11 ERROR OutOfMemoryError in worker-3",
    }
    if filename in simulated:
        return simulated[filename]
    # Try reading a real file (sandboxed to current directory)
    safe_path = Path(filename).name  # strip directory traversal
    if Path(safe_path).exists():
        try:
            return Path(safe_path).read_text()[:2000]
        except Exception as e:
            return f"Error reading '{safe_path}': {e}"
    return f"Error: file '{filename}' not found. Available: {', '.join(simulated.keys())}"


def word_count(text: str) -> str:
    words = text.split()
    chars = len(text)
    lines = text.count("\n") + 1
    return f"Words: {len(words)}, Characters: {chars}, Lines: {lines}"


def json_extract(json_text: str, key: str) -> str:
    """Extract a value from a JSON string by key."""
    try:
        data = json.loads(json_text)
        if isinstance(data, dict) and key in data:
            return json.dumps(data[key])
        if isinstance(data, list):
            results = [item.get(key) for item in data if isinstance(item, dict) and key in item]
            return json.dumps(results) if results else f"Key '{key}' not found in array items"
        return f"Key '{key}' not found"
    except json.JSONDecodeError:
        return "Error: invalid JSON input"


def text_summarize(text: str) -> str:
    """Use LLM to summarize text."""
    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize the following text in 1-2 sentences. Be concise."},
                {"role": "user", "content": text[:2000]},
            ],
            temperature=0.3, max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def code_execute(code: str) -> str:
    """Execute Python code in a sandboxed environment. Only safe operations allowed."""
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "format": format,
        "frozenset": frozenset, "int": int, "isinstance": isinstance,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "print": lambda *a, **kw: None,  # suppress print
        "range": range, "reversed": reversed, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type,
        "zip": zip,
    }
    safe_globals = {"__builtins__": safe_builtins, "math": math, "json": json}
    output_capture = []
    safe_globals["print"] = lambda *a: output_capture.append(" ".join(str(x) for x in a))
    try:
        exec(code, safe_globals)
        if output_capture:
            return "\n".join(output_capture)
        # Try to find a result variable
        for var in ["result", "output", "answer"]:
            if var in safe_globals:
                return str(safe_globals[var])
        return "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Register all tools on a harness
# ---------------------------------------------------------------------------


def register_tools(harness):
    """Register all tools including new enhanced ones."""
    harness.tools.register(
        "calculator", calculator,
        "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, log.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}}, "required": ["expression"]},
        risk="low",
    )
    harness.tools.register(
        "get_current_time", get_current_time,
        "Get the current date and time.",
        {"type": "object", "properties": {}},
        risk="low",
    )
    harness.tools.register(
        "reverse_string", reverse_string,
        "Reverse a given string.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "Text to reverse"}}, "required": ["text"]},
        risk="low",
    )
    harness.tools.register(
        "web_search", web_search,
        "Search the web for information on a topic. Returns a text summary.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]},
        risk="low",
    )
    harness.tools.register(
        "read_file", read_file,
        "Read the contents of a file. Available: config.yaml, metrics.csv, notes.txt, users.json, errors.log",
        {"type": "object", "properties": {"filename": {"type": "string", "description": "Name of the file to read"}}, "required": ["filename"]},
        risk="medium",
    )
    harness.tools.register(
        "word_count", word_count,
        "Count words, characters, and lines in a text string.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "Text to count"}}, "required": ["text"]},
        risk="low",
    )
    harness.tools.register(
        "json_extract", json_extract,
        "Extract a value from a JSON string by key.",
        {"type": "object", "properties": {
            "json_text": {"type": "string", "description": "JSON string to parse"},
            "key": {"type": "string", "description": "Key to extract"},
        }, "required": ["json_text", "key"]},
        risk="low",
    )
    harness.tools.register(
        "text_summarize", text_summarize,
        "Summarize a long text into 1-2 concise sentences using AI.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "Text to summarize"}}, "required": ["text"]},
        risk="low",
    )
    harness.tools.register(
        "code_execute", code_execute,
        "Execute Python code in a safe sandbox. Use 'result' variable or print() for output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}}, "required": ["code"]},
        risk="high",
    )


# ---------------------------------------------------------------------------
# Example tasks (expanded from 5 to 10)
# ---------------------------------------------------------------------------

EXAMPLE_TASKS = [
    {"name": "Multi-step math", "task": "What is 47 * 83? Then find the square root of that result. Finally summarise both results in a single sentence."},
    {"name": "Web search + calculation", "task": "Search for the speed of light. Then calculate how many kilometers light travels in one minute. Summarise the answer."},
    {"name": "File analysis", "task": "Read the file metrics.csv. Find the day with the highest number of requests and the day with the lowest latency. Summarise both findings."},
    {"name": "Multi-tool chain", "task": "What is the current date and time? Reverse the string 'Harness Engineering'. Count the words in the reversed result. Summarise all three results."},
    {"name": "Research + memory recall", "task": "Search for the population of France. Search for the largest ocean. Calculate the population divided by 1000. Summarise all findings in a single sentence using the stored results."},
    {"name": "JSON data extraction", "task": "Read the file users.json. Extract the 'name' field from the JSON. Then count how many users there are using the calculator."},
    {"name": "Error log analysis", "task": "Read the file errors.log. Count the words in it. Summarise what errors occurred and when."},
    {"name": "Code execution", "task": "Use code_execute to calculate the first 10 Fibonacci numbers. Then summarise the result."},
    {"name": "Multi-source research", "task": "Search for the distance from Earth to the Moon. Search for the speed of light. Calculate how long it takes light to travel from Earth to the Moon in seconds."},
    {"name": "Config audit", "task": "Read config.yaml. Read notes.txt. Summarise what the app version is and what the key risk is from the meeting notes."},
]


# ---------------------------------------------------------------------------
# Sub-Agent
# ---------------------------------------------------------------------------


class SubAgent:
    """A lightweight sub-agent that handles a specific task with its own context."""

    def __init__(self, name: str, system_prompt: str, tools: ToolRegistry | None = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        kwargs = dict(model=MODEL, messages=messages, temperature=0.3, max_tokens=1024)
        if self.tools and self.tools.schemas:
            kwargs["tools"] = self.tools.schemas
            kwargs["tool_choice"] = "auto"
        response = get_client().chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # Execute any tool calls
        if msg.tool_calls:
            results = []
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = self.tools.execute(name, args) if self.tools else f"No tools available"
                results.append(result)
            return "; ".join(results)
        return msg.content.strip() if msg.content else ""


SUB_AGENT_DEFINITIONS = {
    "researcher": SubAgent(
        "researcher",
        "You are a research specialist. Find and return factual information. Be precise and cite sources when possible.",
    ),
    "analyst": SubAgent(
        "analyst",
        "You are a data analyst. Analyze data, find patterns, and provide clear summaries with numbers.",
    ),
    "writer": SubAgent(
        "writer",
        "You are a concise writer. Take raw information and craft clear, well-structured summaries.",
    ),
}


# ---------------------------------------------------------------------------
# Main Harness — orchestrates the agent loop
# ---------------------------------------------------------------------------


class Harness:
    def __init__(self, config: HarnessConfig | None = None):
        self.config = config or HarnessConfig()
        self.tools = ToolRegistry()
        self.memory = MemoryManager(
            persistent=self.config.persistent_memory,
        )
        self.context = ContextEngine()
        self.planner = Planner()
        self.verifier = Verifier(guardrails_enabled=self.config.guardrails_enabled)
        self.tokens = TokenTracker(budget=self.config.token_budget)
        self.audit = AuditLog()
        self.stats = {
            "tool_calls": 0, "llm_calls": 0,
            "verify_passes": 0, "verify_total": 0,
            "retries": 0, "guardrail_checks": 0,
            "guardrail_blocks": 0, "sub_agent_calls": 0,
            "approvals_requested": 0, "approvals_granted": 0,
        }
        self.log_lines: list[str] = []
        self._approval_callback: Callable | None = None
        self._rate_limit_tracker: dict[str, list[float]] = {}

    def set_approval_callback(self, callback: Callable):
        """Set a callback for human-in-the-loop approval. callback(tool_name, args) -> bool"""
        self._approval_callback = callback

    def _log(self, tag: str, msg: str):
        self.log_lines.append(f"[{tag:<12}] {msg}")

    def _check_rate_limit(self, tool_name: str, max_calls: int = 10, window: float = 60.0) -> bool:
        now = time_mod.time()
        if tool_name not in self._rate_limit_tracker:
            self._rate_limit_tracker[tool_name] = []
        self._rate_limit_tracker[tool_name] = [
            t for t in self._rate_limit_tracker[tool_name] if now - t < window
        ]
        if len(self._rate_limit_tracker[tool_name]) >= max_calls:
            return False
        self._rate_limit_tracker[tool_name].append(now)
        return True

    def _llm_call(self, messages: list[dict], use_tools: bool = False,
                   call_type: str = "step") -> object:
        # Check budget before calling
        self.tokens.check_budget()

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
            # Budget warning at 80%
            if self.tokens.budget > 0 and self.tokens.budget_pct >= 80:
                warning = f"Token budget {self.tokens.budget_pct:.0f}% used ({self.tokens.total}/{self.tokens.budget})"
                self._log("BUDGET", f"⚠ {warning}")
                self.audit.log("budget_warning", {"message": warning, "pct": self.tokens.budget_pct})

        return response

    def _execute_tool_calls(self, response) -> list[tuple[str, str, str]]:
        results = []
        msg = response.choices[0].message
        if not msg.tool_calls:
            return results
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            # Rate limiting
            if not self._check_rate_limit(name):
                self._log("RATE_LIMIT", f"Tool '{name}' rate limited")
                results.append((name, "", "Error: rate limit exceeded"))
                continue

            # Human-in-the-loop approval for high-risk tools
            risk = self.tools.get_risk(name)
            if self.config.human_approval and risk in ("medium", "high"):
                self.stats["approvals_requested"] += 1
                self.audit.log("approval_requested", {"tool": name, "args": args, "risk": risk})
                if self._approval_callback:
                    approved = self._approval_callback(name, args)
                else:
                    approved = True  # Auto-approve if no callback set
                if approved:
                    self.stats["approvals_granted"] += 1
                    self._log("APPROVAL", f"✓ Approved: {name} (risk: {risk})")
                    self.audit.log("approval_requested", {"tool": name, "status": "approved"})
                else:
                    self._log("APPROVAL", f"✗ Denied: {name} (risk: {risk})")
                    self.audit.log("approval_requested", {"tool": name, "status": "denied"})
                    results.append((name, "", "Error: tool execution denied by human approval"))
                    continue

            result = self.tools.execute(name, args)
            args_str = ", ".join(f'"{v}"' for v in args.values())
            self._log("TOOL", f'{name}({args_str}) = {result[:120]}')
            self.stats["tool_calls"] += 1
            self.audit.log("tool_call", {"tool": name, "args": args, "result": result[:200]})

            # Run guardrails on tool output
            if self.config.guardrails_enabled:
                guardrail_results = self.verifier.run_guardrails(result)
                for check_name, (passed, reason) in guardrail_results:
                    self.stats["guardrail_checks"] += 1
                    if not passed:
                        self.stats["guardrail_blocks"] += 1
                        self._log("GUARDRAIL", f"✗ [{check_name}] {reason}")
                    self.audit.log("guardrail", {"check": check_name, "passed": passed, "reason": reason})

            results.append((name, args_str, result))
        return results

    def _verify(self, check_fn, *args) -> tuple[bool, str]:
        self.stats["verify_total"] += 1
        passed, reason = check_fn(*args)
        symbol = "✓" if passed else "✗"
        self._log("VERIFY", f"{symbol} {reason}")
        if passed:
            self.stats["verify_passes"] += 1
        self.audit.log("verification", {"passed": passed, "reason": reason})
        return passed, reason

    def _run_sub_agent(self, agent_name: str, task: str) -> str:
        """Delegate a task to a sub-agent."""
        if agent_name not in SUB_AGENT_DEFINITIONS:
            return f"Error: unknown sub-agent '{agent_name}'"
        agent = SUB_AGENT_DEFINITIONS[agent_name]
        self.stats["sub_agent_calls"] += 1
        self._log("SUB_AGENT", f"Delegating to '{agent_name}': {task[:60]}")
        self.audit.log("sub_agent", {"agent_name": agent_name, "task": task[:200]})
        result = agent.run(task)
        self._log("SUB_AGENT", f"'{agent_name}' returned: {result[:100]}")
        return result

    def _run_step_with_retry(self, task: str, step: str, step_num: int,
                              total_steps: int) -> str:
        retry_feedback = None
        step_result = ""
        max_attempts = 1 + (self.config.max_retries if self.config.retry_enabled else 0)

        for attempt in range(max_attempts):
            if attempt > 0:
                self.stats["retries"] += 1
                self._log("RETRY", f"Attempt {attempt + 1}/{max_attempts} for step {step_num}")
                self.audit.log("retry", {"step": step_num, "attempt": attempt + 1, "feedback": retry_feedback})

            try:
                self.tokens.check_budget()
            except TokenBudgetExceeded as e:
                self._log("BUDGET", f"✗ {e}")
                return f"Budget exceeded: {e}"

            guardrail_instructions = None
            if self.config.guardrails_enabled:
                guardrail_instructions = (
                    "- Do NOT include personal information (emails, phone numbers, SSNs, credit cards) in your response.\n"
                    "- Keep responses under 5000 characters.\n"
                    "- Do not generate harmful or dangerous content."
                )

            messages = self.context.build(
                task, self.memory, self.tools,
                step_instruction=step,
                retry_feedback=retry_feedback,
                guardrail_instructions=guardrail_instructions,
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
                    self._log("MEMORY", f"Stored: {key} = {step_result[:80]}")
                    self.audit.log("memory_store", {"key": key, "value": step_result[:200]})

                return step_result
            else:
                step_result = msg.content.strip() if msg.content else ""
                self._log("LLM", f'→ "{step_result[:120]}{"..." if len(step_result) > 120 else ""}"')

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
        """Run the harness. Returns (log, final_answer, memory_state, stats, tokens, audit_json)."""
        t0 = time_mod.time()
        self.log_lines = []

        self._log("CONFIG", self.config.status_line())
        self._log("TASK", f'"{task}"')
        self._log("TOOLS", f'{len(self.tools.names)} registered: {", ".join(self.tools.names)}')
        if self.config.token_budget > 0:
            self._log("BUDGET", f"Token budget: {self.config.token_budget}")
        self.audit.log("task_start", {"task": task, "config": self.config.status_line()})
        self._log("", "")

        # Planning
        if self.config.planner_enabled:
            self._log("PLANNER", "Decomposing task...")
            steps = self.planner.decompose(task)
            self._log("PLANNER", f"Decomposed into {len(steps)} steps")
            for i, step in enumerate(steps, 1):
                self._log("PLANNER", f"  {i}. {step}")
            self.audit.log("plan", {"steps": steps})
        else:
            steps = [task]
            self._log("PLANNER", "Disabled — running as single step")
        self._log("", "")

        # Execute each step
        final_answer = ""
        for i, step in enumerate(steps, 1):
            self._log(f"STEP {i}/{len(steps)}", step)
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
        stats_lines = [
            f"Steps completed:      {len(steps)}",
            f"Tool calls:           {self.stats['tool_calls']}",
            f"Memory entries:       {self.memory.count}",
            f"Verification:         {self.stats['verify_passes']}/{self.stats['verify_total']} passed",
            f"Retries:              {self.stats['retries']}",
            f"LLM calls:            {self.stats['llm_calls']}",
        ]
        if self.config.guardrails_enabled:
            stats_lines.append(f"Guardrail checks:     {self.stats['guardrail_checks']}")
            stats_lines.append(f"Guardrail blocks:     {self.stats['guardrail_blocks']}")
        if self.config.human_approval:
            stats_lines.append(f"Approvals requested:  {self.stats['approvals_requested']}")
            stats_lines.append(f"Approvals granted:    {self.stats['approvals_granted']}")
        if self.config.sub_agents_enabled:
            stats_lines.append(f"Sub-agent calls:      {self.stats['sub_agent_calls']}")
        stats_lines.append(f"Elapsed:              {elapsed:.1f}s")

        stats_output = "\n".join(stats_lines)
        token_output = self.tokens.summary()
        audit_output = self.audit.to_json()

        return log_output, final_answer, memory_output, stats_output, token_output, audit_output


# ---------------------------------------------------------------------------
# Comparison configs
# ---------------------------------------------------------------------------

COMPARISON_CONFIGS = [
    ("All ON", HarnessConfig(name="all_on", tools_enabled=True, memory_enabled=True, planner_enabled=True, verifier_enabled=True, retry_enabled=True)),
    ("No Planner", HarnessConfig(name="no_planner", tools_enabled=True, memory_enabled=True, planner_enabled=False, verifier_enabled=True, retry_enabled=True)),
    ("No Verifier", HarnessConfig(name="no_verifier", tools_enabled=True, memory_enabled=True, planner_enabled=True, verifier_enabled=False, retry_enabled=False)),
    ("No Memory", HarnessConfig(name="no_memory", tools_enabled=True, memory_enabled=False, planner_enabled=True, verifier_enabled=True, retry_enabled=True)),
    ("With Guardrails", HarnessConfig(name="guardrails", tools_enabled=True, memory_enabled=True, planner_enabled=True, verifier_enabled=True, retry_enabled=True, guardrails_enabled=True)),
    ("Budget Limited", HarnessConfig(name="budget", tools_enabled=True, memory_enabled=True, planner_enabled=True, verifier_enabled=True, retry_enabled=True, token_budget=2000)),
]
