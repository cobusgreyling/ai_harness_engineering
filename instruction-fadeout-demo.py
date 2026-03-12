"""
Instruction Fade-Out Demo
Demonstrates how LLMs progressively forget system prompt instructions
over long conversations, and how event-driven reminders fix it.
"""

import os
import json
import re
from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "private/nvidia/nemotron-3-super-120b-a12b"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

CLIENT = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
    default_headers={"NVCF-POLL-SECONDS": "1800"},
)

SYSTEM_PROMPT = (
    "You are a research assistant. CRITICAL FORMAT RULE: Every response MUST be "
    "valid JSON with exactly three keys: 'answer' (your response text), "
    "'confidence' (a float 0.0-1.0), and 'sources' (a list of strings). "
    "No markdown, no plain text, no explanations outside the JSON structure. "
    "Only output the JSON object, nothing else."
)

SYSTEM_REMINDER = (
    "[SYSTEM REMINDER] You MUST respond in valid JSON format with keys: "
    "'answer', 'confidence', 'sources'. No other format is acceptable."
)

# Diverse questions designed to tempt the model away from JSON format
QUESTIONS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a Python function to reverse a string.",
    "What are the pros and cons of microservices?",
    "Tell me a brief story about a robot learning to paint.",
    "Compare TCP and UDP protocols.",
    "What caused the 2008 financial crisis?",
    "How does a neural network learn?",
    "What is the difference between REST and GraphQL?",
    "Explain quantum entanglement in simple terms.",
    "What are the best practices for API security?",
    "How does garbage collection work in Java?",
    "What is the trolley problem in ethics?",
    "Describe the architecture of a modern web browser.",
    "What is the significance of the Turing test?",
    "How do transformers work in NLP?",
    "What are design patterns in software engineering?",
    "Explain the CAP theorem.",
    "What is the difference between AI and AGI?",
    "How does blockchain consensus work?",
]


def is_valid_json_response(text: str) -> tuple[bool, str]:
    """Check if response follows the required JSON format."""
    text = text.strip()

    # Try to extract JSON from the response
    # Sometimes models wrap JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False, "NOT JSON"

    required_keys = {"answer", "confidence", "sources"}
    actual_keys = set(data.keys())

    if not required_keys.issubset(actual_keys):
        missing = required_keys - actual_keys
        return False, f"MISSING KEYS: {missing}"

    if not isinstance(data.get("sources"), list):
        return False, "SOURCES NOT A LIST"

    try:
        conf = float(data.get("confidence", -1))
        if not (0.0 <= conf <= 1.0):
            return False, "CONFIDENCE OUT OF RANGE"
    except (TypeError, ValueError):
        return False, "CONFIDENCE NOT NUMERIC"

    return True, "VALID"


def run_conversation(use_reminders: bool = False, reminder_interval: int = 3):
    """Run a multi-turn conversation and track format compliance."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []

    mode = "WITH REMINDERS" if use_reminders else "WITHOUT REMINDERS"
    print(f"\n{'='*70}")
    print(f"  INSTRUCTION FADE-OUT TEST — {mode}")
    print(f"{'='*70}\n")

    for i, question in enumerate(QUESTIONS):
        turn = i + 1

        # Inject system reminder before the user message
        if use_reminders and turn > 1 and (turn - 1) % reminder_interval == 0:
            messages.append({
                "role": "system",
                "content": SYSTEM_REMINDER,
            })
            print(f"  [REMINDER INJECTED at turn {turn}]")

        messages.append({"role": "user", "content": question})

        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )

        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})

        valid, reason = is_valid_json_response(reply)
        status = "PASS" if valid else "FAIL"

        results.append({
            "turn": turn,
            "question": question[:50],
            "valid": valid,
            "reason": reason,
        })

        # Truncated preview of response
        preview = reply[:80].replace("\n", " ")
        print(f"  Turn {turn:>2} [{status}] {reason:<20} | {question[:45]}")
        if not valid:
            print(f"           Response: {preview}...")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["valid"])
    failed = total - passed

    print(f"\n{'─'*70}")
    print(f"  RESULTS: {passed}/{total} compliant ({passed/total*100:.0f}%)")
    print(f"  Format violations: {failed}")

    if failed > 0:
        first_fail = next((r for r in results if not r["valid"]), None)
        if first_fail:
            print(f"  First violation at: Turn {first_fail['turn']}")

    # Show compliance by segment
    segments = [
        ("Turns 1-5", results[0:5]),
        ("Turns 6-10", results[5:10]),
        ("Turns 11-15", results[10:15]),
        ("Turns 16-20", results[15:20]),
    ]

    print(f"\n  Compliance by segment:")
    for label, segment in segments:
        seg_pass = sum(1 for r in segment if r["valid"])
        bar = "█" * seg_pass + "░" * (len(segment) - seg_pass)
        print(f"    {label}: {bar} {seg_pass}/{len(segment)}")

    print()
    return results


def main():
    if not NVIDIA_API_KEY:
        print("Error: set NVIDIA_API_KEY environment variable")
        return

    print("\n" + "█" * 70)
    print("  INSTRUCTION FADE-OUT EXPERIMENT")
    print("  Model: NVIDIA Nemotron 3 Super (120B/12B active)")
    print("  Task: Maintain JSON format compliance over 20 turns")
    print("█" * 70)

    # Run 1: No reminders
    results_no_remind = run_conversation(use_reminders=False)

    # Run 2: With reminders every 3 turns
    results_with_remind = run_conversation(use_reminders=True, reminder_interval=3)

    # Comparison
    no_remind_pass = sum(1 for r in results_no_remind if r["valid"])
    with_remind_pass = sum(1 for r in results_with_remind if r["valid"])

    print("=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  Without reminders: {no_remind_pass}/20 compliant ({no_remind_pass/20*100:.0f}%)")
    print(f"  With reminders:    {with_remind_pass}/20 compliant ({with_remind_pass/20*100:.0f}%)")
    improvement = with_remind_pass - no_remind_pass
    if improvement > 0:
        print(f"  Improvement:       +{improvement} turns recovered")
    print()

    # Turn-by-turn comparison table
    print("  Turn-by-turn comparison:")
    print(f"  {'Turn':<6} {'No Reminder':<15} {'With Reminder':<15}")
    print(f"  {'─'*6} {'─'*15} {'─'*15}")
    for nr, wr in zip(results_no_remind, results_with_remind):
        nr_status = "PASS" if nr["valid"] else "FAIL"
        wr_status = "PASS" if wr["valid"] else "FAIL"
        recovered = " <-- recovered" if not nr["valid"] and wr["valid"] else ""
        print(f"  {nr['turn']:<6} {nr_status:<15} {wr_status:<15}{recovered}")
    print()


if __name__ == "__main__":
    main()
