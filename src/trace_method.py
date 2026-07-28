"""
Core method: reasoning-trace analysis + trace-anchored early-commit.

Two contributions live here:
1. TRACE SIGNALS: from a reasoning_content string, extract the per-step candidate-answer
   trajectory and derive overthinking signals (oscillation, first-stable position,
   reach-then-abandon).
2. EARLY-COMMIT: detect answer stabilization and commit via a cheap second call,
   avoiding the "keep thinking and abandon the right answer" failure mode.

Answer extraction is itself validated (see scripts/validate_extractor.py) so we can
report its accuracy — reviewers will (rightly) ask.
"""
from __future__ import annotations
import re, math
from dataclasses import dataclass

# ---------- answer extraction ----------
_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_ANSWER_TAG = re.compile(r"(?:answer|=)\s*[:=]?\s*\$?(-?\d[\d,]*\.?\d*)", re.I)
# For multiple-choice (GPQA): 'Answer: C', 'answer is (B)', 'final answer: D'
_MCQ_TAG = re.compile(r"answer\s*(?:is)?\s*[:=]?\s*\(?([A-D])\)?", re.I)


def extract_mcq_letter(text: str):
    """Extract an A-D choice letter from a model answer, if present."""
    if not text:
        return None
    m = _MCQ_TAG.findall(text)
    if m:
        return m[-1].upper()
    # bare trailing letter
    m2 = re.findall(r"\b([A-D])\b", text)
    return m2[-1].upper() if m2 else None


def norm_num(s: str):
    if s is None:
        return None
    s = str(s).strip().replace("$", "").replace(",", "").replace("%", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(round(f, 6))
    except ValueError:
        return None


def numbers_in(text: str):
    out = []
    for m in _NUM.finditer(text or ""):
        v = norm_num(m.group())
        if v is not None:
            out.append(v)
    return out


def extract_final_answer(text: str):
    """Prefer an explicit 'Answer: X' tag; else last number in the text.
    Handles numeric (GSM8K), and falls back to the raw tag payload for symbolic
    (MATH) answers so we don't drop valid non-numeric commits."""
    if not text:
        return None
    # 1) explicit numeric tag
    tags = _ANSWER_TAG.findall(text)
    if tags:
        return norm_num(tags[-1])
    # 2) general 'Answer: <payload>' (symbolic / boxed / letter)
    m = re.search(r"answer\s*[:=]\s*(.+?)\s*$", text.strip(), re.I | re.S)
    if m:
        payload = m.group(1).strip().strip(".").strip()
        payload = payload.split("\n")[0].strip()
        n = norm_num(payload)
        if n is not None:
            return n
        if payload:
            return payload
    # 3) boxed
    b = _boxed_inline(text)
    if b is not None:
        return norm_num(b) or b
    # 4) last number
    nums = numbers_in(text)
    return nums[-1] if nums else None


def _boxed_inline(s: str):
    key = r"\boxed{"
    idx = s.rfind(key)
    if idx < 0:
        return None
    i = idx + len(key)
    depth, out = 1, []
    while i < len(s) and depth > 0:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()


def split_steps(reasoning: str):
    """Split a reasoning trace into reasoning 'steps' (sentence/newline units)."""
    if not reasoning:
        return []
    parts = re.split(r'(?<=[.!?])\s+|\n+', reasoning)
    return [p.strip() for p in parts if p.strip()]


# ---------- trace signals ----------
@dataclass
class TraceSignals:
    n_steps: int
    trajectory: list          # candidate answer after each step that mentions a number
    n_switches: int           # times the running candidate changed
    switch_rate: float        # switches / len(trajectory)
    first_stable_idx: int | None   # step index where the *final-candidate* value first appears and holds
    final_candidate: str | None
    reached_then_abandoned: bool   # a value appeared, then the final candidate differs
    tail_entropy: float       # entropy of candidate values over the last window (instability)
    n_distinct_tail: int


def _entropy(vals):
    if not vals:
        return 0.0
    from collections import Counter
    c = Counter(vals)
    tot = len(vals)
    return -sum((k / tot) * math.log(k / tot, 2) for k in c.values())


def analyze_trace(reasoning: str, gold: str | None = None,
                  tail_window: int = 6) -> TraceSignals:
    steps = split_steps(reasoning)
    traj = []
    for s in steps:
        nums = numbers_in(s)
        if nums:
            traj.append(nums[-1])   # last number in a step = current working answer
    n_sw = sum(1 for i in range(1, len(traj)) if traj[i] != traj[i - 1])
    final_cand = traj[-1] if traj else None
    # first index where the FINAL candidate value first appears (i.e., when it stabilized)
    first_stable = None
    if final_cand is not None:
        for i, v in enumerate(traj):
            if v == final_cand:
                first_stable = i
                break
    # reach-then-abandon: gold (if known) appeared earlier but final differs;
    # gold-agnostic version: some value repeated >=2 then got replaced by a different final
    reached_then_abandoned = False
    if gold is not None and final_cand is not None:
        g = norm_num(gold) or gold
        if any(v == g for v in traj) and final_cand != g:
            reached_then_abandoned = True
    tail = traj[-tail_window:]
    return TraceSignals(
        n_steps=len(steps), trajectory=traj, n_switches=n_sw,
        switch_rate=(n_sw / len(traj)) if traj else 0.0,
        first_stable_idx=first_stable, final_candidate=final_cand,
        reached_then_abandoned=reached_then_abandoned,
        tail_entropy=_entropy(tail), n_distinct_tail=len(set(tail)),
    )


# ---------- early-commit mechanism ----------
def stabilization_prefix(reasoning: str, patience: int = 3):
    """Find the earliest step after which the running candidate answer stays constant
    for `patience` consecutive numeric steps. Return (prefix_text, committed_value, idx).
    If never stable, return (full reasoning, final candidate, None)."""
    steps = split_steps(reasoning)
    running = None
    stable_count = 0
    stable_val = None
    numeric_step_idx = -1
    for i, s in enumerate(steps):
        nums = numbers_in(s)
        if not nums:
            continue
        numeric_step_idx += 1
        v = nums[-1]
        if v == running:
            stable_count += 1
        else:
            running = v
            stable_count = 1
        if stable_count >= patience and stable_val is None:
            stable_val = v
            prefix = " ".join(steps[: i + 1])
            return prefix, v, i
    # no stabilization -> whole trace
    traj = [numbers_in(s)[-1] for s in steps if numbers_in(s)]
    return reasoning, (traj[-1] if traj else None), None
