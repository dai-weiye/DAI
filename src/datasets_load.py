"""
Dataset loaders + gold-answer normalization. No fabrication: loads real files from data/.
"""
from __future__ import annotations
import json, re, pathlib

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"


def _norm_num(s: str):
    s = s.strip().replace("$", "").replace(",", "").replace("%", "")
    s = s.rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def load_gsm8k(n: int | None = None, path: str = "gsm8k_test.jsonl"):
    items = []
    with open(DATA / path) as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            d = json.loads(line)
            gold = d["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"id": f"gsm8k-{i}", "dataset": "gsm8k",
                          "question": d["question"], "gold": _norm_num(gold) or gold,
                          "gold_type": "number"})
    return items


def _boxed(s: str):
    """Extract the last \\boxed{...} content, brace-balanced."""
    key = r"\boxed{"
    idx = s.rfind(key)
    if idx < 0:
        return None
    i = idx + len(key)
    depth = 1
    out = []
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


def load_math500(n: int | None = None, path: str = "math500_test.jsonl"):
    p = DATA / path
    if not p.exists():
        return []
    items = []
    with open(p) as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            d = json.loads(line)
            ans = d.get("answer")
            if ans is None:
                sol = d.get("solution", "")
                ans = _boxed(sol)
            items.append({"id": f"math500-{i}", "dataset": "math500",
                          "question": d.get("problem") or d.get("question"),
                          "gold": (ans or "").strip(),
                          "gold_type": "expr",
                          "subject": d.get("subject") or d.get("type"),
                          "level": d.get("level")})
    return items


def load_gpqa_diamond(n: int | None = None, path: str = "gpqa_diamond.csv", seed: int = 0):
    """GPQA-Diamond MCQ (198 items, CC BY 4.0). Deterministically shuffle the 4 options
    per item using a fixed seed so the gold letter is reproducible but not always 'A'.
    Only loads if the file exists (repo/data provenance already checked)."""
    import csv, random
    p = DATA / path
    if not p.exists():
        return []
    items = []
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(seed)
    for i, d in enumerate(rows):
        if n is not None and i >= n:
            break
        correct = (d.get("Correct Answer") or "").strip()
        incorrect = [(d.get(f"Incorrect Answer {j}") or "").strip() for j in (1, 2, 3)]
        opts = [correct] + [x for x in incorrect if x]
        if len(opts) < 2:
            continue
        order = list(range(len(opts)))
        rng.shuffle(order)
        shuffled = [opts[k] for k in order]
        gold_idx = shuffled.index(correct)
        letters = ["A", "B", "C", "D"][: len(shuffled)]
        gold_letter = letters[gold_idx]
        q = d.get("Question", "").strip()
        opt_text = "\n".join(f"({l}) {t}" for l, t in zip(letters, shuffled))
        question = (f"{q}\n\n{opt_text}\n\n"
                    "Choose the single best option and end with 'Answer: <letter>'.")
        items.append({"id": f"gpqa-{i}", "dataset": "gpqa_diamond",
                      "question": question, "gold": gold_letter, "gold_type": "mcq",
                      "subject": d.get("High-level domain") or d.get("Subdomain")})
    return items


DATASETS = {"gsm8k": load_gsm8k, "math500": load_math500, "gpqa_diamond": load_gpqa_diamond}
