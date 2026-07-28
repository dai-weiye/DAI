"""
Pilot for Idea 3: can we (a) extract the intermediate answer trajectory from a
reasoning model's trace, (b) observe overthinking, (c) force early-commit via API?

Runs a few GSM8K items on v4-pro, incl. a distractor-injected variant (Gema-style),
and inspects the reasoning_content trace for answer oscillation.
Costs a handful of calls; cached so re-runs are free.
"""
import sys, re, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm_client import chat

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "gsm8k_test.jsonl"

def load_gsm8k(n):
    items = []
    with open(DATA) as f:
        for i, line in enumerate(f):
            if i >= n: break
            d = json.loads(line)
            gold = d["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"q": d["question"], "gold": gold})
    return items

NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")

def all_numbers(text):
    """Every number mentioned, in order, normalized."""
    out = []
    for m in NUM.finditer(text):
        s = m.group().replace("$", "").replace(",", "")
        try:
            out.append(str(int(float(s))) if float(s) == int(float(s)) else str(float(s)))
        except ValueError:
            pass
    return out

def answer_trajectory(reasoning, gold):
    """Split trace into sentences; for each, record the last number mentioned as the
    'current candidate answer'. Report where gold first appears and whether it's abandoned."""
    if not reasoning:
        return None
    sents = re.split(r'(?<=[.!?\n])\s+', reasoning)
    traj = []
    for s in sents:
        nums = all_numbers(s)
        if nums:
            traj.append(nums[-1])
    gold_norm = gold.replace(",", "")
    first_hit = next((i for i, v in enumerate(traj) if v == gold_norm), None)
    switches = sum(1 for i in range(1, len(traj)) if traj[i] != traj[i-1])
    # was gold reached before the end but the final candidate differs? -> abandoned
    abandoned = (first_hit is not None and traj and traj[-1] != gold_norm)
    return {"len": len(traj), "first_gold_pos": first_hit, "switches": switches,
            "final_candidate": traj[-1] if traj else None, "abandoned_gold": abandoned,
            "traj_tail": traj[-8:]}

def extract_final(text):
    nums = all_numbers(text)
    return nums[-1] if nums else None

def run():
    items = load_gsm8k(8)
    print("=== (b) Natural trace inspection on v4-pro ===")
    for it in items[:5]:
        r = chat([{"role":"user","content": it["q"] + "\nEnd your answer with 'Answer: <number>'."}],
                 model="deepseek-v4-pro", temperature=0.0, max_tokens=1500)
        final = extract_final(r.text)
        traj = answer_trajectory(r.reasoning, it["gold"])
        correct = final == it["gold"].replace(",", "")
        print(f"gold={it['gold']:>6} final={str(final):>6} {'OK' if correct else 'XX'} "
              f"rt={r.reasoning_tokens:>4} traj_switches={traj['switches'] if traj else '?'} "
              f"first_gold@{traj['first_gold_pos'] if traj else '?'} "
              f"abandoned={traj['abandoned_gold'] if traj else '?'}")

    print("\n=== (b') Distractor-injected (Gema-style) to induce overthinking ===")
    for it in items[:3]:
        distractor = (" (Note: a similar problem last week had a different setup, and "
                      "some people mistakenly double the final total, but consider carefully.)")
        q = it["q"] + distractor + "\nEnd your answer with 'Answer: <number>'."
        r = chat([{"role":"user","content": q}], model="deepseek-v4-pro",
                 temperature=0.0, max_tokens=2000, sample_idx=1)
        final = extract_final(r.text)
        traj = answer_trajectory(r.reasoning, it["gold"])
        correct = final == it["gold"].replace(",", "")
        print(f"gold={it['gold']:>6} final={str(final):>6} {'OK' if correct else 'XX'} "
              f"rt={r.reasoning_tokens:>4} switches={traj['switches'] if traj else '?'} "
              f"first_gold@{traj['first_gold_pos'] if traj else '?'} "
              f"abandoned={traj['abandoned_gold'] if traj else '?'} tail={traj['traj_tail'] if traj else '?'}")

    print("\n=== (c) Can we force early-commit? max_tokens truncation on reasoning model ===")
    it = items[0]
    for mt in [80, 200, 1500]:
        r = chat([{"role":"user","content": it["q"] + "\nEnd with 'Answer: <number>'."}],
                 model="deepseek-v4-pro", temperature=0.0, max_tokens=mt, sample_idx=2)
        print(f"max_tokens={mt:>4} finish={r.finish_reason:>10} rt={r.reasoning_tokens:>4} "
              f"ct={r.completion_tokens:>4} text={r.text[:50]!r}")

if __name__ == "__main__":
    run()
