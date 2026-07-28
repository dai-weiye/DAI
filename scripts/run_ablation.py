"""
Ablation + sensitivity, mostly REUSING cached reasoning traces (near-zero new cost):
  - patience sweep for the stabilization detector (1..5): pure re-analysis of cached traces
  - committer choice: strong vs weak (already in main records)
  - distractor-strength robustness: run a few distractor variants (new calls, capped)
  - efficiency: token/latency accounting from logs

Reads results/main_records.jsonl for the cached reasoning; re-derives commit decisions
offline where possible. Distractor-strength needs new reasoning calls -> capped subset.
"""
import sys, json, pathlib, argparse
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml
from llm_client import chat, get_spend
from datasets_load import load_gsm8k, load_math500, load_gpqa_diamond
from trace_method import analyze_trace, extract_final_answer, stabilization_prefix
from grade import grade

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/main.yaml"))


def reasoning_for(question, cond, distractor):
    """Retrieve (from cache) the reasoning pass used in main; identical params."""
    q = question + (distractor if cond == "adversarial" else "")
    prompt = q + "\nShow your reasoning, then end with 'Answer: <answer>'."
    r = chat([{"role": "user", "content": prompt}], model=CFG["models"]["reasoner"],
             temperature=CFG["sampling"]["greedy_temp"],
             max_tokens=CFG["sampling"]["max_tokens_reason"], sample_idx=0)
    return r


def patience_sweep(datasets, conditions, patiences=(1, 2, 3, 4, 5)):
    """For each patience, re-derive the stabilization commit value from the SAME cached
    trace and grade it. No new API calls (reasoning is cached; we grade the stable value
    directly, i.e. the 'oracle_commit' proxy at that patience)."""
    loaders = {"gsm8k": load_gsm8k, "math500": load_math500, "gpqa_diamond": load_gpqa_diamond}
    dist = CFG["adversarial"]["distractor"]
    rows = []
    for ds in datasets:
        n = CFG["datasets"].get(ds, 0)
        if n <= 0:
            continue
        items = loaders[ds](n)
        for cond in conditions:
            for pat in patiences:
                correct = total = 0
                for it in items:
                    r = reasoning_for(it["question"], cond, dist)
                    reasoning = r.reasoning or r.text
                    _, stable_val, _ = stabilization_prefix(reasoning, patience=pat)
                    correct += grade(stable_val, it["gold"], it.get("gold_type", "number"))
                    total += 1
                rows.append({"dataset": ds, "condition": cond, "patience": pat,
                             "acc_stable_commit": correct / total, "n": total})
                print(f"{ds}/{cond} patience={pat}: stable-commit acc={correct/total:.3f} (n={total})")
    return rows


def efficiency_report():
    """Token + latency accounting from the call log."""
    logs = [json.loads(l) for l in open(ROOT / "logs/calls.jsonl") if l.strip()]
    by_model = defaultdict(lambda: {"calls": 0, "pt": 0, "ct": 0, "rt": 0, "lat": 0.0, "cost": 0.0})
    for d in logs:
        m = by_model[d["model"]]
        m["calls"] += 1
        m["pt"] += d.get("pt", 0); m["ct"] += d.get("ct", 0); m["rt"] += d.get("rt", 0)
        m["lat"] += d.get("lat", 0.0); m["cost"] += d.get("cost_usd", 0.0)
    print("\n=== efficiency (from call logs) ===")
    for m, s in by_model.items():
        c = s["calls"]
        print(f"{m}: {c} calls | avg_rtok={s['rt']/c:.0f} avg_ctok={s['ct']/c:.0f} "
              f"avg_lat={s['lat']/c:.2f}s total_cost=${s['cost']:.4f}")
    return {m: dict(s) for m, s in by_model.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["gsm8k", "math500", "gpqa_diamond"])
    ap.add_argument("--conditions", nargs="*", default=["clean", "adversarial"])
    ap.add_argument("--out", default=str(ROOT / "results/ablation_summary.json"))
    args = ap.parse_args()
    out = {}
    out["patience_sweep"] = patience_sweep(args.datasets, args.conditions)
    out["efficiency"] = efficiency_report()
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}; new spend this run ${get_spend():.4f}")
