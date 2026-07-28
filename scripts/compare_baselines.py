"""
Compare the reasoning-EFFORT confidence signal against two black-box UQ baselines:
  - verbalized confidence (model states 0-100)
  - self-consistency agreement (fraction agreeing with modal vote)
on the SAME items, for error detection (AUROC) and risk-coverage.

Merges baseline_*.jsonl with main_records.jsonl on (id, condition). No API cost.
"""
import sys, json, pathlib
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name):
    p = ROOT / "results" / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


main = {(r["id"], r["condition"]): r
        for r in (json.loads(l) for l in open(ROOT / "results/main_records.jsonl"))}
vb = {(r["id"], r["condition"]): r for r in load("baseline_verbalized.jsonl")}
sc = {(r["id"], r["condition"]): r for r in load("baseline_selfconsistency.jsonl")}


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg)


def auroc_ci(scores, labels, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, float); labels = np.asarray(labels); n = len(scores)
    vals = [auroc(scores[i], labels[i]) for i in (rng.integers(0, n, n) for _ in range(nb))]
    vals = [v for v in vals if not np.isnan(v)]; vals.sort()
    return auroc(scores, labels), vals[int(.025*len(vals))], vals[int(.975*len(vals))]


# Build a table on the intersection where ALL signals exist (fair comparison).
def compare(keys, tag):
    rows = []
    for k in keys:
        if k not in main or k not in vb:
            continue
        m = main[k]
        err = 0 if m.get("correct_full") else 1
        row = {"err": err, "reasoning_tokens": m["reasoning_tokens"],
               "unc_verbalized": vb[k]["unc_verbalized"]}
        if k in sc:
            row["unc_sc"] = sc[k]["unc_sc"]
        rows.append(row)
    if not rows:
        print(f"[{tag}] no overlap"); return
    labels = [r["err"] for r in rows]
    print(f"\n[{tag}] n={len(rows)} (err rate={np.mean(labels):.2f})")
    for sig in ["reasoning_tokens", "unc_verbalized", "unc_sc"]:
        have = [r for r in rows if sig in r]
        if len(have) < 5 or len(set(r["err"] for r in have)) < 2:
            print(f"  {sig:18} n/a"); continue
        a, lo, hi = auroc_ci([r[sig] for r in have], [r["err"] for r in have])
        name = {"reasoning_tokens": "reasoning tokens (ours)",
                "unc_verbalized": "verbalized confidence",
                "unc_sc": "self-consistency (5x)"}[sig]
        print(f"  {name:26} AUROC={a:.3f}  95%CI[{lo:.3f},{hi:.3f}]  (n={len(have)})")


# self-consistency only run on numeric datasets; report two comparisons
keys_num = [k for k in main if k[0].split("-")[0] in ("gsm8k", "math500")]
keys_all = list(main.keys())
compare(keys_all, "ALL datasets (effort vs verbalized)")
compare(keys_num, "GSM8K+MATH (effort vs verbalized vs self-consistency)")

# complementarity: does combining effort + verbalized beat either alone? (simple avg of ranks)
def rank(x):
    x = np.asarray(x, float); return np.argsort(np.argsort(x)).astype(float)/len(x)
rows = [{"err": (0 if main[k].get("correct_full") else 1),
         "rt": main[k]["reasoning_tokens"], "vb": vb[k]["unc_verbalized"]}
        for k in main if k in vb]
labels = [r["err"] for r in rows]
comb = rank([r["rt"] for r in rows]) + rank([r["vb"] for r in rows])
print(f"\n[complementarity, ALL] effort+verbalized combined AUROC="
      f"{auroc(comb, labels):.3f} vs effort {auroc([r['rt'] for r in rows], labels):.3f} "
      f"vs verbalized {auroc([r['vb'] for r in rows], labels):.3f}")

json.dump({"note": "baseline comparison"}, open(ROOT / "results/baseline_compare.json", "w"))
