"""
Difficulty-control analysis (reviewer W: is reasoning_tokens just a difficulty proxy?).

Independent, MODEL-FREE difficulty measures per item:
  - gsm8k: number of reasoning steps in the GOLD solution (count of '<<...>>' calc annotations
           or newlines) + gold-solution token length.
  - math500: dataset 'level' (1-5) + gold solution length.
  - gpqa: question length (no gold solution reasoning available) -- weaker; report separately.

Tests:
  (1) Partial Spearman correlation of reasoning_tokens with error, CONTROLLING for the
      independent difficulty measure. If it stays significant, effort adds signal beyond difficulty.
  (2) Within difficulty-bin AUROC: bin items by independent difficulty, compute
      reasoning_tokens error-detection AUROC WITHIN each bin. If >0.5 within bins, not just a proxy.
  (3) Does reasoning_tokens beat the difficulty measure at error detection? (AUROC comparison)

Pure offline (uses cached main_records + re-loaded gold solutions). No API cost.
"""
import sys, json, pathlib, re
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
recs = [json.loads(l) for l in open(ROOT / "results/main_records.jsonl") if l.strip()]
for r in recs:
    r["err"] = 0 if r.get("correct_full") else 1


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else 0.0


def partial_spearman(x, y, z):
    """Spearman(x,y) controlling for z: correlation of residuals after regressing rank(x),rank(y) on rank(z)."""
    def rank(v):
        v = np.asarray(v, float); return np.argsort(np.argsort(v)).astype(float)
    rx, ry, rz = rank(x), rank(y), rank(z)
    def resid(a, b):
        b1 = np.vstack([np.ones_like(b), b]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    d = np.sqrt((ex**2).sum()*(ey**2).sum())
    return float((ex*ey).sum()/d) if d > 0 else 0.0


# --- build independent difficulty per item ---
def gsm8k_difficulty():
    diffs = {}
    with open(DATA / "gsm8k_test.jsonl") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            sol = d["answer"]
            steps = sol.count("<<")             # number of calculator annotations
            toklen = len(sol.split())
            diffs[f"gsm8k-{i}"] = {"steps": steps, "gold_len": toklen}
    return diffs


def math_difficulty():
    diffs = {}
    with open(DATA / "math500_test.jsonl") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            lvl = d.get("level", 0)
            try:
                lvl = int(re.findall(r"\d+", str(lvl))[0])
            except Exception:
                lvl = 0
            sol = d.get("solution", "")
            diffs[f"math500-{i}"] = {"level": lvl, "gold_len": len(sol.split())}
    return diffs


gsm_d = gsm8k_difficulty(); math_d = math_difficulty()
for r in recs:
    if r["dataset"] == "gsm8k":
        r["difficulty"] = gsm_d.get(r["id"], {}).get("gold_len", 0)
        r["diff_steps"] = gsm_d.get(r["id"], {}).get("steps", 0)
    elif r["dataset"] == "math500":
        r["difficulty"] = math_d.get(r["id"], {}).get("gold_len", 0)
        r["diff_level"] = math_d.get(r["id"], {}).get("level", 0)
    else:
        r["difficulty"] = len(r.get("question", "").split())  # weak proxy for gpqa


print("=== Difficulty-control analysis: is reasoning_tokens just a difficulty proxy? ===\n")

# Per dataset (numeric-solution datasets have the cleanest independent difficulty)
for ds in ["gsm8k", "math500"]:
    rs = [r for r in recs if r["dataset"] == ds]
    rtok = [r["reasoning_tokens"] for r in rs]
    err = [r["err"] for r in rs]
    diff = [r["difficulty"] for r in rs]
    print(f"[{ds}] n={len(rs)}")
    print(f"  Spearman(reasoning_tokens, error)                 = {spearman(rtok, err):+.3f}")
    print(f"  Spearman(gold_solution_len, error) [difficulty]   = {spearman(diff, err):+.3f}")
    print(f"  PARTIAL Spearman(rtok, error | difficulty)        = {partial_spearman(rtok, err, diff):+.3f}")
    print(f"  AUROC reasoning_tokens for error                  = {auroc(rtok, err):.3f}")
    print(f"  AUROC gold_solution_len (difficulty) for error    = {auroc(diff, err):.3f}")
    # within difficulty-bin AUROC (terciles)
    q = np.quantile(diff, [1/3, 2/3])
    bins = defaultdict(list)
    for r in rs:
        b = 0 if r["difficulty"] <= q[0] else (1 if r["difficulty"] <= q[1] else 2)
        bins[b].append(r)
    print("  within-difficulty-tercile AUROC(reasoning_tokens):")
    for b in sorted(bins):
        rb = bins[b]
        a = auroc([x["reasoning_tokens"] for x in rb], [x["err"] for x in rb])
        print(f"    tercile {b}: AUROC={a:.3f} (n={len(rb)}, err={sum(x['err'] for x in rb)})")
    print()

# MATH by level (cleanest independent difficulty)
rs = [r for r in recs if r["dataset"] == "math500" and r.get("diff_level")]
print("[math500] reasoning_tokens error-AUROC WITHIN each MATH level (independent difficulty):")
bylvl = defaultdict(list)
for r in rs:
    bylvl[r["diff_level"]].append(r)
for lvl in sorted(bylvl):
    rb = bylvl[lvl]
    a = auroc([x["reasoning_tokens"] for x in rb], [x["err"] for x in rb])
    print(f"  level {lvl}: AUROC={a:.3f} (n={len(rb)}, err={sum(x['err'] for x in rb)})")

out = {"note": "difficulty control", "generated": True}
json.dump(out, open(ROOT / "results/difficulty_control.json", "w"), indent=2)
