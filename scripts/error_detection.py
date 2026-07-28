"""
Can GOLD-FREE trace signals predict per-item error (selective prediction / abstention)?
Pure offline analysis of cached records. No API cost.

We test whether switch_rate, n_distinct_tail, tail_entropy, reasoning_tokens predict
whether the model's OWN final answer is wrong -- WITHIN each (dataset, condition)
stratum (so we don't just recover 'hard datasets are hard').

Metrics: AUROC of each signal for error detection; risk-coverage curve for abstention
(if we abstain on the most-uncertain items, how fast does accuracy on the rest rise?).
"""
import sys, json, pathlib
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from grade import grade

ROOT = pathlib.Path(__file__).resolve().parents[1]
recs = [json.loads(l) for l in open(ROOT / "results/main_records.jsonl") if l.strip()]
for r in recs:
    r["err"] = 0 if r.get("correct_full") else 1

SIGNALS = ["switch_rate", "n_switches", "n_distinct_tail", "tail_entropy",
           "reasoning_tokens", "n_steps"]


def auroc(scores, labels):
    """AUROC that `score` ranks positives (errors) above negatives. Rank-based (Mann-Whitney)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = np.array(labels) == 1
    n_pos = pos.sum(); n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


print("=== AUROC of gold-free signals for ERROR detection (pooled within-stratum via z-score) ===")
# z-score each signal within its (dataset,condition) stratum, then pool
strata = defaultdict(list)
for r in recs:
    strata[(r["dataset"], r["condition"])].append(r)
for r in recs:
    r["_z"] = {}
for sig in SIGNALS:
    for k, rs in strata.items():
        vals = np.array([float(r.get(sig) or 0) for r in rs])
        mu, sd = vals.mean(), vals.std() + 1e-9
        for r in rs:
            r["_z"][sig] = (float(r.get(sig) or 0) - mu) / sd
    z = np.array([r["_z"][sig] for r in recs])
    labels = np.array([r["err"] for r in recs])
    print(f"  {sig:18} within-stratum AUROC = {auroc(z, labels):.3f}")

# also raw pooled (includes difficulty signal), for contrast
print("\n(for contrast) raw pooled AUROC (confounded by dataset difficulty):")
for sig in SIGNALS:
    s = np.array([float(r.get(sig) or 0) for r in recs])
    print(f"  {sig:18} pooled AUROC = {auroc(s, np.array([r['err'] for r in recs])):.3f}")

# Combined simple confidence = z(switch_rate)+z(tail_entropy)+z(reasoning_tokens)
comb = np.array([r["_z"]["switch_rate"] + r["_z"]["tail_entropy"] + r["_z"]["reasoning_tokens"]
                 for r in recs])
labels = np.array([r["err"] for r in recs])
print(f"\ncombined (switch_rate+tail_entropy+reasoning_tokens, within-stratum z) AUROC = "
      f"{auroc(comb, labels):.3f}")

# Risk-coverage: abstain on most-uncertain fraction, report accuracy on the rest
print("\n=== Risk-coverage (abstain by combined uncertainty, pooled) ===")
idx = np.argsort(comb)  # ascending uncertainty; keep the LOW-uncertainty ones
correct = np.array([1 - r["err"] for r in recs])
print(f"  {'coverage':>9} {'selective_acc':>14} {'full_acc':>9}")
base = correct.mean()
for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    keep = idx[:int(cov * len(recs))]
    print(f"  {cov:>9.0%} {correct[keep].mean():>14.3f} {base:>9.3f}")

# abandonment predictive recap
ab = [r for r in recs if r.get("reached_then_abandoned")]
print(f"\n[recap] reach-then-abandon (gold-based) error rate = "
      f"{np.mean([r['err'] for r in ab]):.3f} (n={len(ab)})")
