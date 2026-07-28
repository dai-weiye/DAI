"""
Truncation-aware re-analysis (addresses the round-2 rigor reviewer).

Splits errors into (a) TRUNCATED = model emitted no final answer (full_ans is null,
completion at cap) vs (b) GENUINE = a valid final answer that is wrong. Reports:
  - truncation rate per stratum
  - error-detection AUROC of reasoning_tokens on VALID-ANSWER-ONLY items (truncated removed)
  - AUROC controlling for truncation (partial), and on genuine-error vs correct only
so we can state honestly how much of the signal is truncation vs reasoning dynamics.

Works on any main_records file (pass path as argv[1]); no API cost.
"""
import sys, json, pathlib
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results/main_records.jsonl")
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
recs = [json.loads(l) for l in open(path) if l.strip()]


def is_truncated(r):
    # truncated if no final answer parsed AND completion hit (near) the cap
    return r.get("full_ans") in (None, "") and r.get("completion_tokens", 0) >= CAP - 8


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg)


def zwithin(recs_subset, field):
    strata = defaultdict(list)
    for i, r in enumerate(recs_subset):
        strata[(r["dataset"], r["condition"])].append(i)
    z = np.zeros(len(recs_subset))
    for _, idx in strata.items():
        v = np.array([float(recs_subset[i].get(field) or 0) for i in idx])
        z[idx] = (v - v.mean()) / (v.std() + 1e-9)
    return z


n = len(recs)
errs = [r for r in recs if not r.get("correct_full")]
trunc = [r for r in recs if is_truncated(r)]
trunc_err = [r for r in errs if is_truncated(r)]
print(f"file={pathlib.Path(path).name}  cap={CAP}")
print(f"total={n}  errors={len(errs)}  truncated={len(trunc)} ({len(trunc)/n:.0%})")
print(f"  of errors, truncated: {len(trunc_err)}/{len(errs)} = {len(trunc_err)/max(len(errs),1):.0%}")

print("\ntruncation rate by stratum:")
strata = defaultdict(lambda: [0, 0])
for r in recs:
    s = strata[(r["dataset"], r["condition"])]
    s[0] += is_truncated(r); s[1] += 1
for k in sorted(strata):
    print(f"  {k[0]:12}/{k[1]:12} {strata[k][0]}/{strata[k][1]} = {strata[k][0]/strata[k][1]:.0%}")

# --- error detection with vs without truncated items ---
lab_all = [0 if r.get("correct_full") else 1 for r in recs]
z_all = zwithin(recs, "reasoning_tokens")
print(f"\n[all items] reasoning_tokens error AUROC (within-stratum) = {auroc(z_all, lab_all):.3f}")

valid = [r for r in recs if not is_truncated(r)]
lab_v = [0 if r.get("correct_full") else 1 for r in valid]
z_v = zwithin(valid, "reasoning_tokens")
print(f"[valid-answer only, n={len(valid)}] reasoning_tokens error AUROC = {auroc(z_v, lab_v):.3f}")
print(f"  (valid-only error rate = {np.mean(lab_v):.2f})")

# per-stratum on valid only
print("\n[valid-only] per-stratum reasoning_tokens AUROC:")
vs = defaultdict(list)
for r in valid:
    vs[(r["dataset"], r["condition"])].append(r)
for k in sorted(vs):
    rr = vs[k]
    a = auroc([x["reasoning_tokens"] for x in rr], [0 if x.get("correct_full") else 1 for x in rr])
    ne = sum(0 if x.get("correct_full") else 1 for x in rr)
    print(f"  {k[0]:12}/{k[1]:12} AUROC={a:.3f}  (n={len(rr)}, err={ne})")

# switches signal on valid-only (does the *dynamics* signal survive?)
z_sw = zwithin(valid, "n_switches")
print(f"\n[valid-only] n_switches error AUROC = {auroc(z_sw, lab_v):.3f}")
