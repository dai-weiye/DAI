"""
Comparison + ablation table for the non-termination paper.

Central honest claim: the "read confidence off the reasoning trace" paradigm — and
paid black-box baselines — all look predictive when truncated traces are included, and
all collapse toward chance once we restrict to FINISHED (valid-only) answers at the 8k
budget. This is the paper's key ablation: the confound (truncation), not the method,
was doing the work.

Two ablation axes, evaluated as error-detection AUROC of each uncertainty signal against
the SAME label (the 8k reasoner's own answer is wrong), on the SAME items:
  (A) budget:      2048  vs  8192
  (B) population:  all items  vs  valid-only (terminated)

Baselines (black-box, from prior runs, aligned by (id,condition)):
  - verbalized confidence (+1 call)     [kadavath2022know]
  - self-consistency agreement (+k calls)[wang2023selfconsistency]
Our gold-free trace signals: reasoning_tokens, n_steps, n_switches, switch_rate,
tail_entropy, n_distinct_tail.

NO API cost. Pure offline. Writes results/comparison_ablation.json + a LaTeX table.
"""
import json, pathlib
from collections import defaultdict
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg)


def auroc_ci(scores, labels, n=2000):
    s = np.asarray(scores, float); l = np.asarray(labels); m = len(s); vals = []
    for _ in range(n):
        idx = RNG.integers(0, m, m)
        a = auroc(s[idx], l[idx])
        if not np.isnan(a):
            vals.append(a)
    pt = auroc(s, l)
    if not vals:
        return pt, float("nan"), float("nan")
    vals.sort()
    return pt, vals[int(0.025*len(vals))], vals[int(0.975*len(vals))]


def zscore_within(recs, field):
    strata = defaultdict(list)
    for i, r in enumerate(recs):
        strata[(r["dataset"], r["condition"])].append(i)
    z = np.zeros(len(recs))
    for _, idxs in strata.items():
        v = np.array([float(recs[i].get(field) or 0) for i in idxs])
        mu, sd = v.mean(), v.std() + 1e-9
        for j, i in enumerate(idxs):
            z[i] = (v[j] - mu) / sd
    return z


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def nonterm(r, cap):
    return r.get("full_ans") in (None, "") and r.get("completion_tokens", 0) >= cap - 8


SIGNALS = [("reasoning_tokens", "Reasoning tokens"), ("n_steps", "Reasoning steps"),
           ("n_switches", "Answer switches"), ("switch_rate", "Switch rate"),
           ("tail_entropy", "Tail entropy"), ("n_distinct_tail", "Distinct tail")]


def eval_signals(recs, cap, valid_only):
    if valid_only:
        recs = [r for r in recs if not nonterm(r, cap)]
    lab = np.array([0 if r.get("correct_full") else 1 for r in recs])
    out = {}
    for field, name in SIGNALS:
        z = zscore_within(recs, field)
        out[name] = auroc(z, lab)
    return out, len(recs), float(lab.mean())


def eval_baseline(bl, unc_field, rec8k, cap, valid_only):
    """Evaluate a black-box baseline's uncertainty against the 8k model error label,
    on the SAME items, within-stratum z-scored. Higher uncertainty -> more likely error."""
    rows = []
    for r in bl:
        key = (r["id"], r["condition"])
        rr = rec8k.get(key)
        if rr is None:
            continue
        if valid_only and nonterm(rr, cap):
            continue
        rows.append({"dataset": r["dataset"], "condition": r["condition"],
                     "unc": float(r.get(unc_field, 0)),
                     "err": 0 if rr.get("correct_full") else 1})
    if not rows:
        return float("nan"), 0
    z = zscore_within(rows, "unc")
    lab = np.array([x["err"] for x in rows])
    return auroc(z, lab), len(rows)


def main():
    recs2k = load(ROOT / "results/main_records.jsonl")
    recs8k = load(ROOT / "results/main_records_8k.jsonl")
    rec8k_map = {(r["id"], r["condition"]): r for r in recs8k}
    vb = load(ROOT / "results/baseline_verbalized.jsonl")
    sc = load(ROOT / "results/baseline_selfconsistency.jsonl")

    out = {"ablation": {}}

    # Ablation grid: {2048, 8192} x {all, valid-only}
    print("=== ABLATION: error-detection AUROC of the effort signal ===")
    print(f"{'':30}{'all items':>12}{'valid-only':>12}")
    grid = {}
    for cap, recs, tag in [(2048, recs2k, "2048"), (8192, recs8k, "8192")]:
        all_sig, n_all, e_all = eval_signals(recs, cap, False)
        vo_sig, n_vo, e_vo = eval_signals(recs, cap, True)
        grid[tag] = {"all": all_sig, "valid": vo_sig,
                     "n_all": n_all, "n_valid": n_vo}
        rt_all = all_sig["Reasoning tokens"]; rt_vo = vo_sig["Reasoning tokens"]
        print(f"budget {tag:24}{rt_all:>12.3f}{rt_vo:>12.3f}   (n={n_all}/{n_vo})")
    out["ablation"] = grid

    # Full comparison table at 8192: all signals + baselines, all-items vs valid-only
    print("\n=== COMPARISON (8192 budget): all uncertainty signals ===")
    print(f"{'signal':26}{'all-items':>11}{'valid-only':>12}")
    comp = {}
    # our signals
    a_all, _, _ = eval_signals(recs8k, 8192, False)
    a_vo, nvo, _ = eval_signals(recs8k, 8192, True)
    for _, name in SIGNALS:
        comp[name] = {"all": a_all[name], "valid": a_vo[name], "type": "trace (free)"}
        print(f"{name:26}{a_all[name]:>11.3f}{a_vo[name]:>12.3f}")
    # baselines
    vb_all, nvb_all = eval_baseline(vb, "unc_verbalized", rec8k_map, 8192, False)
    vb_vo, nvb_vo = eval_baseline(vb, "unc_verbalized", rec8k_map, 8192, True)
    sc_all, nsc_all = eval_baseline(sc, "unc_sc", rec8k_map, 8192, False)
    sc_vo, nsc_vo = eval_baseline(sc, "unc_sc", rec8k_map, 8192, True)
    comp["Verbalized confidence (+1 call)"] = {"all": vb_all, "valid": vb_vo, "type": "baseline (paid)"}
    comp["Self-consistency (+k calls)"] = {"all": sc_all, "valid": sc_vo, "type": "baseline (paid)"}
    print(f"{'Verbalized conf. (+1 call)':26}{vb_all:>11.3f}{vb_vo:>12.3f}   (n={nvb_all}/{nvb_vo})")
    print(f"{'Self-consistency (+k)':26}{sc_all:>11.3f}{sc_vo:>12.3f}   (n={nsc_all}/{nsc_vo})")
    out["comparison_8192"] = comp
    out["comparison_n"] = {"trace_all": len(recs8k), "trace_valid": nvo,
                           "verbalized_all": nvb_all, "verbalized_valid": nvb_vo,
                           "sc_all": nsc_all, "sc_valid": nsc_vo}

    json.dump(out, open(ROOT / "results/comparison_ablation.json", "w"), indent=2)
    print(f"\nwrote {ROOT/'results/comparison_ablation.json'}")


if __name__ == "__main__":
    main()
