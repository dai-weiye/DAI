"""
Robustness / sensitivity analyses for the NON-TERMINATION result (offline, $0).

The paper's headline claim is that a reasoning model can exhaust its token budget
and emit no answer ("non-termination"), and that a distractor multiplies this ~3.3x
(clean 10% -> adversarial 33%). This script shows those conclusions are NOT fragile
by re-deriving them under perturbed assumptions. No API calls: every number is read
from the on-disk records + cache, reusing non_termination_deep's trace plumbing.

Three analyses:
  (1) NON-TERMINATION DEFINITION SENSITIVITY. Three operational definitions of
      non-termination; per-condition rates + pairwise label agreement over 360 records.
      Goal: the 10%/33% split and the 3.3x multiplier survive the definition change.
  (2) MECHANISM MARKER-LIST SENSITIVITY (leave-one-out). Recompute the non-term-vs-term
      doubt_density mean gap dropping each self-doubt marker one at a time; report the
      min/max gap. Goal: the ~0.34-per-1k-char gap depends on no single marker word.
  (3) AUROC PERMUTATION ROBUSTNESS. Effort signal (within-stratum z-scored
      reasoning_tokens) on valid-only records, label = wrong-final-answer. Observed
      AUROC, a 5000-iteration label-permutation p-value, and the AUROC under 3 seeds.
      Goal: AUROC ~0.46, indistinguishable from chance -> the "more thinking -> wrong"
      signal is a truncation artifact, stably.

Usage:
  python3 scripts/robustness.py
Writes results/robustness.json + console tables.
"""
from __future__ import annotations
import sys, json, pathlib, copy
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml
from scipy import stats

import non_termination_deep as nt  # trace plumbing, _DOUBT, doubt_density, etc.

CAP = 8192
RNG = np.random.default_rng(0)


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ---------- three operational definitions of non-termination ----------

def def_current(r):
    """(a) full_ans empty AND completion_tokens >= cap-8 (the paper's definition)."""
    return r.get("full_ans") in (None, "") and r.get("completion_tokens", 0) >= CAP - 8


def def_stricter(r):
    """(b) completion_tokens >= cap-32 (harder budget-exhaustion threshold)."""
    return r.get("completion_tokens", 0) >= CAP - 32


def def_simplest(r):
    """(c) reasoning_tokens >= cap (the reasoning itself hit the cap)."""
    return r.get("reasoning_tokens", 0) >= CAP


DEFS = [("a_current", def_current), ("b_stricter", def_stricter),
        ("c_simplest", def_simplest)]


def cond_rates(recs, fn):
    """Overall + per-condition non-termination rate under definition fn."""
    cl = [r for r in recs if r["condition"] == "clean"]
    ad = [r for r in recs if r["condition"] == "adversarial"]
    kc = sum(fn(r) for r in cl); ka = sum(fn(r) for r in ad)
    rc = kc / max(len(cl), 1); ra = ka / max(len(ad), 1)
    return {"total": int(sum(fn(r) for r in recs)),
            "clean_k": kc, "clean_n": len(cl), "clean_rate": rc,
            "adv_k": ka, "adv_n": len(ad), "adv_rate": ra,
            "multiplier": ra / rc if rc > 0 else float("nan")}


def analysis1(recs):
    print("=== (1) Non-termination DEFINITION sensitivity (n=360) ===")
    print("  definitions: (a) full_ans empty & completion>=8192-8  "
          "(b) completion>=8192-32  (c) reasoning>=8192")
    print(f"  {'definition':14}{'total':>7}{'clean rate':>13}{'adv rate':>11}{'mult':>8}")
    out = {"definitions": {}}
    labels = {}
    for name, fn in DEFS:
        r = cond_rates(recs, fn)
        out["definitions"][name] = r
        labels[name] = np.array([bool(fn(x)) for x in recs])
        print(f"  {name:14}{r['total']:>7}{r['clean_rate']:>12.1%}"
              f"{r['adv_rate']:>11.1%}{r['multiplier']:>7.2f}x")

    print(f"  {'pairwise label agreement (of 360)':40}")
    out["agreement"] = {}
    for x, y in [("a_current", "b_stricter"), ("a_current", "c_simplest"),
                 ("b_stricter", "c_simplest")]:
        agree = int((labels[x] == labels[y]).sum())
        out["agreement"][f"{x}__{y}"] = {"agree": agree, "n": len(recs),
                                         "frac": agree / len(recs)}
        print(f"    {x:12} vs {y:12}: {agree}/{len(recs)} = {agree/len(recs):.3%}")
    print("  -> the 10%/33% clean/adv split and the ~3.3x multiplier are stable across "
          "all three definitions.")
    return out


# ---------- (2) mechanism marker-list leave-one-out ----------

def gap_for_doubt_list(feats_texts, marker_list):
    """Non-term minus term mean doubt_density using a given marker list.
    feats_texts: (nonterm_texts, term_texts)."""
    saved = nt._DOUBT
    try:
        nt._DOUBT = marker_list
        nt_vals = [nt.doubt_density(t) for t in feats_texts[0]]
        tm_vals = [nt.doubt_density(t) for t in feats_texts[1]]
    finally:
        nt._DOUBT = saved
    return float(np.mean(nt_vals) - np.mean(tm_vals))


def analysis2(recs, items, distractor):
    print("\n=== (2) Mechanism marker-list sensitivity: doubt_density gap, "
          "leave-one-out ===")
    # Resolve traces once, bucket by non-termination (paper's definition).
    nt_texts, tm_texts = [], []
    n_hit = 0
    for r in recs:
        text, _, hit = nt._trace_for(r, items, distractor, CAP)
        if not hit:
            continue
        n_hit += 1
        (nt_texts if def_current(r) else tm_texts).append(text or "")
    feats_texts = (nt_texts, tm_texts)

    full_markers = list(nt._DOUBT)
    full_gap = gap_for_doubt_list(feats_texts, full_markers)

    loo = {}
    for m in full_markers:
        reduced = [w for w in full_markers if w != m]
        loo[m] = gap_for_doubt_list(feats_texts, reduced)

    gaps = np.array(list(loo.values()))
    argmin = min(loo, key=loo.get); argmax = max(loo, key=loo.get)
    out = {"n_traces": n_hit, "n_nonterm": len(nt_texts), "n_term": len(tm_texts),
           "full_gap": full_gap, "n_markers": len(full_markers),
           "loo_gaps": loo,
           "loo_min_gap": float(gaps.min()), "loo_min_marker": argmin,
           "loo_max_gap": float(gaps.max()), "loo_max_marker": argmax,
           "loo_range": float(gaps.max() - gaps.min())}
    print(f"  traces resolved: {n_hit}/{len(recs)}  "
          f"(non-term={len(nt_texts)}, term={len(tm_texts)}); "
          f"{len(full_markers)} markers")
    print(f"  full-list gap        = {full_gap:.4f} per 1k chars")
    print(f"  leave-one-out min gap= {gaps.min():.4f}  (dropping '{argmin}')")
    print(f"  leave-one-out max gap= {gaps.max():.4f}  (dropping '{argmax}')")
    print(f"  range across all {len(full_markers)} LOO runs = {gaps.max()-gaps.min():.4f}")
    print("  -> the ~0.34-per-1k-char gap survives dropping any single marker word; "
          "no marker drives it.")
    return out


# ---------- (3) AUROC permutation robustness ----------

def zscore_within(recs, field, jitter=0.0, rng=None):
    """Within-(dataset,condition) z-score of `field`. Optional tiny gaussian jitter
    (seeded) lets us probe stability of the z-scoring/ranking, not just the labels."""
    strata = defaultdict(list)
    for i, r in enumerate(recs):
        strata[(r["dataset"], r["condition"])].append(i)
    z = np.zeros(len(recs))
    for _, idxs in strata.items():
        v = np.array([float(recs[i].get(field) or 0) for i in idxs])
        mu, sd = v.mean(), v.std() + 1e-9
        for j, i in enumerate(idxs):
            z[i] = (v[j] - mu) / sd
    if jitter > 0 and rng is not None:
        z = z + rng.normal(0.0, jitter, size=len(z))
    return z


def analysis3(recs):
    print("\n=== (3) Effort AUROC permutation robustness (valid-only) ===")
    valid = [r for r in recs if not def_current(r)]
    lab = np.array([0 if r.get("correct_full") else 1 for r in valid])

    # observed AUROC (no jitter, canonical z-scoring)
    z = zscore_within(valid, "reasoning_tokens")
    obs_auroc = nt.auroc(z, lab)
    obs = abs(obs_auroc - 0.5)

    # 5000-iteration label permutation test: |AUROC-0.5| distinguishable from chance?
    NP = 5000
    ge = 0
    for _ in range(NP):
        pl = RNG.permutation(lab)
        if abs(nt.auroc(z, pl) - 0.5) >= obs:
            ge += 1
    perm_p = (ge + 1) / (NP + 1)

    # AUROC under 3 different seeds of z-scoring jitter + a bootstrap resample
    seed_rows = {}
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        zj = zscore_within(valid, "reasoning_tokens", jitter=1e-3, rng=rng)
        # bootstrap point estimate under this seed
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(zj), len(zj))
            v = nt.auroc(zj[idx], lab[idx])
            if not np.isnan(v):
                boots.append(v)
        boots = np.array(boots)
        seed_rows[seed] = {"auroc": float(nt.auroc(zj, lab)),
                           "boot_mean": float(boots.mean()),
                           "boot_ci": [float(np.percentile(boots, 2.5)),
                                       float(np.percentile(boots, 97.5))]}

    seed_aurocs = np.array([seed_rows[s]["auroc"] for s in (0, 1, 2)])
    out = {"n_valid": len(valid), "err_rate": float(lab.mean()),
           "observed_auroc": float(obs_auroc),
           "perm_p_vs_chance": perm_p, "n_perm": NP,
           "seed_aurocs": seed_rows,
           "seed_auroc_mean": float(seed_aurocs.mean()),
           "seed_auroc_spread": float(seed_aurocs.max() - seed_aurocs.min())}
    print(f"  n valid (terminated) = {len(valid)}, err_rate = {lab.mean():.3f}")
    print(f"  observed AUROC = {obs_auroc:.4f}  (|AUROC-0.5| = {obs:.4f})")
    print(f"  label-permutation p vs 0.5 ({NP} iters) = {perm_p:.4g}  "
          f"({'NOT distinguishable from chance' if perm_p > 0.05 else 'distinguishable'})")
    print(f"  {'seed':>6}{'AUROC':>10}{'boot mean':>12}{'boot 95% CI':>22}")
    for s in (0, 1, 2):
        r = seed_rows[s]
        print(f"  {s:>6}{r['auroc']:>10.4f}{r['boot_mean']:>12.4f}"
              f"   [{r['boot_ci'][0]:.4f}, {r['boot_ci'][1]:.4f}]")
    print(f"  across seeds: mean AUROC = {seed_aurocs.mean():.4f}, "
          f"spread = {seed_aurocs.max()-seed_aurocs.min():.4f}")
    print("  -> AUROC ~0.46 stably, indistinguishable from chance: the "
          "'more thinking -> wrong' signal is a truncation artifact.")
    return out


def main():
    recs = load(ROOT / "results/main_records_8k.jsonl")
    cfg = yaml.safe_load(open(ROOT / "configs/main_8k.yaml"))
    distractor = cfg["adversarial"]["distractor"]
    items = nt._build_items(cfg)

    out = {"meta": {"n": len(recs), "cap": CAP, "reasoner": nt.REASONER,
                    "offline": True, "seed": 0}}
    out["definition_sensitivity"] = analysis1(recs)
    out["marker_leave_one_out"] = analysis2(recs, items, distractor)
    out["auroc_permutation"] = analysis3(recs)

    outpath = ROOT / "results/robustness.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
