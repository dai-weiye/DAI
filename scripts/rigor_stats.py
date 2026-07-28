"""
Rigorous statistics for the non-termination paper (offline, $0).

Every headline claim gets a hypothesis test + effect size + 95% CI, so reviewers
see the results are not just point estimates. Specifically:

  (1) Distractor -> non-termination: 2x2 (clean/adv) x (term/non-term) per dataset and
      pooled. Fisher's exact test, odds ratio + 95% CI (Wald on log-OR), risk ratio.
      Holm-Bonferroni correction across the per-dataset family.
  (2) Non-termination rate 95% CIs: Wilson score intervals (better than normal approx
      for proportions near 0/1, e.g. gsm8k-clean = 0%).
  (3) 3.3x distractor multiplier: bootstrap 95% CI on the ratio adv_rate/clean_rate.
  (4) Budget sensitivity 2048->8192: McNemar-style / two-proportion test per stratum
      with CI on the rate difference.
  (5) Effort AUROC = 0.46: bootstrap 95% CI AND a label-permutation test (is it
      distinguishable from 0.5? we WANT non-significance -> confirms chance).
  (6) Valid-only accuracy clean vs adv (0.72 vs 0.73): two-proportion test (we WANT
      non-significance -> distractor doesn't hurt finished answers).
  (7) Mechanism gaps: already bootstrapped in non_termination_deep; here add a
      Mann-Whitney U test (distribution-level, not just mean gap).

Writes results/rigor_stats.json + console tables. No API calls.
"""
import json, pathlib
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
RNG = np.random.default_rng(0)
CAP = 8192


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def nonterm(r, cap=CAP):
    return r.get("full_ans") in (None, "") and r.get("completion_tokens", 0) >= cap - 8


# ---------- proportion CIs ----------

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0, c-hw), min(1, c+hw)


def two_prop_test(k1, n1, k2, n2):
    """Two-sided z-test for difference of proportions + CI on the difference."""
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    se0 = np.sqrt(p*(1-p)*(1/n1+1/n2))
    z = (p1-p2)/se0 if se0 > 0 else 0.0
    pval = 2*(1-stats.norm.cdf(abs(z)))
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    diff = p1-p2
    return diff, diff-1.96*se, diff+1.96*se, pval


def odds_ratio_ci(a, b, c, d):
    """2x2 table [[a,b],[c,d]]; OR with Wald 95% CI on log-OR (Haldane correction)."""
    a_, b_, c_, d_ = a+0.5, b+0.5, c+0.5, d+0.5
    or_ = (a_*d_)/(b_*c_)
    se = np.sqrt(1/a_+1/b_+1/c_+1/d_)
    lo = np.exp(np.log(or_)-1.96*se); hi = np.exp(np.log(or_)+1.96*se)
    return or_, lo, hi


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    idx = np.argsort(pvals)
    m = len(pvals); adj = np.empty(m)
    prev = 0
    for rank, i in enumerate(idx):
        adj[i] = min(1.0, max(prev, (m-rank)*pvals[i]))
        prev = adj[i]
    return adj


def auroc(scores, labels):
    s = np.asarray(scores, float); l = np.asarray(labels)
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    pos = l == 1; np_, nn = pos.sum(), (~pos).sum()
    if np_ == 0 or nn == 0:
        return float("nan")
    return (ranks[pos].sum()-np_*(np_+1)/2)/(np_*nn)


def zscore_within(recs, field):
    strata = defaultdict(list)
    for i, r in enumerate(recs):
        strata[(r["dataset"], r["condition"])].append(i)
    z = np.zeros(len(recs))
    for _, idxs in strata.items():
        v = np.array([float(recs[i].get(field) or 0) for i in idxs])
        mu, sd = v.mean(), v.std()+1e-9
        for j, i in enumerate(idxs):
            z[i] = (v[j]-mu)/sd
    return z


def main():
    recs = load(ROOT/"results/main_records_8k.jsonl")
    recs2k = load(ROOT/"results/main_records.jsonl")
    out = {}

    DS = ["gsm8k", "math500", "gpqa_diamond"]

    # (1)+(2) distractor -> non-termination, per dataset: Fisher, OR, Wilson CIs
    print("=== (1) Distractor -> non-termination (per dataset + pooled) ===")
    print(f"{'dataset':14}{'clean rate [95% CI]':>26}{'adv rate [95% CI]':>26}{'OR [95% CI]':>20}{'Fisher p':>11}")
    per = {}; pvals = []; keys = []
    for ds in DS + ["POOLED"]:
        sub = recs if ds == "POOLED" else [r for r in recs if r["dataset"] == ds]
        cl = [r for r in sub if r["condition"] == "clean"]
        ad = [r for r in sub if r["condition"] == "adversarial"]
        a = sum(nonterm(r) for r in ad); b = len(ad)-a           # adv: nonterm, term
        c = sum(nonterm(r) for r in cl); d = len(cl)-c           # clean: nonterm, term
        pc, pcl, pch = wilson(c, len(cl))
        pa, pal, pah = wilson(a, len(ad))
        or_, olo, ohi = odds_ratio_ci(a, b, c, d)
        _, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
        per[ds] = {"clean": [pc, pcl, pch], "adv": [pa, pal, pah],
                   "or": [or_, olo, ohi], "fisher_p": fisher_p,
                   "n_clean": len(cl), "n_adv": len(ad)}
        if ds != "POOLED":
            pvals.append(fisher_p); keys.append(ds)
        print(f"{ds:14}{pc:5.0%} [{pcl:.0%},{pch:.0%}]{'':>8}{pa:5.0%} [{pal:.0%},{pah:.0%}]{'':>8}"
              f"{or_:5.1f} [{olo:.1f},{ohi:.1f}]{fisher_p:>11.2g}")
    adj = holm(pvals)
    for k, p, pa_ in zip(keys, pvals, adj):
        per[k]["fisher_p_holm"] = float(pa_)
    print(f"  Holm-adjusted per-dataset p: " + ", ".join(f"{k}={pa_:.2g}" for k, pa_ in zip(keys, adj)))
    out["distractor_nonterm"] = per

    # (3) 3.3x multiplier bootstrap CI (pooled)
    print("\n=== (3) Distractor multiplier (adv_rate / clean_rate), bootstrap 95% CI ===")
    cl = np.array([nonterm(r) for r in recs if r["condition"] == "clean"], float)
    ad = np.array([nonterm(r) for r in recs if r["condition"] == "adversarial"], float)
    mults = []
    for _ in range(5000):
        cs = cl[RNG.integers(0, len(cl), len(cl))].mean()
        as_ = ad[RNG.integers(0, len(ad), len(ad))].mean()
        if cs > 0:
            mults.append(as_/cs)
    mults = np.array(mults); point = ad.mean()/cl.mean()
    lo, hi = np.percentile(mults, [2.5, 97.5])
    out["multiplier"] = {"point": point, "ci": [float(lo), float(hi)]}
    print(f"  multiplier = {point:.1f}x  95% CI [{lo:.1f}, {hi:.1f}]  (clean {cl.mean():.0%}, adv {ad.mean():.0%})")

    # (4) budget sensitivity per stratum (adversarial): 2048 vs 8192 two-prop test
    print("\n=== (4) Budget sensitivity 2048->8192 (adversarial), two-proportion test ===")
    bud = {}
    for ds in DS:
        a2 = [r for r in recs2k if r["dataset"] == ds and r["condition"] == "adversarial"]
        a8 = [r for r in recs if r["dataset"] == ds and r["condition"] == "adversarial"]
        k2 = sum(nonterm(r, 2048) for r in a2); k8 = sum(nonterm(r, 8192) for r in a8)
        diff, dlo, dhi, p = two_prop_test(k2, len(a2), k8, len(a8))
        bud[ds] = {"rate_2k": k2/len(a2), "rate_8k": k8/len(a8),
                   "diff": diff, "ci": [dlo, dhi], "p": p}
        print(f"  {ds:14} {k2/len(a2):4.0%} -> {k8/len(a8):4.0%}  Δ={diff:+.0%} [{dlo:+.0%},{dhi:+.0%}]  p={p:.2g}"
              f"  {'(still >0: persists)' if k8/len(a8) > 0.15 else ''}")
    out["budget_sensitivity"] = bud

    # (5) effort AUROC = chance? bootstrap CI + permutation test (want NON-significant)
    print("\n=== (5) Effort AUROC (valid-only): is it distinguishable from 0.5? ===")
    valid = [r for r in recs if not nonterm(r)]
    z = zscore_within(valid, "reasoning_tokens")
    lab = np.array([0 if r.get("correct_full") else 1 for r in valid])
    a0 = auroc(z, lab)
    boots = []
    for _ in range(5000):
        idx = RNG.integers(0, len(z), len(z))
        v = auroc(z[idx], lab[idx])
        if not np.isnan(v):
            boots.append(v)
    blo, bhi = np.percentile(boots, [2.5, 97.5])
    # permutation: shuffle labels, how often |AUROC-0.5| >= observed?
    obs = abs(a0-0.5); ge = 0; NP = 5000
    for _ in range(NP):
        pl = RNG.permutation(lab)
        if abs(auroc(z, pl)-0.5) >= obs:
            ge += 1
    perm_p = (ge+1)/(NP+1)
    out["effort_auroc"] = {"auroc": a0, "ci": [float(blo), float(bhi)],
                           "perm_p_vs_chance": perm_p, "n": len(valid)}
    print(f"  AUROC={a0:.3f}  95% CI [{blo:.3f}, {bhi:.3f}]  (CI includes 0.5: {blo <= 0.5 <= bhi})")
    print(f"  permutation p vs 0.5 = {perm_p:.2g}  ({'NOT distinguishable from chance -> confirms artifact' if perm_p > 0.05 else 'distinguishable'})")

    # (6) valid-only accuracy clean vs adv (want NON-significant)
    print("\n=== (6) Valid-only accuracy: clean vs adversarial (want n.s.) ===")
    vc = [r for r in valid if r["condition"] == "clean"]
    va = [r for r in valid if r["condition"] == "adversarial"]
    kc = sum(bool(r.get("correct_full")) for r in vc); ka = sum(bool(r.get("correct_full")) for r in va)
    diff, dlo, dhi, p = two_prop_test(kc, len(vc), ka, len(va))
    out["valid_acc_clean_vs_adv"] = {"clean": kc/len(vc), "adv": ka/len(va),
                                     "diff": diff, "ci": [dlo, dhi], "p": p}
    print(f"  clean {kc/len(vc):.3f} vs adv {ka/len(va):.3f}  Δ={diff:+.3f} [{dlo:+.3f},{dhi:+.3f}]  p={p:.2g}"
          f"  ({'n.s.: distractor does NOT hurt finished answers' if p > 0.05 else 'sig'})")

    # (7) mechanism: Mann-Whitney U on the raw distributions (needs traces -> reuse deep script's approach)
    print("\n=== (7) Mechanism signals: Mann-Whitney U (distribution-level) ===")
    try:
        import non_termination_deep as nt
        items = nt._build_items(json.load(open(ROOT/"configs/main_8k.yaml")) if False else __import__("yaml").safe_load(open(ROOT/"configs/main_8k.yaml")))
        distr = __import__("yaml").safe_load(open(ROOT/"configs/main_8k.yaml"))["adversarial"]["distractor"]
        feats = {"doubt_density": ([], []), "loop_ratio": ([], [])}
        for r in recs:
            text, _, hit = nt._trace_for(r, items, distr, CAP)
            if not hit:
                continue
            isnt = nonterm(r)
            for f, fn in [("doubt_density", nt.doubt_density), ("loop_ratio", nt.loop_ratio)]:
                feats[f][0 if isnt else 1].append(fn(text))
        mech = {}
        for f, (ntv, tmv) in feats.items():
            U, p = stats.mannwhitneyu(ntv, tmv, alternative="greater")
            # rank-biserial effect size
            rbc = 1 - 2*U/(len(ntv)*len(tmv))
            mech[f] = {"U": float(U), "p": float(p), "rank_biserial": float(-rbc),
                       "n_nt": len(ntv), "n_tm": len(tmv)}
            print(f"  {f:16} Mann-Whitney U p={p:.2g}  rank-biserial={-rbc:+.2f}  (nt>{'term' })")
        out["mechanism_mwu"] = mech
    except Exception as e:
        print(f"  [skipped mechanism MWU: {e}]")

    json.dump(out, open(ROOT/"results/rigor_stats.json", "w"), indent=2)
    print(f"\nwrote {ROOT/'results/rigor_stats.json'}")


if __name__ == "__main__":
    main()
