"""
Rigorous selective-prediction / error-detection study over cached records.
NO API cost (pure offline analysis of results/main_records.jsonl).

Methodology (avoids the exploration-phase pitfalls):
  - Signals are GOLD-FREE (computable at inference): reasoning_tokens, n_switches,
    n_steps, switch_rate, tail_entropy, n_distinct_tail, and a logistic combo.
  - Error label = model's OWN final answer is wrong (correct_full == False).
  - AUROC reported (a) within-stratum via z-scored pooling (controls dataset difficulty)
    and (b) per dataset/condition. Bootstrap 95% CIs on AUROC.
  - Selective prediction with PROPER train/test: fit combiner + choose threshold on a
    train split, evaluate risk-coverage on a disjoint test split. Seeded, k-fold.
  - Cross-dataset generalization: train combiner on 2 datasets, test on the held-out one.
  - Cost-aware ROUTING: abstain on high-uncertainty items and 'route' them (diagnostic:
    measure how many routed + accuracy retained), the deployable use-case.

Outputs results/selective_summary.json + console tables.
"""
import sys, json, pathlib, random
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Optional args: [records_path] [--valid-only] [cap]
_args = [a for a in sys.argv[1:]]
_valid_only = "--valid-only" in _args
_args = [a for a in _args if a != "--valid-only"]
_path = _args[0] if _args else str(ROOT / "results/main_records.jsonl")
_cap = int(_args[1]) if len(_args) > 1 else 8192
recs = [json.loads(l) for l in open(_path) if l.strip()]
if _valid_only:
    def _trunc(r):
        return r.get("full_ans") in (None, "") and r.get("completion_tokens", 0) >= _cap - 8
    recs = [r for r in recs if not _trunc(r)]
print(f"[selective_prediction] file={pathlib.Path(_path).name} valid_only={_valid_only} n={len(recs)}")
for r in recs:
    r["err"] = 0 if r.get("correct_full") else 1

SIGNALS = ["reasoning_tokens", "n_switches", "n_steps", "switch_rate",
           "tail_entropy", "n_distinct_tail"]
RNG = np.random.default_rng(0)


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg)


def auroc_ci(scores, labels, n_boot=2000):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    n = len(scores); vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        a = auroc(scores[idx], labels[idx])
        if not np.isnan(a):
            vals.append(a)
    point = auroc(scores, labels)
    if not vals:
        # degenerate stratum (e.g. 0 errors): AUROC undefined, no CI to report.
        return point, float("nan"), float("nan")
    vals.sort()
    return point, vals[int(0.025*len(vals))], vals[int(0.975*len(vals))]


def zscore_within_strata(field):
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


def _sigmoid(z):
    # np.errstate: numpy on macOS Accelerate BLAS raises spurious over/underflow +
    # divide-by-zero flags on small matmuls even though outputs are finite & correct
    # (verified: no inf/nan in inputs or results). Suppress the known false positive.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, iters=500, lr=0.1, l2=1e-2):
    """Tiny logistic regression (no sklearn dependency risk); standardized inputs.
    Mild L2 keeps weights bounded so degenerate/near-constant strata can't blow up
    the matmul (numerical honesty for the reported combiner AUROCs)."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    # guard against pathological z-scores from near-zero-variance strata
    Xb = np.clip(Xb, -10, 10)
    w = np.zeros(Xb.shape[1])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(iters):
            p = _sigmoid(Xb @ w)
            grad = (Xb.T @ (p - y)) / len(y)
            grad[1:] += l2 * w[1:]  # regularize weights but not the bias
            w -= lr * grad
    return w


def predict_logreg(w, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    Xb = np.clip(Xb, -10, 10)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _sigmoid(Xb @ w)


def main():
    out = {}
    labels = np.array([r["err"] for r in recs])

    # (1) within-stratum AUROC + CI per signal
    print("=== (1) Within-stratum error-detection AUROC (gold-free signals) ===")
    out["within_stratum_auroc"] = {}
    for s in SIGNALS:
        z = zscore_within_strata(s)
        a, lo, hi = auroc_ci(z, labels)
        out["within_stratum_auroc"][s] = {"auroc": a, "ci": [lo, hi]}
        print(f"  {s:18} AUROC={a:.3f}  95%CI[{lo:.3f},{hi:.3f}]")

    # (2) per dataset/condition AUROC for the best single signal (reasoning_tokens),
    #     WITH bootstrap CIs and effective positive (error) counts (W3 fix: expose that
    #     GSM8K-clean rests on very few errors).
    print("\n=== (2) Per-stratum AUROC (reasoning_tokens) with CIs and error counts ===")
    out["per_stratum_auroc_rtok"] = {}
    strata = defaultdict(list)
    for i, r in enumerate(recs):
        strata[(r["dataset"], r["condition"])].append(i)
    for k, idxs in sorted(strata.items()):
        sc = np.array([recs[i]["reasoning_tokens"] for i in idxs])
        lab = labels[np.array(idxs)]
        n_err = int(lab.sum())
        a, lo, hi = auroc_ci(sc, lab)
        out["per_stratum_auroc_rtok"][f"{k[0]}/{k[1]}"] = {
            "auroc": a, "ci": [lo, hi], "n_err": n_err, "n": len(idxs)}
        flag = "  <-- n_err<=2: unreliable" if n_err <= 2 or (len(idxs)-n_err) <= 2 else ""
        print(f"  {k[0]:12}/{k[1]:12} AUROC={a:.3f} CI[{lo:.2f},{hi:.2f}] "
              f"n_err={n_err}/{len(idxs)}{flag}")

    # (3) selective prediction with proper 5-fold CV (combiner fit on train, eval on test)
    #     LEAKAGE-SAFE: within-stratum standardization uses TRAIN-FOLD statistics only,
    #     then applies them to the held-out fold. (Previously z-scored on full data ->
    #     mild leakage; fixed here so held-out numbers are honest.)
    print("\n=== (3) Selective prediction: 5-fold CV (leakage-safe standardization) ===")
    raw = {s: np.array([float(r.get(s) or 0) for r in recs]) for s in SIGNALS}
    stratum_id = np.array([f"{r['dataset']}/{r['condition']}" for r in recs])
    n = len(recs); idx = np.arange(n); RNG.shuffle(idx)
    folds = np.array_split(idx, 5)
    cov_grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    sel_acc = {c: [] for c in cov_grid}
    test_aurocs = []

    def zscore_fit_apply(tr, te):
        """Compute per-signal, per-stratum mean/std on TRAIN only; apply to both."""
        Ztr = np.zeros((len(tr), len(SIGNALS))); Zte = np.zeros((len(te), len(SIGNALS)))
        for j, s in enumerate(SIGNALS):
            for st in np.unique(stratum_id):
                m_tr = (stratum_id[tr] == st); m_te = (stratum_id[te] == st)
                if m_tr.sum() == 0:
                    mu, sd = raw[s][tr].mean(), raw[s][tr].std() + 1e-9  # fallback: global train
                else:
                    mu, sd = raw[s][tr][m_tr].mean(), raw[s][tr][m_tr].std() + 1e-9
                Ztr[m_tr, j] = (raw[s][tr][m_tr] - mu) / sd
                if m_te.sum():
                    Zte[m_te, j] = (raw[s][te][m_te] - mu) / sd
        return Ztr, Zte

    for f in range(5):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(5) if j != f])
        Ztr, Zte = zscore_fit_apply(tr, te)
        w = fit_logreg(Ztr, labels[tr])
        conf_te = predict_logreg(w, Zte)  # higher = more likely ERROR
        test_aurocs.append(auroc(conf_te, labels[te]))
        order = np.argsort(conf_te)  # ascending error-prob: keep most-confident
        correct_te = 1 - labels[te]
        for c in cov_grid:
            keep = order[:int(c*len(te))]
            sel_acc[c].append(correct_te[keep].mean())
    out["cv_test_auroc"] = {"mean": float(np.mean(test_aurocs)),
                            "folds": [float(x) for x in test_aurocs]}
    out["selective_risk_coverage_cv"] = {}
    print(f"  held-out combiner AUROC (mean over folds) = {np.mean(test_aurocs):.3f}")
    print(f"  {'coverage':>9} {'sel_acc(test)':>14}")
    base = (1 - labels).mean()
    for c in cov_grid:
        m = float(np.mean(sel_acc[c])); sd = float(np.std(sel_acc[c]))
        out["selective_risk_coverage_cv"][f"{c:.1f}"] = {"acc": m, "std": sd}
        print(f"  {c:>9.0%} {m:>14.3f}  (±{sd:.3f})   base={base:.3f}")

    # (4) cross-dataset generalization: train on 2 datasets, test on held-out dataset.
    #     Leakage-safe: standardize each signal using TRAIN-dataset per-condition stats,
    #     apply to held-out. Also report an EFFORT-ONLY combiner (tokens, steps) that is
    #     valid on GPQA where trajectory signals are inert (W4 fix).
    print("\n=== (4) Cross-dataset generalization (leakage-safe; full vs effort-only) ===")
    out["cross_dataset_auroc"] = {}
    EFFORT = ["reasoning_tokens", "n_steps"]
    dsets = ["gsm8k", "math500", "gpqa_diamond"]

    def zfit_apply_by(sig_list, tr, te):
        Ztr = np.zeros((len(tr), len(sig_list))); Zte = np.zeros((len(te), len(sig_list)))
        for j, s in enumerate(sig_list):
            for st in np.unique(stratum_id):
                mtr = stratum_id[tr] == st; mte = stratum_id[te] == st
                ref = raw[s][tr][mtr] if mtr.sum() else raw[s][tr]
                mu, sd = ref.mean(), ref.std() + 1e-9
                Ztr[mtr, j] = (raw[s][tr][mtr] - mu) / sd
                if mte.sum():
                    Zte[mte, j] = (raw[s][te][mte] - mu) / sd
        return Ztr, Zte

    for held in dsets:
        tr = np.array([i for i, r in enumerate(recs) if r["dataset"] != held])
        te = np.array([i for i, r in enumerate(recs) if r["dataset"] == held])
        row = {}
        for name, sigs in [("full", SIGNALS), ("effort_only", EFFORT)]:
            Ztr, Zte = zfit_apply_by(sigs, tr, te)
            w = fit_logreg(Ztr, labels[tr])
            a = auroc(predict_logreg(w, Zte), labels[te])
            row[name] = a
        out["cross_dataset_auroc"][held] = row
        print(f"  test={held:14} full={row['full']:.3f}  effort_only={row['effort_only']:.3f}  (n={len(te)})")

    # (5) routing diagnostic: error rate concentrated in abstained set (uses full-data
    #     combiner for display only; the honest generalization numbers are the CV ones above).
    print("\n=== (5) Routing value: error rate in abstained (routed) vs kept set ===")
    Zfull, _ = zfit_apply_by(SIGNALS, np.arange(n), np.arange(0))
    w = fit_logreg(Zfull, labels)
    conf = predict_logreg(w, Zfull)
    order = np.argsort(conf)
    out["routing"] = {}
    for c in [0.9, 0.8, 0.7, 0.6, 0.5]:
        keep = order[:int(c*n)]; routed = order[int(c*n):]
        kept_err = labels[keep].mean(); routed_err = labels[routed].mean()
        out["routing"][f"{c:.1f}"] = {"kept_err": float(kept_err),
                                      "routed_err": float(routed_err),
                                      "n_routed": int(len(routed))}
        print(f"  keep {c:.0%}: kept err={kept_err:.3f}  routed err={routed_err:.3f}  "
              f"(routed {len(routed)} items)")

    json.dump(out, open(ROOT / "results/selective_summary.json", "w"), indent=2)
    print(f"\nwrote {ROOT/'results/selective_summary.json'}")


if __name__ == "__main__":
    main()
