"""
Analysis of main_records.jsonl: overthinking measurement + method comparison +
significance tests. Pure local computation, no API cost. Outputs:
  - results/analysis_summary.json  (all numbers for the paper)
  - console tables
"""
import sys, json, pathlib, math, random
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

ROOT = pathlib.Path(__file__).resolve().parents[1]
REC = ROOT / "results/main_records.jsonl"


def load(path=REC):
    return [json.loads(l) for l in open(path) if l.strip()]


def acc(recs, key):
    xs = [1 if r.get(key) else 0 for r in recs]
    return sum(xs) / len(xs) if xs else float("nan"), len(xs)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def paired_bootstrap_diff(recs, key_a, key_b, n_boot=10000, seed=0):
    """Bootstrap 95% CI for mean(key_a) - mean(key_b), paired by item."""
    rng = random.Random(seed)
    a = [1 if r.get(key_a) else 0 for r in recs]
    b = [1 if r.get(key_b) else 0 for r in recs]
    n = len(a)
    diffs = []
    idx = list(range(n))
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in range(n)]
        da = sum(a[i] for i in s) / n
        db = sum(b[i] for i in s) / n
        diffs.append(da - db)
    diffs.sort()
    point = sum(a) / n - sum(b) / n
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return point, lo, hi


def mcnemar(recs, key_a, key_b):
    """McNemar test on paired binary correctness. Returns (b, c, chi2, approx_p)."""
    b = c = 0  # b: a right & b wrong; c: a wrong & b right
    for r in recs:
        ra, rb = bool(r.get(key_a)), bool(r.get(key_b))
        if ra and not rb:
            b += 1
        elif rb and not ra:
            c += 1
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
    # survival of chi2 with df=1
    try:
        from scipy.stats import chi2 as chi2dist
        p = float(chi2dist.sf(chi2, 1))
    except Exception:
        p = math.erfc(math.sqrt(chi2 / 2))
    return b, c, chi2, p


def summarize(recs):
    out = {}
    # group by (dataset, condition)
    groups = defaultdict(list)
    for r in recs:
        groups[(r["dataset"], r["condition"])].append(r)

    print(f"\n{'dataset/cond':22} {'n':>4} {'full':>6} {'ec_str':>7} {'ec_wk':>6} "
          f"{'oracle':>7} {'abandon%':>8} {'switch':>7} {'rtok':>6}")
    for (ds, cond), rs in sorted(groups.items()):
        n = len(rs)
        a_full = acc(rs, "correct_full")[0]
        a_ecs = acc(rs, "correct_ec_strong")[0]
        a_ecw = acc(rs, "correct_ec_weak")[0]
        a_orc = acc(rs, "correct_stable_val")[0]
        aband = mean([1 if r.get("reached_then_abandoned") else 0 for r in rs])
        sw = mean([r.get("n_switches") for r in rs])
        rt = mean([r.get("reasoning_tokens") for r in rs])
        print(f"{ds+'/'+cond:22} {n:>4} {a_full:>6.3f} {a_ecs:>7.3f} {a_ecw:>6.3f} "
              f"{a_orc:>7.3f} {aband*100:>7.1f}% {sw:>7.1f} {rt:>6.0f}")
        key = f"{ds}/{cond}"
        out[key] = {
            "n": n, "acc_full": a_full, "acc_ec_strong": a_ecs,
            "acc_ec_weak": a_ecw, "acc_oracle_stable": a_orc,
            "abandon_rate": aband, "mean_switches": sw, "mean_reasoning_tokens": rt,
        }
        # significance: ec_strong vs full
        pt, lo, hi = paired_bootstrap_diff(rs, "correct_ec_strong", "correct_full")
        b, c, chi2, p = mcnemar(rs, "correct_ec_strong", "correct_full")
        out[key]["ec_strong_minus_full"] = {"point": pt, "ci95": [lo, hi],
                                            "mcnemar_b_full_only": b,
                                            "mcnemar_c_ec_only": c, "chi2": chi2, "p": p}
        print(f"    ec_strong - full = {pt:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  "
              f"McNemar b={b} c={c} p={p:.4f}")

    # overthinking predictive check: does reached_then_abandoned predict full-wrong?
    ab = [r for r in recs if r.get("reached_then_abandoned")]
    nab = [r for r in recs if not r.get("reached_then_abandoned")]
    if ab:
        err_ab = 1 - acc(ab, "correct_full")[0]
        err_nab = 1 - acc(nab, "correct_full")[0]
        out["abandon_predicts_error"] = {"err_rate_when_abandoned": err_ab,
                                         "err_rate_when_not": err_nab,
                                         "n_abandoned": len(ab), "n_not": len(nab)}
        print(f"\n[predictive] full-answer error rate | abandoned={err_ab:.3f} "
              f"(n={len(ab)}) vs not-abandoned={err_nab:.3f} (n={len(nab)})")
    return out


if __name__ == "__main__":
    recs = load()
    print(f"loaded {len(recs)} records")
    out = summarize(recs)
    outpath = ROOT / "results/analysis_summary.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nwrote {outpath}")
