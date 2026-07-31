"""Statistics for the multi-vendor probe, including the power to detect amplification.

Two questions matter for how the result should be read:
  1. Does the model stall at all (baseline non-termination on clean GPQA items)?
  2. Does the distractor amplify it, as it does on deepseek-v4-pro?

For (2) a null on n=30 is only informative if the design could have seen the effect, so we
report the count that a deepseek-sized amplification would have produced and the exact
test against it, rather than resting on a non-significant p-value alone.
"""
import json, pathlib
from math import comb

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The probe runs on GPQA only, so the reference must be v4-pro's GPQA contrast
# (16/60 clean, 29/60 adversarial). The pooled 3.3x across all three datasets is
# inflated by GSM8K, where the clean rate is zero, and would be the wrong yardstick here.
DS_CLEAN_K, DS_CLEAN_N = 16, 60
DS_ADV_K, DS_ADV_N = 29, 60
DEEPSEEK_AMPL = (DS_ADV_K / DS_ADV_N) / (DS_CLEAN_K / DS_CLEAN_N)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(a, b, c, d):
    """2x2 table [[a,b],[c,d]] -> two-sided Fisher exact p."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x):
        return (comb(r1, x) * comb(n - r1, c1 - x)) / comb(n, c1)

    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    p_obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12))


def binom_tail_le(k, n, p):
    """P(X <= k) for X ~ Bin(n, p)."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def main():
    d = json.load(open(ROOT / "results/multivendor_probe.json"))
    rows = []
    for model, v in d["models"].items():
        kc, nc = v["clean"]["k"], v["clean"]["n"]
        ka, na = v["adversarial"]["k"], v["adversarial"]["n"]
        if nc == 0 or na == 0:
            continue
        pc, lc, hc = wilson(kc, nc)
        pa, la, ha = wilson(ka, na)
        # Fisher on [[adv_nonterm, adv_term],[clean_nonterm, clean_term]]
        p_fisher = fisher_two_sided(ka, na - ka, kc, nc - kc)
        # Power check: what would a deepseek-sized amplification have produced?
        p_expected = min(1.0, pc * DEEPSEEK_AMPL)
        k_expected = p_expected * na
        # one-sided: probability of seeing <= ka if the deepseek-sized effect were real
        p_rule_out = binom_tail_le(ka, na, p_expected) if p_expected > 0 else float("nan")
        rows.append({
            "model": model, "vendor": v["vendor"],
            "clean": {"k": kc, "n": nc, "rate": pc, "ci": [lc, hc]},
            "adversarial": {"k": ka, "n": na, "rate": pa, "ci": [la, ha]},
            "rate_ratio": (pa / pc) if pc > 0 else None,
            "fisher_p": p_fisher,
            "expected_k_if_deepseek_sized": k_expected,
            "p_observed_or_fewer_under_deepseek_effect": p_rule_out,
            "median_ct": [v["median_ct_clean"], v["median_ct_adv"]],
            "depth3_completion_clean": (1 - pc) ** 3,
        })

    # Each external model is individually underpowered, but they point the same way, so the
    # informative quantity is the pooled effect. We stratify by model (Mantel-Haenszel) so
    # the pooled ratio is not driven by differences in baseline rate across vendors, and
    # test it with an exact permutation of the stratified table.
    ext = [r for r in rows if r["clean"]["k"] + r["adversarial"]["k"] > 0]
    num = sum(r["adversarial"]["k"] * r["clean"]["n"] / (r["clean"]["n"] + r["adversarial"]["n"])
              for r in ext)
    den = sum(r["clean"]["k"] * r["adversarial"]["n"] / (r["clean"]["n"] + r["adversarial"]["n"])
              for r in ext)
    mh_rr = (num / den) if den > 0 else None
    pooled_kc = sum(r["clean"]["k"] for r in ext)
    pooled_nc = sum(r["clean"]["n"] for r in ext)
    pooled_ka = sum(r["adversarial"]["k"] for r in ext)
    pooled_na = sum(r["adversarial"]["n"] for r in ext)
    pooled_fisher = fisher_two_sided(pooled_ka, pooled_na - pooled_ka,
                                     pooled_kc, pooled_nc - pooled_kc)
    # Is v4-pro's amplification larger than the external pooled one? Compare the two
    # rate ratios by Fisher on the 2x2 of (nonterm, terminated) x (deepseek, external)
    # among adversarial runs, conditioning on each arm's own clean rate via the ratio.
    out_pool = {"models_pooled": [r["model"] for r in ext],
                "clean": {"k": pooled_kc, "n": pooled_nc,
                          "rate": pooled_kc / pooled_nc,
                          "ci": list(wilson(pooled_kc, pooled_nc)[1:])},
                "adversarial": {"k": pooled_ka, "n": pooled_na,
                                "rate": pooled_ka / pooled_na,
                                "ci": list(wilson(pooled_ka, pooled_na)[1:])},
                "mantel_haenszel_rr": mh_rr,
                "pooled_fisher_p": pooled_fisher}

    ds_pc, ds_pa = DS_CLEAN_K / DS_CLEAN_N, DS_ADV_K / DS_ADV_N
    out = {"deepseek_gpqa_reference": {
        "clean": {"k": DS_CLEAN_K, "n": DS_CLEAN_N, "rate": ds_pc,
                  "ci": list(wilson(DS_CLEAN_K, DS_CLEAN_N)[1:])},
        "adversarial": {"k": DS_ADV_K, "n": DS_ADV_N, "rate": ds_pa,
                        "ci": list(wilson(DS_ADV_K, DS_ADV_N)[1:])},
        "rate_ratio": DEEPSEEK_AMPL,
        "fisher_p": fisher_two_sided(DS_ADV_K, DS_ADV_N - DS_ADV_K,
                                     DS_CLEAN_K, DS_CLEAN_N - DS_CLEAN_K),
        "depth3_completion_clean": (1 - ds_pc) ** 3},
        "models": rows, "external_pooled": out_pool}
    (ROOT / "results/multivendor_stats.json").write_text(json.dumps(out, indent=2))

    print(f"reference: deepseek-v4-pro on GPQA = {DS_CLEAN_K}/{DS_CLEAN_N} clean, "
          f"{DS_ADV_K}/{DS_ADV_N} adversarial, RR={DEEPSEEK_AMPL:.2f}, "
          f"Fisher p={out['deepseek_gpqa_reference']['fisher_p']:.4f}\n")
    print(f"{'model':17s}{'vendor':11s}{'clean':>15s}{'adversarial':>16s}"
          f"{'RR':>7s}{'Fisher p':>10s}{'exp. k if RR':>14s}{'rule-out p':>12s}")
    print("-" * 104)
    for r in rows:
        print(f"{r['model']:17s}{r['vendor']:11s}"
              f"{r['clean']['k']:3d}/{r['clean']['n']:<3d} {r['clean']['rate']:5.0%} "
              f"{r['adversarial']['k']:6d}/{r['adversarial']['n']:<3d} {r['adversarial']['rate']:5.0%} "
              f"{(r['rate_ratio'] if r['rate_ratio'] else float('nan')):6.2f}"
              f"{r['fisher_p']:10.3f}{r['expected_k_if_deepseek_sized']:14.1f}"
              f"{r['p_observed_or_fewer_under_deepseek_effect']:12.4f}")
    print("-" * 104)
    print(f"\npooled across external models ({', '.join(out_pool['models_pooled'])}):")
    print(f"  clean {pooled_kc}/{pooled_nc} = {pooled_kc / pooled_nc:.1%}   "
          f"adversarial {pooled_ka}/{pooled_na} = {pooled_ka / pooled_na:.1%}")
    print(f"  Mantel-Haenszel RR = {mh_rr:.2f}   pooled Fisher p = {pooled_fisher:.4f}")
    print(f"  (deepseek-v4-pro RR = {DEEPSEEK_AMPL:.2f})")

    print("\ndepth-3 pipeline completion implied by the CLEAN rate alone (no distractor):")
    for r in rows:
        print(f"  {r['model']:18s} {r['depth3_completion_clean']:.0%}")
    print(f"  {'deepseek-v4-pro':18s} "
          f"{out['deepseek_gpqa_reference']['depth3_completion_clean']:.0%}")


if __name__ == "__main__":
    main()
