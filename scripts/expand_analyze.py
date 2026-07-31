#!/usr/bin/env python3
"""
Analyze the expanded-sample run with the SAME extractor/grader/non-termination
predicate as the original 360-record study, so the two are strictly comparable.

Instead of trusting expand_run.py's quick regex, we re-derive every field from the
cached raw `content` in results/cache_expand/ using src/trace_method.extract_final_answer,
src/grade.grade, and analysis_utils.nonterm. Writes:
  - results/expand_records_clean.jsonl  (normalized per-item records)
  - results/expand_summary.json         (rates, Wilson CIs, rate/odds ratios)
and prints a comparison table vs the original headline numbers.
"""
import sys, json, pathlib, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from collections import defaultdict
import numpy as np
from trace_method import extract_final_answer
from grade import grade
from analysis_utils import wilson

HERE = pathlib.Path(__file__).resolve().parents[1]
CACHE = HERE / "results/cache_expand"
PROMPTS = HERE / "results/expand_prompts.jsonl"
CAP = 8192
MODEL = "deepseek-v4-pro"


def odds_ratio_ci(a, b, c, d):
    """2x2 [[a,b],[c,d]]; OR with Wald 95% CI on log-OR (Haldane +0.5), matching
    scripts/rigor_stats.py so expanded numbers are computed identically."""
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a_ * d_) / (b_ * c_)
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    lo = np.exp(np.log(or_) - 1.96 * se)
    hi = np.exp(np.log(or_) + 1.96 * se)
    return or_, lo, hi


def cache_key(prompt):
    return hashlib.sha256((MODEL + "|" + str(CAP) + "|" + prompt).encode()).hexdigest()


def nonterm_pred(full_ans, completion_tokens):
    return (full_ans in (None, "")) and completion_tokens >= CAP - 8


def main():
    jobs = [json.loads(l) for l in open(PROMPTS) if l.strip()]
    recs = []
    missing = 0
    for j in jobs:
        cf = CACHE / f"{cache_key(j['prompt'])}.json"
        if not cf.exists():
            missing += 1
            continue
        d = json.loads(cf.read_text())
        content = d.get("content", "")
        ct = d.get("completion_tokens", 0)
        # official extractor + grader (same as original study)
        ans = extract_final_answer(content)
        rec = {
            "id": j["id"], "dataset": j["dataset"], "condition": j["condition"],
            "gold": j["gold"],
            "full_ans": ans, "completion_tokens": ct,
            "finish_reason": d.get("finish_reason"),
            "correct_full": bool(grade(ans, j["gold"], j.get("gold_type", "number"))),
            "nonterm": bool(nonterm_pred(ans, ct)),
        }
        recs.append(rec)

    outp = HERE / "results/expand_records_clean.jsonl"
    with open(outp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    DS = ["gsm8k", "math500", "gpqa_diamond"]
    summary = {"n_total": len(recs), "missing_from_cache": missing, "per_dataset": {}}

    print(f"\n{'='*64}\nEXPANDED SAMPLE  (n={len(recs)}, missing={missing})\n{'='*64}")
    print(f"{'dataset':14s} {'clean nt':>16s} {'adv nt':>16s} {'rate ratio':>11s}")
    pooled = {"clean": [0, 0], "adv": [0, 0]}
    for ds in DS:
        cl = [r for r in recs if r["dataset"] == ds and r["condition"] == "clean"]
        ad = [r for r in recs if r["dataset"] == ds and r["condition"] == "adversarial"]
        kc, nc = sum(r["nonterm"] for r in cl), len(cl)
        ka, na = sum(r["nonterm"] for r in ad), len(ad)
        pc, lc, hc = wilson(kc, nc)
        pa, la, ha = wilson(ka, na)
        rr = (pa / pc) if pc > 0 else float("inf")
        pooled["clean"][0] += kc; pooled["clean"][1] += nc
        pooled["adv"][0] += ka; pooled["adv"][1] += na
        summary["per_dataset"][ds] = {
            "clean": {"k": kc, "n": nc, "rate": pc, "lo": lc, "hi": hc},
            "adv": {"k": ka, "n": na, "rate": pa, "lo": la, "hi": ha},
            "rate_ratio": rr}
        print(f"{ds:14s} {f'{pc:.0%}[{lc:.0%},{hc:.0%}] {kc}/{nc}':>16s} "
              f"{f'{pa:.0%}[{la:.0%},{ha:.0%}] {ka}/{na}':>16s} {rr:>10.1f}x")

    kc, nc = pooled["clean"]; ka, na = pooled["adv"]
    pc = kc / nc if nc else float("nan"); pa = ka / na if na else float("nan")
    # pooled odds ratio (adv vs clean)
    a, b = ka, na - ka  # adv nonterm, adv term
    c, d = kc, nc - kc  # clean nonterm, clean term
    orr = (a * d) / (b * c) if b > 0 and c > 0 else float("inf")
    summary["pooled"] = {"clean_rate": pc, "adv_rate": pa,
                         "rate_ratio": (pa / pc) if pc > 0 else None,
                         "odds_ratio": orr,
                         "clean": {"k": kc, "n": nc}, "adv": {"k": ka, "n": na}}
    # finished-answer accuracy (valid-only), clean vs adv
    def facc(cond):
        v = [r for r in recs if r["condition"] == cond and not r["nonterm"]]
        return (np.mean([r["correct_full"] for r in v]) if v else float("nan"), len(v))
    fc, nfc = facc("clean"); fa, nfa = facc("adversarial")
    summary["finished_acc"] = {"clean": fc, "n_clean": nfc, "adv": fa, "n_adv": nfa}

    print(f"\nPOOLED non-termination: clean {pc:.0%} ({kc}/{nc}) -> adv {pa:.0%} ({ka}/{na})")
    rr_str = f"{pa/pc:.2f}x" if pc > 0 else "inf (clean=0)"
    print(f"  rate ratio = {rr_str}   odds ratio = {orr:.2f}")
    print(f"finished-answer accuracy (valid-only): clean {fc:.3f} (n={nfc}) vs adv {fa:.3f} (n={nfa})")
    print(f"\n[original 360-record headline: 10%->33%, rate ratio 3.3x, OR 4.4, "
          f"finished acc 0.72 vs 0.73]")

    json.dump(summary, open(HERE / "results/expand_summary.json", "w"), indent=2)

    # Also emit a rigor_stats-compatible file so the nonterm/forest figures can be
    # regenerated on the expanded sample with the existing plotting scripts.
    rigor = {"distractor_nonterm": {}}
    pooled_a = pooled_b = pooled_c = pooled_d = 0
    for ds in DS:
        s = summary["per_dataset"].get(ds)
        if not s or s["clean"]["n"] == 0 or s["adv"]["n"] == 0:
            continue
        kc, nc = s["clean"]["k"], s["clean"]["n"]
        ka, na = s["adv"]["k"], s["adv"]["n"]
        a, b, c, d = ka, na - ka, kc, nc - kc
        orr, olo, ohi = odds_ratio_ci(a, b, c, d)
        pooled_a += a; pooled_b += b; pooled_c += c; pooled_d += d
        rigor["distractor_nonterm"][ds] = {
            "clean": [s["clean"]["rate"], s["clean"]["lo"], s["clean"]["hi"]],
            "adv": [s["adv"]["rate"], s["adv"]["lo"], s["adv"]["hi"]],
            "or": [orr, olo, ohi], "n_clean": nc, "n_adv": na}
    if pooled_b and pooled_c:
        orr, olo, ohi = odds_ratio_ci(pooled_a, pooled_b, pooled_c, pooled_d)
        pr_c = pooled_c / (pooled_c + pooled_d)
        pr_a = pooled_a / (pooled_a + pooled_b)
        _, plc, phc = wilson(pooled_c, pooled_c + pooled_d)
        _, pla, pha = wilson(pooled_a, pooled_a + pooled_b)
        rigor["distractor_nonterm"]["POOLED"] = {
            "clean": [pr_c, plc, phc], "adv": [pr_a, pla, pha],
            "or": [orr, olo, ohi]}
    json.dump(rigor, open(HERE / "results/rigor_stats_expand.json", "w"), indent=2)

    print(f"\nwrote {outp}, results/expand_summary.json, results/rigor_stats_expand.json")


if __name__ == "__main__":
    main()
