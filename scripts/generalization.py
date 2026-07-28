"""
Generalization experiments for the non-termination finding. Three variants, each a
real DeepSeek API run (cached, budget-guarded), reusing the exact prompt construction
and non-termination criterion from the main study:

  (E1) SECOND MODEL: does non-termination also appear on deepseek-v4-flash (a smaller
       same-family reasoner)? Not a cross-provider test (honest caveat), but shows the
       phenomenon is not unique to v4-pro. GPQA clean+adversarial subset.
  (E2) LARGER BUDGET (32768): does adversarial non-termination on GPQA-Diamond persist
       when the budget is 4x the 8192 run (16x the original 2048)? The decisive test of
       "it is a failure to terminate, not a too-small budget."
  (E3) MULTIPLE DISTRACTORS: is the 3.3x amplification specific to our one distractor
       template, or does it hold for other adversarial phrasings? GPQA subset, 2 new
       distractor styles vs clean.

Non-termination criterion (identical to the paper): finish_reason implies the answer
is empty AND completion_tokens >= cap-8. Detected from the API's own accounting.

Budget: hard MAX_SPEND_USD guard. Estimated worst case < $2. All calls cached so re-runs
are free. Writes results/generalization.json.

Usage:
  MAX_SPEND_USD=3.5 python3 scripts/generalization.py --n 30
"""
import sys, os, json, pathlib, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml
from llm_client import chat, get_spend
from datasets_load import load_gpqa_diamond
from trace_method import extract_final_answer

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKERS = 8   # llm_client is thread-safe (locked cache + spend); parallelize the slow calls

# Original distractor (from configs/main_8k.yaml) + two new styles for E3.
DISTRACTORS = {
    "orig": (" Note: a similar-looking problem circulated online with a different answer, "
             "and some solvers double the final total by mistake; think very carefully and "
             "do not be misled."),
    "authority": (" Note: a widely-cited textbook gives a different answer to a nearly "
                  "identical question, and several experts disagree about the correct "
                  "choice; reconsider carefully before committing."),
    "self-doubt": (" Warning: this question is a known trick where the obvious answer is "
                   "usually wrong and even careful solvers second-guess themselves; "
                   "double-check every step before answering."),
}


def is_nonterm(text, completion_tokens, cap):
    return (extract_final_answer(text) in (None, "")) and completion_tokens >= cap - 8


def run_one(item, condition, model, cap, distractor_text):
    q = item["question"]
    if condition != "clean":
        q = q + distractor_text
    prompt = q + "\nShow your reasoning, then end with 'Answer: <answer>'."
    r = chat([{"role": "user", "content": prompt}], model=model,
             temperature=0.0, max_tokens=cap, sample_idx=0)
    return {"id": item["id"], "condition": condition, "model": model, "cap": cap,
            "completion_tokens": r.completion_tokens, "reasoning_tokens": r.reasoning_tokens,
            "finish_reason": r.finish_reason,
            "nonterm": bool(is_nonterm(r.text, r.completion_tokens, cap))}


def rate(records, cond):
    sub = [x for x in records if x["condition"] == cond]
    if not sub:
        return float("nan"), 0
    return sum(x["nonterm"] for x in sub) / len(sub), len(sub)


def run_batch(jobs, tag):
    """jobs = list of (item, condition, model, cap, distractor_text). Runs concurrently."""
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_one, *j) for j in jobs]
        for i, f in enumerate(futs, 1):
            results.append(f.result())
            if i % 20 == 0:
                print(f"  [{tag}] {i}/{len(jobs)} done, spend ${get_spend():.3f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="GPQA items per condition (E1/E3)")
    ap.add_argument("--n_e2", type=int, default=20, help="GPQA items for E2 (slow 16k calls)")
    ap.add_argument("--e2_cap", type=int, default=16384, help="E2 larger budget (16k = 8x original)")
    args = ap.parse_args()
    os.environ.setdefault("MAX_SPEND_USD", "3.5")
    items = load_gpqa_diamond(args.n)
    items_e2 = items[:args.n_e2]
    if not items:
        print("no GPQA data"); return
    out = {"n_per_condition": len(items), "n_e2": len(items_e2), "start_spend": get_spend()}

    # E1: second model (v4-flash) at the 8192 budget, clean + adversarial(orig)
    print("=== E1: second model deepseek-v4-flash (8192 budget) ===", flush=True)
    jobs = [(it, cond, "deepseek-v4-flash", 8192, DISTRACTORS["orig"])
            for it in items for cond in ("clean", "adversarial")]
    e1 = run_batch(jobs, "E1")
    rc, nc = rate(e1, "clean"); ra, na = rate(e1, "adversarial")
    out["E1_second_model"] = {"model": "deepseek-v4-flash", "clean_rate": rc, "adv_rate": ra,
                              "multiplier": (ra / rc) if rc > 0 else None,
                              "n_clean": nc, "n_adv": na, "records": e1}
    print(f"  v4-flash non-term: clean {rc:.0%} (n={nc}), adv {ra:.0%} (n={na}), "
          f"mult {ra/rc if rc>0 else float('nan'):.1f}x", flush=True)

    # E2: larger budget (16384 = 8x original 2048, 2x the 8192 run) on GPQA
    cap2 = args.e2_cap
    print(f"\n=== E2: {cap2}-token budget on v4-pro (GPQA, n={len(items_e2)}) ===", flush=True)
    jobs = [(it, cond, "deepseek-v4-pro", cap2, DISTRACTORS["orig"])
            for it in items_e2 for cond in ("clean", "adversarial")]
    e2 = run_batch(jobs, "E2")
    rc2, nc2 = rate(e2, "clean"); ra2, na2 = rate(e2, "adversarial")
    out["E2_budget_larger"] = {"cap": cap2, "clean_rate": rc2, "adv_rate": ra2,
                               "multiplier": (ra2 / rc2) if rc2 > 0 else None,
                               "n_clean": nc2, "n_adv": na2, "records": e2}
    print(f"  {cap2} budget non-term: clean {rc2:.0%}, adv {ra2:.0%}  "
          f"(8192 run was clean 27%, adv 48% on full GPQA)", flush=True)

    # E3: multiple distractor styles at 8192 on v4-pro
    print("\n=== E3: multiple distractor styles (v4-pro, 8192) ===", flush=True)
    jobs = [(it, "clean", "deepseek-v4-pro", 8192, "") for it in items]
    for style in ("authority", "self-doubt"):
        jobs += [(it, style, "deepseek-v4-pro", 8192, DISTRACTORS[style]) for it in items]
    e3 = run_batch(jobs, "E3")
    rc3, _ = rate(e3, "clean")
    out["E3_distractors"] = {"clean_rate": rc3, "styles": {}, "records": e3}
    print(f"  clean {rc3:.0%}", flush=True)
    for style in ("authority", "self-doubt"):
        rs, ns = rate(e3, style)
        out["E3_distractors"]["styles"][style] = {"rate": rs, "n": ns,
                                                   "multiplier": (rs / rc3) if rc3 > 0 else None}
        print(f"  distractor '{style}': {rs:.0%} (n={ns}), mult {rs/rc3 if rc3>0 else float('nan'):.1f}x", flush=True)

    out["end_spend"] = get_spend()
    out["new_spend"] = get_spend() - out["start_spend"]
    json.dump(out, open(ROOT / "results/generalization.json", "w"), indent=2)
    print(f"\nnew spend this run: ${out['new_spend']:.4f}")
    print(f"wrote {ROOT/'results/generalization.json'}")


if __name__ == "__main__":
    main()
