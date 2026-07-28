"""
Main experiment runner for the overthinking / early-commit study.

For each item x dataset x condition{clean, adversarial}, we run ONE reasoning pass on
the reasoner (cached), analyze its trace, and evaluate several methods that all reuse
that same trace/output — so the expensive reasoning is paid once:

  Methods evaluated per item (all from the SAME reasoning pass unless noted):
    - full        : the reasoner's own final answer (baseline; = "think to the end")
    - early_commit_strong : trace-anchored commit via v4-pro committer
    - early_commit_weak   : trace-anchored commit via v4-flash committer
    - oracle_commit       : best of {full, first-stable-value} if EITHER equals gold
                            (upper bound on what commit *could* achieve; diagnostic only)
  Additional baseline (separate calls, only on a subset to control cost):
    - self_consistency@k  : majority vote over k temperature samples (reasoner)

Outputs a per-item JSONL with everything needed for later analysis (no re-runs needed).
Deterministic given cache. Respects MAX_SPEND_USD.
"""
import sys, os, json, pathlib, argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml
from llm_client import chat, get_spend
from datasets_load import load_gsm8k, load_math500, load_gpqa_diamond
from trace_method import (analyze_trace, extract_final_answer, stabilization_prefix,
                          numbers_in, norm_num)
from grade import grade

ROOT = pathlib.Path(__file__).resolve().parents[1]


def commit_from_prefix(question, prefix, committer, sidx):
    msg = [{"role": "user", "content":
            f"Problem:\n{question}\n\nReasoning so far:\n{prefix}\n\n"
            "Based ONLY on the reasoning above, give the final answer now without "
            "reconsidering. Respond as 'Answer: <answer>'."}]
    r = chat(msg, model=committer, temperature=0.0, max_tokens=80, sample_idx=sidx)
    return extract_final_answer(r.text), r


def run_item(it, cfg, condition):
    q = it["question"]
    if condition == "adversarial":
        q = q + cfg["adversarial"]["distractor"]
    prompt = q + "\nShow your reasoning, then end with 'Answer: <answer>'."
    reasoner = cfg["models"]["reasoner"]
    # ONE reasoning pass (cached)
    r = chat([{"role": "user", "content": prompt}], model=reasoner,
             temperature=cfg["sampling"]["greedy_temp"],
             max_tokens=cfg["sampling"]["max_tokens_reason"], sample_idx=0)
    reasoning = r.reasoning or r.text  # some models put chain in text
    sig = analyze_trace(reasoning, gold=it["gold"], tail_window=cfg["method"]["tail_window"])
    full_ans = extract_final_answer(r.text)

    # early-commit: find stabilization prefix, commit via strong + weak committers
    prefix, stable_val, stable_idx = stabilization_prefix(
        reasoning, patience=cfg["method"]["patience"])
    ec_strong, _ = commit_from_prefix(q, prefix, cfg["models"]["committer_strong"], 11)
    ec_weak, _ = commit_from_prefix(q, prefix, cfg["models"]["committer_weak"], 12)

    gt = it.get("gold_type", "number")
    rec = {
        "id": it["id"], "dataset": it["dataset"], "condition": condition,
        "gold": it["gold"],
        "reasoning_tokens": r.reasoning_tokens, "completion_tokens": r.completion_tokens,
        "n_steps": sig.n_steps, "n_switches": sig.n_switches,
        "switch_rate": round(sig.switch_rate, 4),
        "first_stable_idx": sig.first_stable_idx, "stable_commit_idx": stable_idx,
        "tail_entropy": round(sig.tail_entropy, 4), "n_distinct_tail": sig.n_distinct_tail,
        "reached_then_abandoned": sig.reached_then_abandoned,
        "final_candidate": sig.final_candidate, "stable_val": stable_val,
        "trajectory": sig.trajectory,
        "full_ans": full_ans, "ec_strong": ec_strong, "ec_weak": ec_weak,
        "correct_full": grade(full_ans, it["gold"], gt),
        "correct_ec_strong": grade(ec_strong, it["gold"], gt),
        "correct_ec_weak": grade(ec_weak, it["gold"], gt),
        "correct_stable_val": grade(stable_val, it["gold"], gt),
    }
    # oracle diagnostic: could committing at stabilization ever help/hurt?
    rec["oracle_commit_gain"] = int(rec["correct_stable_val"]) - int(rec["correct_full"])
    return rec


def self_consistency(it, cfg, condition, k):
    q = it["question"]
    if condition == "adversarial":
        q = q + cfg["adversarial"]["distractor"]
    prompt = q + "\nShow your reasoning, then end with 'Answer: <answer>'."
    votes = []
    for s in range(k):
        r = chat([{"role": "user", "content": prompt}], model=cfg["models"]["reasoner"],
                 temperature=cfg["sampling"]["sc_temp"],
                 max_tokens=cfg["sampling"]["max_tokens_reason"], sample_idx=100 + s)
        a = extract_final_answer(r.text)
        if a is not None:
            votes.append(a)
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/main.yaml"))
    ap.add_argument("--datasets", nargs="*", default=["gsm8k", "math500"])
    ap.add_argument("--conditions", nargs="*", default=["clean", "adversarial"])
    ap.add_argument("--sc", action="store_true", help="also run self-consistency baseline")
    ap.add_argument("--sc_n", type=int, default=50, help="subset size for SC (cost control)")
    ap.add_argument("--out", default=str(ROOT / "results/main_records.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.environ.setdefault("MAX_SPEND_USD", str(cfg["budget"]["max_spend_usd"]))

    loaders = {"gsm8k": load_gsm8k, "math500": load_math500,
               "gpqa_diamond": load_gpqa_diamond}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # build the full job list
    jobs = []
    for ds in args.datasets:
        n = cfg["datasets"].get(ds, 0)
        if n <= 0:
            continue
        items = loaders[ds](n)
        if not items:
            print(f"[skip] {ds}: no data")
            continue
        for it in items:
            for cond in args.conditions:
                jobs.append((it, cond))
    print(f"[plan] {len(jobs)} item-conditions across {args.datasets} x {args.conditions}")

    def work(job):
        it, cond = job
        rec = run_item(it, cfg, cond)
        if args.sc:
            rec["sc_ans"] = self_consistency(it, cfg, cond, cfg["sampling"]["sc_k"])
            rec["correct_sc"] = grade(rec["sc_ans"], it["gold"], it.get("gold_type", "number"))
        return rec

    written = 0
    with open(out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                if written % 25 == 0:
                    print(f"  {written}/{len(jobs)} done, spend ${get_spend():.3f}")
            except RuntimeError as e:
                print(f"[budget/stop] {e}")
                break
    print(f"WROTE {written} records to {out}; new spend ${get_spend():.4f}")


if __name__ == "__main__":
    main()
