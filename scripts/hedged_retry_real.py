"""Measure, rather than assume, whether a hedged-timeout retry escapes a stall.

The liveness simulation credits retry with recovering most of the lost completion rate,
but only under the assumption that retries hang independently. Non-termination is
problem-induced, so that assumption is the weak point: a resample of the same item may
re-enter the same doubt loop. This script tests it directly.

For every adversarial run that non-terminated at the 8192 budget we issue up to two
resamples at temperature 0.7 and record whether they terminate. The measured conditional
escape probability replaces the independence assumption in the depth-K arithmetic, so the
mitigation number reported in the paper is grounded in real retries.

Usage: MAX_SPEND_USD=4 python3 scripts/hedged_retry_real.py
"""
import sys, os, json, pathlib, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm_client import chat, get_spend  # noqa: E402
from trace_method import extract_final_answer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = 8192
DISTRACTOR = (" Note: a similar-looking problem circulated online with a different answer, "
              "and some solvers double the final total by mistake; think very carefully and "
              "do not be misled.")
RETRY_TEMP = 0.7
MAX_RETRIES = 2


def nonterm(text, ct):
    return (extract_final_answer(text) in (None, "")) and ct >= CAP - 8


def question_of(rec, items_by_id):
    q = items_by_id[(rec["dataset"], rec["id"])]
    return q + (DISTRACTOR if rec["condition"] == "adversarial" else "")


def retry_item(rec, prompt):
    """Resample a stalled item up to MAX_RETRIES times; report the first escape."""
    outcomes = []
    for k in range(1, MAX_RETRIES + 1):
        try:
            r = chat([{"role": "user", "content": prompt}], model="deepseek-v4-pro",
                     temperature=RETRY_TEMP, max_tokens=CAP, sample_idx=100 + k)
        except Exception as e:
            # An unreachable API is not evidence about the model; drop the item rather
            # than scoring a missing retry as a failure to escape.
            return {"dataset": rec["dataset"], "id": rec["id"],
                    "condition": rec["condition"], "retries": outcomes,
                    "incomplete": True, "error": str(e)[:80],
                    "escaped": None, "escaped_on": None}
        nt = nonterm(r.text, r.completion_tokens)
        outcomes.append({"attempt": k, "nonterm": bool(nt),
                         "completion_tokens": r.completion_tokens})
        if not nt:
            break
    return {"dataset": rec["dataset"], "id": rec["id"], "condition": rec["condition"],
            "retries": outcomes,
            "escaped": any(not o["nonterm"] for o in outcomes),
            "escaped_on": next((o["attempt"] for o in outcomes if not o["nonterm"]), None)}


def load_questions():
    sys.path.insert(0, str(ROOT / "src"))
    from datasets_load import load_gsm8k, load_math500, load_gpqa_diamond
    out = {}
    for name, fn in (("gsm8k", load_gsm8k), ("math500", load_math500),
                     ("gpqa_diamond", load_gpqa_diamond)):
        for it in fn(200):
            out[(name, it["id"])] = it["question"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap on stalled items to retry")
    args = ap.parse_args()
    os.environ.setdefault("MAX_SPEND_USD", "4.0")

    recs = [json.loads(l) for l in open(ROOT / "results/main_records_8k.jsonl")]
    stalled = [r for r in recs
               if r["condition"] == "adversarial"
               and (not r.get("full_ans")) and r["completion_tokens"] >= CAP - 8]
    if args.limit:
        stalled = stalled[:args.limit]
    qs = load_questions()
    stalled = [r for r in stalled if (r["dataset"], r["id"]) in qs]
    print(f"=== real hedged retry: {len(stalled)} stalled adversarial runs, "
          f"<= {MAX_RETRIES} resamples each at T={RETRY_TEMP} ===", flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(retry_item, r, question_of(r, qs) +
                          "\nShow your reasoning, then end with 'Answer: <answer>'.")
                for r in stalled]
        for i, f in enumerate(futs, 1):
            out.append(f.result())
            if i % 10 == 0:
                print(f"  {i}/{len(futs)}  spend ${get_spend():.3f}", flush=True)

    dropped = [o for o in out if o.get("incomplete")]
    out = [o for o in out if not o.get("incomplete")]
    if dropped:
        print(f"  [note] {len(dropped)} items dropped (API unreachable), "
              f"reporting on {len(out)}", flush=True)
    n = len(out)
    esc = sum(o["escaped"] for o in out)
    first = sum(1 for o in out if o["escaped_on"] == 1)
    # per-attempt escape probability, conditional on the previous attempt having stalled
    p1 = first / n if n else float("nan")
    still = [o for o in out if o["escaped_on"] != 1]
    p2 = (sum(1 for o in still if o["escaped_on"] == 2) / len(still)) if still else float("nan")

    # Independence would predict the same escape probability as a fresh draw (1 - q).
    q_adv = 0.3333
    summary = {
        "n_stalled": n, "n_dropped_api": len(dropped),
        "escaped_any": esc, "escape_rate": esc / n if n else None,
        "p_escape_attempt1": p1, "p_escape_attempt2_given_still_stalled": p2,
        "independence_prediction_attempt1": 1 - q_adv,
        "correlated": bool(p1 < 1 - q_adv),
        "records": out,
    }
    # depth-K completion with retry, using the MEASURED escape probability per stage
    p_stage_ok = (1 - q_adv) + q_adv * (esc / n if n else 0.0)
    summary["stage_success_with_retry"] = p_stage_ok
    summary["completion_with_retry"] = {K: p_stage_ok ** K for K in (1, 2, 3, 5)}
    summary["completion_naive"] = {K: (1 - q_adv) ** K for K in (1, 2, 3, 5)}

    (ROOT / "results/hedged_retry_real.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 66)
    print(f"stalled runs retried              : {n}")
    print(f"escaped within {MAX_RETRIES} retries        : {esc} ({esc/n:.0%})" if n else "")
    print(f"escape on 1st retry               : {p1:.0%}  "
          f"(independence would predict {1-q_adv:.0%})")
    print(f"escape on 2nd | 1st still stalled : {p2:.0%}")
    print(f"per-stage success with retry      : {p_stage_ok:.0%}")
    print(f"depth-3 completion  naive {(1-q_adv)**3:.0%} -> with retry "
          f"{p_stage_ok**3:.0%}")
    print(f"new spend: ${get_spend():.4f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
