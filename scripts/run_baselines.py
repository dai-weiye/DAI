"""
UQ baselines to compare against the reasoning-effort confidence signal (reviewer W5).
Two black-box baselines, both on the SAME items as main_records.jsonl:

  1. Verbalized confidence: after the model answers, ask it "How confident are you
     (0-100)?" in a fresh call. Confidence = 100 - stated (so higher = more likely error).
     One extra call per item.
  2. Self-consistency agreement: sample k chains at temperature>0; confidence signal =
     1 - (votes for the modal answer / k) = disagreement. k extra calls per item.

We then compare error-detection AUROC of {reasoning_tokens, combiner} vs
{verbalized, self-consistency-disagreement} on the SAME items, and their risk-coverage.

Cost control: verbalized on all datasets (cheap, 1 call/item). Self-consistency only on
GSM8K+MATH numeric (k=5) to bound cost; capped by MAX_SPEND_USD.
"""
import sys, os, json, pathlib, argparse, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml
from llm_client import chat, get_spend
from datasets_load import load_gsm8k, load_math500, load_gpqa_diamond
from trace_method import extract_final_answer
from grade import grade

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/main.yaml"))
DIST = CFG["adversarial"]["distractor"]
LOADERS = {"gsm8k": load_gsm8k, "math500": load_math500, "gpqa_diamond": load_gpqa_diamond}


def q_with_cond(it, cond):
    q = it["question"] + (DIST if cond == "adversarial" else "")
    return q + "\nShow your reasoning, then end with 'Answer: <answer>'."


def verbalized_conf(it, cond):
    """One extra call: ask the model its confidence 0-100 about its own answer."""
    # reuse the cached main answer to keep it grounded
    r0 = chat([{"role": "user", "content": q_with_cond(it, cond)}],
              model=CFG["models"]["reasoner"], temperature=0.0,
              max_tokens=CFG["sampling"]["max_tokens_reason"], sample_idx=0)
    ans = extract_final_answer(r0.text)
    prompt = (f"Question: {it['question']}\nProposed answer: {ans}\n"
              "On a scale of 0 to 100, how confident are you that this answer is correct? "
              "Reply with ONLY an integer 0-100.")
    r = chat([{"role": "user", "content": prompt}], model=CFG["models"]["reasoner"],
             temperature=0.0, max_tokens=2000, sample_idx=700)
    m = re.findall(r"\d{1,3}", r.text)
    conf = int(m[-1]) if m else 50
    conf = max(0, min(100, conf))
    correct = grade(ans, it["gold"], it.get("gold_type", "number"))
    return {"id": it["id"], "dataset": it["dataset"], "condition": cond,
            "verbalized_conf": conf, "unc_verbalized": 100 - conf,
            "correct": int(bool(correct))}


def self_consistency(it, cond, k=5):
    votes = []
    for s in range(k):
        r = chat([{"role": "user", "content": q_with_cond(it, cond)}],
                 model=CFG["models"]["reasoner"], temperature=CFG["sampling"]["sc_temp"],
                 max_tokens=CFG["sampling"]["max_tokens_reason"], sample_idx=100 + s)
        a = extract_final_answer(r.text)
        if a is not None:
            votes.append(a)
    if not votes:
        return None
    modal, cnt = Counter(votes).most_common(1)[0]
    agree = cnt / len(votes)
    correct_vote = grade(modal, it["gold"], it.get("gold_type", "number"))
    return {"id": it["id"], "dataset": it["dataset"], "condition": cond,
            "sc_agree": agree, "unc_sc": 1 - agree,
            "sc_answer": modal, "correct_sc": int(bool(correct_vote))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="items per dataset (cost control)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    os.environ.setdefault("MAX_SPEND_USD", "3.0")

    # verbalized on all 3 datasets; self-consistency on numeric only
    vb_jobs, sc_jobs = [], []
    for ds in ["gsm8k", "math500", "gpqa_diamond"]:
        for it in LOADERS[ds](args.n):
            for cond in ["clean", "adversarial"]:
                vb_jobs.append((it, cond))
                if ds in ("gsm8k", "math500"):
                    sc_jobs.append((it, cond))
    print(f"[plan] verbalized={len(vb_jobs)} self-consistency={len(sc_jobs)} (k=5)")

    vb_out = ROOT / "results/baseline_verbalized.jsonl"
    sc_out = ROOT / "results/baseline_selfconsistency.jsonl"
    for jobs, fn, out in [(vb_jobs, verbalized_conf, vb_out),
                          (sc_jobs, self_consistency, sc_out)]:
        written = 0
        with open(out, "w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fn, it, cond): (it, cond) for it, cond in jobs}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                    if rec:
                        f.write(json.dumps(rec) + "\n"); f.flush(); written += 1
                except RuntimeError as e:
                    print(f"[budget/stop] {e}"); break
        print(f"wrote {written} -> {out.name}; spend so far ${get_spend():.3f}")
    print(f"TOTAL new spend ${get_spend():.4f}")


if __name__ == "__main__":
    main()
