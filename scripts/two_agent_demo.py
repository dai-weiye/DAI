"""
Real 2-agent pipeline demo (anchors the Monte-Carlo liveness simulation with an actual
distributed run). solver -> verifier, each a real reasoning-LLM call under a fixed budget.

A task COMPLETES only if BOTH stages terminate with an answer; if either stage exhausts
its budget (non-termination), the task is BLOCKED. We measure the empirical end-to-end
completion rate on clean vs adversarial GPQA items, and compare to the independence-model
prediction (1-q)^2 to show the simulation is grounded, not just assumed.

Real DeepSeek calls, cached, budget-guarded. Writes results/two_agent.json.
Usage: MAX_SPEND_USD=4 python3 scripts/two_agent_demo.py --n 20
"""
import sys, os, json, pathlib, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm_client import chat, get_spend
from datasets_load import load_gpqa_diamond
from trace_method import extract_final_answer

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = 8192
DISTRACTOR = (" Note: a similar-looking problem circulated online with a different answer, "
              "and some solvers double the final total by mistake; think very carefully and "
              "do not be misled.")


def nonterm(text, ct):
    return (extract_final_answer(text) in (None, "")) and ct >= CAP - 8


def run_pipeline(item, condition):
    """solver -> verifier. Returns per-stage termination + end-to-end completion."""
    q = item["question"] + (DISTRACTOR if condition == "adversarial" else "")
    # Stage 1: solver produces an answer with reasoning
    s_prompt = q + "\nShow your reasoning, then end with 'Answer: <answer>'."
    s = chat([{"role": "user", "content": s_prompt}], model="deepseek-v4-pro",
             temperature=0.0, max_tokens=CAP, sample_idx=0)
    s_nt = nonterm(s.text, s.completion_tokens)
    s_ans = extract_final_answer(s.text)
    stage2_ran = False
    v_nt = None
    if not s_nt:
        # Stage 2: verifier checks the solver's answer (only runs if solver terminated)
        stage2_ran = True
        v_prompt = (q + f"\n\nA solver proposed this answer: {s_ans}\n"
                    "Verify it by reasoning independently, then end with 'Answer: <answer>'.")
        v = chat([{"role": "user", "content": v_prompt}], model="deepseek-v4-pro",
                 temperature=0.0, max_tokens=CAP, sample_idx=1)
        v_nt = nonterm(v.text, v.completion_tokens)
    completed = (not s_nt) and stage2_ran and (not v_nt)
    return {"id": item["id"], "condition": condition,
            "solver_nonterm": bool(s_nt), "verifier_ran": stage2_ran,
            "verifier_nonterm": bool(v_nt) if v_nt is not None else None,
            "completed": bool(completed)}


def summarize(records, cond):
    sub = [r for r in records if r["condition"] == cond]
    n = len(sub)
    comp = sum(r["completed"] for r in sub) / n
    s_nt = sum(r["solver_nonterm"] for r in sub) / n
    # per-stage non-termination among stages that actually ran
    ran2 = [r for r in sub if r["verifier_ran"]]
    v_nt = (sum(r["verifier_nonterm"] for r in ran2) / len(ran2)) if ran2 else float("nan")
    q = (s_nt + v_nt) / 2 if ran2 else s_nt   # avg per-stage rate
    return {"n": n, "completion_rate": comp, "solver_nonterm": s_nt,
            "verifier_nonterm": v_nt, "blocked_rate": 1 - comp,
            "indep_pred_completion": (1 - q) ** 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    os.environ.setdefault("MAX_SPEND_USD", "4.0")
    items = load_gpqa_diamond(args.n)
    jobs = [(it, cond) for it in items for cond in ("clean", "adversarial")]
    print(f"=== Real 2-agent pipeline (solver->verifier), n={args.n}/condition ===", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_pipeline, it, c) for it, c in jobs]
        for i, f in enumerate(futs, 1):
            recs.append(f.result())
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} done, spend ${get_spend():.3f}", flush=True)
    out = {"n": args.n, "clean": summarize(recs, "clean"),
           "adversarial": summarize(recs, "adversarial"), "records": recs,
           "new_spend": get_spend()}
    for cond in ("clean", "adversarial"):
        s = out[cond]
        print(f"  {cond}: end-to-end completion {s['completion_rate']:.0%} "
              f"(blocked {s['blocked_rate']:.0%}); solver-nt {s['solver_nonterm']:.0%}, "
              f"verifier-nt {s['verifier_nonterm']:.0%}; "
              f"independence-model predicts {s['indep_pred_completion']:.0%}", flush=True)
    json.dump(out, open(ROOT / "results/two_agent.json", "w"), indent=2)
    print(f"new spend: ${out['new_spend']:.4f}\nwrote {ROOT/'results/two_agent.json'}", flush=True)


if __name__ == "__main__":
    main()
