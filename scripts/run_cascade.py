"""
Cascade experiment (honest, non-circular compute-saving story; reviewer W on routing).

Setup: the CHEAP model (v4-flash) answers first. Its OWN effort signal (reasoning tokens)
decides whether to escalate to the EXPENSIVE model (v4-pro, answers already cached in
main_records). This is a genuine cost saver: we only pay for pro on the escalated fraction.

We sweep the escalation fraction and report:
  - accuracy of the cascade vs (flash-only) and (pro-only)
  - total cost of the cascade vs pro-only
  - the accuracy-vs-cost Pareto point

Flash answers are NEW calls (cheap). Pro answers reuse the cache (free). Capped by budget.
"""
import sys, os, json, pathlib, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml, numpy as np
from llm_client import chat, get_spend, PRICE
from datasets_load import load_gsm8k, load_math500, load_gpqa_diamond
from trace_method import extract_final_answer
from grade import grade

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/main.yaml"))
DIST = CFG["adversarial"]["distractor"]
LOADERS = {"gsm8k": load_gsm8k, "math500": load_math500, "gpqa_diamond": load_gpqa_diamond}
pro_recs = {(r["id"], r["condition"]): r
            for r in (json.loads(l) for l in open(ROOT / "results/main_records.jsonl"))}


def flash_answer(it, cond):
    q = it["question"] + (DIST if cond == "adversarial" else "")
    prompt = q + "\nShow your reasoning, then end with 'Answer: <answer>'."
    r = chat([{"role": "user", "content": prompt}], model="deepseek-v4-flash",
             temperature=0.0, max_tokens=CFG["sampling"]["max_tokens_reason"], sample_idx=0)
    ans = extract_final_answer(r.text)
    return {"id": it["id"], "dataset": it["dataset"], "condition": cond,
            "flash_ans": ans, "flash_rtok": r.reasoning_tokens,
            "flash_ctok": r.completion_tokens, "flash_ptok": r.prompt_tokens,
            "correct_flash": int(bool(grade(ans, it["gold"], it.get("gold_type", "number")))),
            "gold": it["gold"], "gold_type": it.get("gold_type", "number")}


def out_cost(model, ptok, ctok):
    p = PRICE[model]; return (ptok * p["in_miss"] + ctok * p["out"]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    os.environ.setdefault("MAX_SPEND_USD", "3.0")

    jobs = []
    for ds in ["gsm8k", "math500", "gpqa_diamond"]:
        for it in LOADERS[ds](args.n):
            for cond in ["clean", "adversarial"]:
                jobs.append((it, cond))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(flash_answer, it, cond): 1 for it, cond in jobs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except RuntimeError as e:
                print(f"[budget/stop] {e}"); break
    print(f"flash answers: {len(rows)}; new spend ${get_spend():.4f}")

    # attach pro answer/correctness/cost from cache
    for r in rows:
        pr = pro_recs.get((r["id"], r["condition"]))
        r["correct_pro"] = int(bool(pr["correct_full"])) if pr else 0
        r["pro_ptok"] = pr.get("completion_tokens", 0) if pr else 0  # approx accounting
        r["pro_ctok"] = pr.get("completion_tokens", 0) if pr else 0
        r["pro_rtok"] = pr.get("reasoning_tokens", 0) if pr else 0

    n = len(rows)
    flash_acc = np.mean([r["correct_flash"] for r in rows])
    pro_acc = np.mean([r["correct_pro"] for r in rows])
    # per-item costs
    flash_cost = [out_cost("deepseek-v4-flash", r["flash_ptok"], r["flash_ctok"]) for r in rows]
    pro_cost = [out_cost("deepseek-v4-pro", 120, r["pro_ctok"]) for r in rows]  # ~120 prompt tok
    total_pro_only = sum(pro_cost)
    total_flash_only = sum(flash_cost)

    def cascade_sweep(subset_idx, tag):
        sub = [rows[i] for i in subset_idx]
        m = len(sub)
        if m == 0:
            return []
        eff = np.array([r["flash_rtok"] for r in sub])
        order = np.argsort(-eff)
        fcost = [flash_cost[i] for i in subset_idx]
        pcost = [pro_cost[i] for i in subset_idx]
        fa = np.mean([r["correct_flash"] for r in sub])
        pa = np.mean([r["correct_pro"] for r in sub])
        tot_pro = sum(pcost); tot_flash = sum(fcost)
        print(f"\n[{tag}] flash-only acc={fa:.3f} (${tot_flash:.4f}); "
              f"pro-only acc={pa:.3f} (${tot_pro:.4f}); n={m}")
        print(f"  {'escalate%':>10} {'acc':>7} {'cost':>9} {'%pro-cost':>10}")
        sweep = []
        for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
            k = int(frac * m); esc = set(order[:k].tolist())
            acc = np.mean([sub[i]["correct_pro"] if i in esc else sub[i]["correct_flash"]
                           for i in range(m)])
            cost = tot_flash + sum(pcost[i] for i in esc)
            sweep.append({"escalate_frac": frac, "acc": float(acc), "cost": float(cost),
                          "pct_pro_cost": float(cost/tot_pro) if tot_pro else 0})
            print(f"  {frac:>10.0%} {acc:>7.3f} ${cost:>8.4f} {cost/tot_pro:>9.1%}")
        return {"flash_acc": float(fa), "pro_acc": float(pa),
                "flash_cost": tot_flash, "pro_cost": tot_pro, "sweep": sweep}

    idx_all = list(range(n))
    idx_clean = [i for i, r in enumerate(rows) if r["condition"] == "clean"]
    idx_adv = [i for i, r in enumerate(rows) if r["condition"] == "adversarial"]
    out = {"all": cascade_sweep(idx_all, "ALL"),
           "clean": cascade_sweep(idx_clean, "CLEAN"),
           "adversarial": cascade_sweep(idx_adv, "ADVERSARIAL")}
    json.dump(out, open(ROOT / "results/cascade.json", "w"), indent=2)
    print(f"\nwrote cascade.json; total new spend ${get_spend():.4f}")


if __name__ == "__main__":
    main()
