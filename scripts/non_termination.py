"""
Non-termination as a failure mode (robust, honest headline candidate).

Compares the 2048-budget and 8192-budget runs: how often does the reasoning model
fail to emit an answer (truncation / non-termination), and how does the adversarial
distractor drive it? This is a real, gold-independent phenomenon that survives the
truncation critique (it IS the truncation, characterized honestly).

Usage: python non_termination.py results/main_records.jsonl 2048 results/main_records_8k.jsonl 8192
"""
import sys, json, pathlib
from collections import defaultdict

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def trunc(r, cap):
    return r.get("full_ans") in (None, "") and r.get("completion_tokens", 0) >= cap - 8

def report(recs, cap, tag):
    strata = defaultdict(lambda: [0, 0])
    for r in recs:
        s = strata[(r["dataset"], r["condition"])]
        s[0] += trunc(r, cap); s[1] += 1
    print(f"\n=== {tag} (cap={cap}, n={len(recs)}) ===")
    print(f"{'stratum':26} {'non-termination rate':>20}")
    overall = [0, 0]
    for k in sorted(strata):
        a, b = strata[k]
        overall[0] += a; overall[1] += b
        print(f"  {k[0]:12}/{k[1]:12} {a:>3}/{b:<3} = {a/b:5.0%}")
    print(f"  {'OVERALL':25} {overall[0]}/{overall[1]} = {overall[0]/overall[1]:.0%}")
    # clean vs adversarial contrast
    clean = sum(trunc(r, cap) for r in recs if r["condition"] == "clean")
    adv = sum(trunc(r, cap) for r in recs if r["condition"] == "adversarial")
    nc = sum(1 for r in recs if r["condition"] == "clean")
    na = sum(1 for r in recs if r["condition"] == "adversarial")
    print(f"  clean non-term: {clean}/{nc}={clean/max(nc,1):.0%}   "
          f"adversarial: {adv}/{na}={adv/max(na,1):.0%}   "
          f"(distractor multiplies non-termination {(adv/max(na,1))/max(clean/max(nc,1),1e-9):.1f}x)")

if __name__ == "__main__":
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        path, cap = args[i], int(args[i+1])
        report(load(path), cap, pathlib.Path(path).stem)
