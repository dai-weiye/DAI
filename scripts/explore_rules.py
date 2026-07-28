"""
Offline exploration of BETTER, gold-free decision rules over the cached answer
trajectories. NO API cost (reuses stored 'trajectory' field). Compares:
  - full            : reasoner's own final answer (baseline)
  - traj_mode       : majority vote over ALL candidate answers in the trace
  - traj_mode_tail  : majority vote over the last-half of the trajectory
  - selective_mode  : if switch_rate >= thresh -> traj_mode else full
  - selective_early : if switch_rate >= thresh -> first_stable value else full
Reports accuracy per (dataset, condition) and overall, with paired McNemar vs full.
"""
import sys, json, pathlib
from collections import Counter, defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from grade import grade

ROOT = pathlib.Path(__file__).resolve().parents[1]
recs = [json.loads(l) for l in open(ROOT / "results/main_records.jsonl") if l.strip()]


def mode(xs):
    if not xs:
        return None
    return Counter(xs).most_common(1)[0][0]


def decide(r, rule, thresh=0.30):
    traj = r.get("trajectory") or []
    full = r.get("full_ans")
    sr = r.get("switch_rate", 0.0)
    if rule == "full":
        return full
    if rule == "traj_mode":
        return mode(traj) if traj else full
    if rule == "traj_mode_tail":
        tail = traj[len(traj)//2:] if traj else []
        return mode(tail) if tail else full
    if rule == "selective_mode":
        return mode(traj) if (traj and sr >= thresh) else full
    if rule == "selective_early":
        if traj and sr >= thresh:
            return traj[0]
        return full
    return full


RULES = ["full", "traj_mode", "traj_mode_tail", "selective_mode", "selective_early"]


def acc_by_group(rule, thresh):
    groups = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for r in recs:
        pred = decide(r, rule, thresh)
        ok = grade(pred, r["gold"], r.get("gold_type", "number"))
        g = groups[(r["dataset"], r["condition"])]
        g[0] += ok; g[1] += 1
        overall[0] += ok; overall[1] += 1
    return groups, overall


def mcnemar_vs_full(rule, thresh):
    b = c = 0
    for r in recs:
        rf = grade(decide(r, "full", thresh), r["gold"], r.get("gold_type", "number"))
        rr = grade(decide(r, rule, thresh), r["gold"], r.get("gold_type", "number"))
        if rf and not rr: b += 1
        elif rr and not rf: c += 1
    return b, c


print(f"{'rule':16}", " ".join(f"{d[:4]}/{c[:3]}" for d, c in sorted(
    {(r['dataset'], r['condition']) for r in recs})), "  OVERALL")
keys = sorted({(r['dataset'], r['condition']) for r in recs})
for rule in RULES:
    for thresh in ([0.30] if rule.startswith("selective") else [0.0]):
        groups, overall = acc_by_group(rule, thresh)
        cells = " ".join(f"{groups[k][0]/groups[k][1]:.2f}   " for k in keys)
        b, c = mcnemar_vs_full(rule, thresh)
        tag = f"{rule}" + (f"@{thresh}" if rule.startswith("selective") else "")
        print(f"{tag:16} {cells}  {overall[0]/overall[1]:.3f}  (vs full: worse={b} better={c})")

# threshold sweep for selective_mode
print("\nselective_mode threshold sweep (overall acc):")
for th in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    _, overall = acc_by_group("selective_mode", th)
    b, c = mcnemar_vs_full("selective_mode", th)
    print(f"  thresh={th}: overall={overall[0]/overall[1]:.3f}  worse={b} better={c}")
