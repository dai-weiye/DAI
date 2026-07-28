"""
Figures for the reframed paper ("Reasoning effort is a confidence signal").
Reads results/main_records.jsonl. No API cost. Colorblind-safe, print-friendly.
Saves PDF+PNG to results/figures/.
"""
import sys, json, pathlib
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
C = {"a": "#4DBBD5", "b": "#E64B35", "c": "#00A087", "d": "#3C5488", "gray": "#8491B4"}
recs = [json.loads(l) for l in open(ROOT / "results/main_records.jsonl") if l.strip()]
for r in recs:
    r["err"] = 0 if r.get("correct_full") else 1


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg) if npos and nneg else float("nan")


def zscore_within_strata(field):
    strata = defaultdict(list)
    for i, r in enumerate(recs):
        strata[(r["dataset"], r["condition"])].append(i)
    z = np.zeros(len(recs))
    for _, idxs in strata.items():
        v = np.array([float(recs[i].get(field) or 0) for i in idxs])
        z[idxs] = (v - v.mean()) / (v.std() + 1e-9)
    return z


def fig_risk_coverage():
    """Risk-coverage: accuracy on retained items as we abstain by uncertainty (reasoning_tokens)."""
    z = zscore_within_strata("reasoning_tokens")
    labels = np.array([r["err"] for r in recs]); correct = 1 - labels
    order = np.argsort(z)  # ascending effort = most confident kept first
    covs = np.linspace(0.4, 1.0, 13)
    accs = [correct[order[:max(1, int(c*len(recs)))]].mean() for c in covs]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(covs, accs, "-o", color=C["a"], ms=4)
    ax.axhline(correct.mean(), ls="--", color=C["gray"])
    ax.text(0.42, correct.mean()+0.012, "answer all", color=C["gray"], fontsize=8)
    ax.text(0.42, accs[0]+0.012, "abstain by effort", color=C["a"], fontsize=8)
    ax.set_xlabel("coverage"); ax.set_ylabel("accuracy on answered")
    ax.set_ylim(min(accs)-0.05, max(accs)+0.05)
    fig.tight_layout(pad=0.4); fig.savefig(FIG / "fig_risk_coverage.pdf"); fig.savefig(FIG / "fig_risk_coverage.png")
    plt.close(fig)


def fig_auroc_bars():
    """Per-stratum AUROC of reasoning_tokens for error detection."""
    strata = defaultdict(list)
    for i, r in enumerate(recs):
        strata[(r["dataset"], r["condition"])].append(i)
    keys = sorted(strata)
    a = [auroc([recs[i]["reasoning_tokens"] for i in strata[k]],
               [recs[i]["err"] for i in strata[k]]) for k in keys]
    labels = [f"{d.replace('_diamond','')}\n{c[:4]}" for d, c in keys]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(range(len(keys)), a, color=C["c"])
    ax.axhline(0.5, ls="--", color=C["gray"], lw=1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUROC (error detection)"); ax.set_ylim(0, 1.08)
    for i, v in enumerate(a):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(pad=0.4); fig.savefig(FIG / "fig_auroc.pdf"); fig.savefig(FIG / "fig_auroc.png")
    plt.close(fig)


def fig_negative_earlycommit():
    """The honest negative result: early-commit vs full accuracy."""
    strata = defaultdict(list)
    for r in recs:
        strata[(r["dataset"], r["condition"])].append(r)
    keys = sorted(strata)
    full = [np.mean([bool(r["correct_full"]) for r in strata[k]]) for k in keys]
    ec = [np.mean([bool(r["correct_ec_strong"]) for r in strata[k]]) for k in keys]
    labels = [f"{d.replace('_diamond','')}\n{c[:4]}" for d, c in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(x - w/2, full, w, label="trust model's answer", color=C["a"])
    ax.bar(x + w/2, ec, w, label="early-commit (naive override)", color=C["b"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(pad=0.4); fig.savefig(FIG / "fig_earlycommit_neg.pdf"); fig.savefig(FIG / "fig_earlycommit_neg.png")
    plt.close(fig)


def fig_trajectory_example():
    """Answer trajectory of a reach-then-abandon case, plotted on a numeric y-axis."""
    def tonum(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    cands = [r for r in recs if r.get("reached_then_abandoned") and r.get("trajectory")
             and len(r["trajectory"]) > 10]
    if not cands:
        return
    r = max(cands, key=lambda r: r["n_switches"])
    traj = r["trajectory"]; gold = tonum(r["gold"])
    xs, ys = [], []
    for i, v in enumerate(traj):
        nv = tonum(v)
        if nv is not None:
            xs.append(i); ys.append(nv)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.plot(xs, ys, "-o", color=C["b"], lw=1, ms=3, zorder=2, label="working answer")
    if gold is not None:
        ax.axhline(gold, ls="--", color=C["c"], lw=1.2, zorder=1, label=f"gold = {r['gold']}")
    ax.set_yscale("symlog")
    ax.set_xlabel("step in reasoning trace"); ax.set_ylabel("working answer (symlog)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(pad=0.4); fig.savefig(FIG / "fig_trajectory.pdf"); fig.savefig(FIG / "fig_trajectory.png")
    plt.close(fig)


def fig_cascade():
    """Accuracy vs cost for the effort-gated flash->pro cascade, per condition."""
    p = ROOT / "results/cascade.json"
    if not p.exists():
        return
    d = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, cond, col in [(axes[0], "clean", C["a"]), (axes[1], "adversarial", C["b"])]:
        s = d[cond]["sweep"]
        cost = [x["cost"] for x in s]; acc = [x["acc"] for x in s]
        ax.plot(cost, acc, "-o", color=col, ms=4)
        ax.axhline(d[cond]["pro_acc"], ls="--", color=C["gray"], lw=1)
        ax.axhline(d[cond]["flash_acc"], ls=":", color=C["gray"], lw=1)
        ax.text(cost[-1], d[cond]["pro_acc"], "pro-only ", fontsize=7, va="bottom", ha="right", color=C["gray"])
        ax.text(cost[0], d[cond]["flash_acc"], " flash-only", fontsize=7, va="top", ha="left", color=C["gray"])
        ax.set_xlabel("cost (USD)"); ax.set_title(cond, fontsize=9)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout(pad=0.5); fig.savefig(FIG / "fig_cascade.pdf"); fig.savefig(FIG / "fig_cascade.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_risk_coverage()
    fig_auroc_bars()
    fig_negative_earlycommit()
    fig_trajectory_example()
    fig_cascade()
    print(f"wrote figures to {FIG}:")
    for p in sorted(FIG.glob("*.pdf")):
        print("  ", p.name)
