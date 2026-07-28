"""
Rigor figures: (1) non-termination bar chart WITH Wilson 95% CI error bars,
(2) odds-ratio forest plot (distractor effect per dataset + pooled).
Nature (NPG) palette. Reads results/rigor_stats.json. Offline.
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT/"results/figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})
NPG = {"blue": "#4DBBD5", "red": "#E64B35", "teal": "#00A087", "navy": "#3C5488",
       "gray": "#8491B4"}
S = json.load(open(ROOT/"results/rigor_stats.json"))
DS = ["gsm8k", "math500", "gpqa_diamond"]
LAB = {"gsm8k": "GSM8K", "math500": "MATH-500", "gpqa_diamond": "GPQA-D"}


def fig_nonterm_ci():
    """Grouped bars, clean vs adversarial, with Wilson 95% CI error bars."""
    d = S["distractor_nonterm"]
    clean = [d[x]["clean"] for x in DS]   # [p, lo, hi]
    adv = [d[x]["adv"] for x in DS]
    x = np.arange(len(DS)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for off, arr, col, lb in [(-w/2, clean, NPG["blue"], "clean"),
                              (w/2, adv, NPG["red"], "adversarial (distractor)")]:
        p = [a[0] for a in arr]
        lo = [a[0]-a[1] for a in arr]; hi = [a[2]-a[0] for a in arr]
        ax.bar(x+off, p, w, color=col, label=lb,
               yerr=[lo, hi], capsize=3, error_kw={"lw": 1.1, "ecolor": "#333"})
        for xi, pi, hii in zip(x+off, p, [a[2] for a in arr]):
            ax.text(xi, hii+0.015, f"{pi:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LAB[x] for x in DS])
    ax.set_ylabel("non-termination rate")
    ax.set_ylim(0, 0.68)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG/"fig_nonterm.pdf"); fig.savefig(FIG/"fig_nonterm.png", dpi=150)
    plt.close(fig)


def fig_forest():
    """Odds-ratio forest plot: distractor -> non-termination, per dataset + pooled."""
    d = S["distractor_nonterm"]
    rows = DS + ["POOLED"]
    ors = [d[r]["or"] for r in rows]   # [or, lo, hi]
    ylab = [LAB.get(r, "Pooled") for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for yi, (o, lo, hi), r in zip(y, ors, rows):
        col = NPG["navy"] if r == "POOLED" else NPG["red"]
        ms = 9 if r == "POOLED" else 6
        ax.plot([lo, hi], [yi, yi], "-", color=col, lw=1.6, zorder=1)
        ax.plot(o, yi, "s" if r == "POOLED" else "o", color=col, ms=ms, zorder=2)
        # place the OR label below the marker to avoid running off the right edge
        ax.text(o, yi-0.32, f"OR {o:.1f} [{lo:.1f}, {hi:.1f}]", va="center",
                ha="center", fontsize=7.5, color=col)
    ax.axvline(1.0, ls="--", color=NPG["gray"], lw=1)
    ax.text(1.0, len(rows)-0.35, "no effect", fontsize=7.5, color=NPG["gray"], ha="center")
    ax.set_yticks(y); ax.set_yticklabels(ylab)
    ax.set_xscale("log"); ax.set_xlim(0.7, 6000)
    ax.set_xlabel("odds ratio (non-termination: adversarial vs. clean)")
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"])
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG/"fig_forest.pdf"); fig.savefig(FIG/"fig_forest.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_nonterm_ci()
    fig_forest()
    print("wrote fig_nonterm.pdf (with CI error bars) and fig_forest.pdf")
