"""
Rigor figures (polished): non-termination bars with Wilson 95% CIs, and an odds-ratio
forest plot. Journal-quality style via figstyle.py. Reads results/rigor_stats.json.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
import matplotlib.pyplot as plt
import numpy as np
from figstyle import apply_style, style_axes, bar_kw, ERRORBAR_KW, PALETTE as P

apply_style()
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
S = json.load(open(ROOT / "results/rigor_stats.json"))
DS = ["gsm8k", "math500", "gpqa_diamond"]
LAB = {"gsm8k": "GSM8K", "math500": "MATH-500", "gpqa_diamond": "GPQA-D"}


def fig_nonterm_ci():
    """Grouped bars, clean vs adversarial, with Wilson 95% CI error bars."""
    d = S["distractor_nonterm"]
    clean = [d[x]["clean"] for x in DS]   # [p, lo, hi]
    adv = [d[x]["adv"] for x in DS]
    x = np.arange(len(DS)); w = 0.34
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for off, arr, col, lb in [(-w/2, clean, P["clean"], "clean"),
                              (w/2, adv, P["adv"], "adversarial (distractor)")]:
        p = [a[0] for a in arr]
        lo = [max(0, a[0]-a[1]) for a in arr]; hi = [max(0, a[2]-a[0]) for a in arr]
        bars = ax.bar(x+off, p, w, label=lb,
                      **{k: v for k, v in bar_kw(col).items() if k != "width"},
                      yerr=[lo, hi], error_kw=ERRORBAR_KW)
        for xi, pi, a in zip(x+off, p, arr):
            ax.text(xi, a[2]+0.022, f"{pi:.0%}", ha="center", fontsize=8.5, color=col)
    ax.set_xticks(x); ax.set_xticklabels([LAB[v] for v in DS])
    ax.set_ylabel("non-termination rate")
    ax.set_ylim(0, 0.70)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(frameon=False, loc="upper left", handlelength=1.1)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_nonterm.pdf"); fig.savefig(FIG / "fig_nonterm.png")
    plt.close(fig)


def fig_forest():
    """Odds-ratio forest plot: distractor -> non-termination, per dataset + pooled."""
    d = S["distractor_nonterm"]
    rows = DS + ["POOLED"]
    ors = [d[r]["or"] for r in rows]
    ylab = [LAB.get(r, "Pooled (all)") for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for yi, (o, lo, hi), r in zip(y, ors, rows):
        pooled = (r == "POOLED")
        col = P["navy"] if pooled else P["adv"]
        ax.plot([lo, hi], [yi, yi], "-", color=col, lw=1.8, zorder=2,
                solid_capstyle="round")
        ax.plot(o, yi, "s" if pooled else "o", color=col,
                ms=11 if pooled else 7, zorder=3,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.text(o, yi - 0.34, f"OR {o:.1f}  [{lo:.1f}, {hi:.1f}]", va="center",
                ha="center", fontsize=8, color=col)
    ax.axvline(1.0, ls=(0, (4, 3)), color=P["gray"], lw=1.0, zorder=1)
    # "no effect" as a vertical label riding the null line, in the empty left strip
    ax.text(0.86, (len(rows)-1)/2.0, "no effect", fontsize=7.5, color=P["gray"],
            ha="center", va="center", rotation=90)
    ax.set_yticks(y); ax.set_yticklabels(ylab)
    ax.set_ylim(-0.7, len(rows)-0.2)
    ax.set_xscale("log"); ax.set_xlim(0.62, 6000)
    ax.set_xlabel("odds ratio: non-termination, adversarial vs. clean")
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.grid(axis="x", which="major", color=P["grid"], lw=0.7)
    ax.grid(axis="y", visible=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#5A6270")
    fig.tight_layout()
    fig.savefig(FIG / "fig_forest.pdf"); fig.savefig(FIG / "fig_forest.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_nonterm_ci()
    fig_forest()
    print("wrote fig_nonterm.pdf (Wilson CIs) and fig_forest.pdf (polished)")
