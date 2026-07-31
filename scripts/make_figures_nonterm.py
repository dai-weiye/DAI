"""
Figures for the NON-TERMINATION paper. Reads results/non_termination_deep.json.
Journal-quality style via figstyle.py. NO API cost. Saves PDF+PNG to results/figures/.

  fig_budget     : budget sensitivity 2048 -> 8192 (slope chart).
  fig_mechanism  : in-trace mechanism signals, non-terminating vs terminating, with CIs.
(fig_nonterm is produced by make_figures_rigor.py with Wilson error bars.)
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
import matplotlib.pyplot as plt
import numpy as np
from figstyle import apply_style, style_axes, bar_kw, ERRORBAR_KW, PALETTE as P

apply_style()
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
DEEP = json.load(open(ROOT / "results/non_termination_deep.json"))
DS = ["gsm8k", "math500", "gpqa_diamond"]
DSLAB = {"gsm8k": "GSM8K", "math500": "MATH-500", "gpqa_diamond": "GPQA-D"}


def fig_budget():
    """Slope chart: adversarial non-termination at 2048 vs 8192, per dataset."""
    bs = DEEP["budget_sensitivity"]
    r2 = [bs[f"{d}/adversarial"]["rate_2k"] for d in DS]
    r8 = [bs[f"{d}/adversarial"]["rate_8k"] for d in DS]
    cols = [P["adv"], P["accent"], P["navy"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, d in enumerate(DS):
        ax.plot([0, 1], [r2[i], r8[i]], "-", color=cols[i], lw=2.0, zorder=2,
                solid_capstyle="round")
        ax.plot([0, 1], [r2[i], r8[i]], "o", color=cols[i], ms=7, zorder=3,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.text(-0.06, r2[i], f"{r2[i]:.0%}", va="center", ha="right",
                fontsize=9, color=cols[i])
        ax.text(1.06, r8[i], f"{r8[i]:.0%}  {DSLAB[d]}", va="center", ha="left",
                fontsize=9, color=cols[i])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["2,048", "8,192"])
    ax.set_xlabel("token budget")
    ax.set_xlim(-0.42, 1.7)
    ax.set_ylabel("non-termination rate (adversarial)")
    ax.set_ylim(0, max(r2) + 0.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_budget.pdf"); fig.savefig(FIG / "fig_budget.png")
    plt.close(fig)


def fig_mechanism():
    """Two panels: self-doubt density and verbatim looping, term vs non-term, with
    a bracketed gap-CI annotation. Slim bars, refined palette, title in caption."""
    m = DEEP["mechanism"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    specs = [("doubt_density", "self-doubt markers per 1k chars", axes[0], 0.001),
             ("loop_ratio", "verbatim 12-gram looping", axes[1], 0.0001)]
    for key, ylab, ax, _pad in specs:
        tm = m[key]["term_mean"]; nt = m[key]["nonterm_mean"]
        bars = ax.bar([0, 1], [tm, nt],
                      color=[P["clean"], P["adv"]], **{k: v for k, v in bar_kw(None).items() if k != "color"})
        top = max(nt, tm)
        for x, v in zip([0, 1], [tm, nt]):
            ax.text(x, v + top * 0.03, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color=P["ink"])
        # significance bracket between the two bars
        lo, hi = m[key]["ci"]
        sig = "$\\ast$" if m[key]["significant"] else "n.s."
        ybr = top * 1.16
        ax.plot([0, 0, 1, 1], [top*1.08, ybr, ybr, top*1.08], lw=0.9, color=P["gray"])
        ax.text(0.5, ybr + top*0.015, sig, ha="center", va="bottom",
                fontsize=11, color=P["ink"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["terminating", "non-\nterminating"])
        ax.set_ylabel(ylab)
        ax.set_ylim(0, top * 1.32)
        style_axes(ax)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig_mechanism.pdf"); fig.savefig(FIG / "fig_mechanism.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_budget()
    fig_mechanism()
    print("wrote fig_budget.pdf, fig_mechanism.pdf (polished)")
