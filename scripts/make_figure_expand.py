"""
Expanded-sample non-termination figure (fig_expand): grouped clean-vs-adversarial
non-termination bars per dataset, with Wilson 95% CIs, on the ~1195-item expanded run.
Same house style as the other figures. Reads results/rigor_stats_expand.json.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
import matplotlib.pyplot as plt
import numpy as np
from figstyle import apply_style, style_axes, bar_kw, ERRORBAR_KW, pct_axis, PALETTE as P

apply_style()
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
S = json.load(open(ROOT / "results/rigor_stats_expand.json"))["distractor_nonterm"]
DS = ["gsm8k", "math500", "gpqa_diamond"]
LAB = {"gsm8k": "GSM8K", "math500": "MATH-500", "gpqa_diamond": "GPQA-D"}


def main():
    clean = [S[d]["clean"] for d in DS]   # [p, lo, hi]
    adv = [S[d]["adv"] for d in DS]
    x = np.arange(len(DS)); w = 0.34
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for off, arr, col, lb in [(-w/2, clean, P["clean"], "clean"),
                              (w/2, adv, P["adv"], "adversarial (distractor)")]:
        p = [a[0] for a in arr]
        lo = [max(0, a[0]-a[1]) for a in arr]; hi = [max(0, a[2]-a[0]) for a in arr]
        ax.bar(x+off, p, w, label=lb,
               **{k: v for k, v in bar_kw(col).items() if k != "width"},
               yerr=[lo, hi], error_kw=ERRORBAR_KW)
        for xi, pi, a in zip(x+off, p, arr):
            ax.text(xi, a[2]+0.02, f"{pi:.0%}", ha="center", fontsize=8.5, color=col,
                    weight="medium")
    ax.set_xticks(x); ax.set_xticklabels([LAB[v] for v in DS])
    ax.set_ylabel("non-termination rate")
    ax.set_ylim(0, 0.66)
    pct_axis(ax)
    ax.legend(loc="upper left", handlelength=1.1)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_expand.pdf"); fig.savefig(FIG / "fig_expand.png")
    plt.close(fig)
    print("wrote fig_expand.pdf (expanded ~1195-item sample, Wilson CIs)")


if __name__ == "__main__":
    main()
