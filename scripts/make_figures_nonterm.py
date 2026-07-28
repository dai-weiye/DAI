"""
Figures for the NON-TERMINATION paper (honest CASE-B reframe).
Reads results/main_records_8k.jsonl, results/main_records.jsonl (2048), and
results/non_termination_deep.json. NO API cost. Colorblind-safe, print-friendly.
Saves PDF+PNG to results/figures/.

New headline figures:
  fig_nonterm    : non-termination rate, clean vs adversarial, per dataset (8k).
  fig_budget     : budget sensitivity 2048 -> 8192 (the failure PERSISTS at 4x budget).
  fig_mechanism  : in-trace mechanism signals (self-doubt density, verbatim looping)
                   for non-terminating vs terminating traces, with bootstrap CIs.
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
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})
# Nature (NPG) palette; semantic: blue=clean/calm, red=adversarial/danger.
C = {"a": "#4DBBD5", "b": "#E64B35", "c": "#00A087", "d": "#3C5488", "gray": "#8491B4"}

DEEP = json.load(open(ROOT / "results/non_termination_deep.json"))
DS = ["gsm8k", "math500", "gpqa_diamond"]
DSLAB = {"gsm8k": "GSM8K", "math500": "MATH-500", "gpqa_diamond": "GPQA-D"}


def fig_nonterm():
    """Grouped bars: non-termination rate clean vs adversarial per dataset (8k)."""
    rate = DEEP["rate_8k"]
    clean = [rate[f"{d}/clean"]["rate"] for d in DS]
    adv = [rate[f"{d}/adversarial"]["rate"] for d in DS]
    x = np.arange(len(DS)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    b1 = ax.bar(x - w/2, clean, w, label="clean", color=C["a"])
    b2 = ax.bar(x + w/2, adv, w, label="adversarial (distractor)", color=C["b"])
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f"{h:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([DSLAB[d] for d in DS])
    ax.set_ylabel("non-termination rate")
    ax.set_ylim(0, max(adv) + 0.12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    o = DEEP["rate_8k"]["_overall"]
    ax.set_title(f"distractor multiplies non-termination "
                 f"{o['adv_rate']:.0%} vs {o['clean_rate']:.0%} "
                 f"(×{o['distractor_multiplier']:.1f})", fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "fig_nonterm.pdf"); fig.savefig(FIG / "fig_nonterm.png")
    plt.close(fig)


def fig_budget():
    """Non-termination (adversarial) at 2048 vs 8192: it drops but PERSISTS."""
    bs = DEEP["budget_sensitivity"]
    adv_keys = [f"{d}/adversarial" for d in DS]
    r2 = [bs[k]["rate_2k"] for k in adv_keys]
    r8 = [bs[k]["rate_8k"] for k in adv_keys]
    x = np.arange(len(DS))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for i in range(len(DS)):
        ax.plot([0, 1], [r2[i], r8[i]], "-o", color=C["b"], ms=6, lw=1.5)
        ax.text(1.02, r8[i], DSLAB[DS[i]], va="center", fontsize=8, color=C["gray"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["2048\nbudget", "8192\nbudget"])
    ax.set_xlim(-0.25, 1.45)
    ax.set_ylabel("non-termination rate (adversarial)")
    ax.set_ylim(0, max(r2) + 0.08)
    ax.set_title("4× the token budget does not eliminate non-termination",
                 fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "fig_budget.pdf"); fig.savefig(FIG / "fig_budget.png")
    plt.close(fig)


def fig_mechanism():
    """In-trace mechanism signals: non-terminating vs terminating (8k), with CIs.
    Two panels (different scales): self-doubt marker density; verbatim looping ratio."""
    m = DEEP["mechanism"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    specs = [("doubt_density", "self-doubt markers / 1k chars", axes[0]),
             ("loop_ratio", "verbatim 12-gram looping", axes[1])]
    for key, ylab, ax in specs:
        nt = m[key]["nonterm_mean"]; tm = m[key]["term_mean"]
        # error bar from the gap CI applied symmetrically around nonterm mean is not
        # exact per-group; show group means with a bracket annotation of the gap CI.
        bars = ax.bar([0, 1], [tm, nt], color=[C["a"], C["b"]], width=0.6)
        for bar, v in zip(bars, [tm, nt]):
            ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["terminating", "non-\nterminating"],
                                                  fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_ylim(0, max(nt, tm) * 1.25)
        lo, hi = m[key]["ci"]
        sig = "*" if m[key]["significant"] else "n.s."
        ax.set_title(f"gap {m[key]['gap']:.3f} [{lo:.3f},{hi:.3f}] {sig}",
                     fontsize=8)
    fig.suptitle("Non-terminating traces show more self-doubt and more verbatim looping",
                 fontsize=9)
    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig_mechanism.pdf"); fig.savefig(FIG / "fig_mechanism.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_nonterm()
    fig_budget()
    fig_mechanism()
    print(f"wrote non-termination figures to {FIG}:")
    for p in ["fig_nonterm.pdf", "fig_budget.pdf", "fig_mechanism.pdf"]:
        print("  ", p)
