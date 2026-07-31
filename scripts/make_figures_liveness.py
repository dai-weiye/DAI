"""
Liveness figure (polished): end-to-end completion vs pipeline depth, clean vs
adversarial, naive vs mitigated. Journal style via figstyle.py. Reads results/liveness.json.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
import matplotlib.pyplot as plt
from figstyle import apply_style, style_axes, PALETTE as P

apply_style()
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT/"results/figures"; FIG.mkdir(parents=True, exist_ok=True)
L = json.load(open(ROOT/"results/liveness.json"))
Ks = [1, 2, 3, 5]


def comp(cond):
    return [L["pipelines"][f"K{K}_{cond}"]["completion_rate"] for K in Ks]


def main():
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(Ks, comp("clean"), "-o", color=P["clean"], lw=2.0, ms=7,
            markeredgecolor="white", markeredgewidth=1.0, label="clean")
    ax.plot(Ks, comp("adversarial"), "-o", color=P["adv"], lw=2.0, ms=7,
            markeredgecolor="white", markeredgewidth=1.0, label="adversarial")
    mit_k = [k for k in Ks if k > 1]
    mit_y = [L["mitigation"][f"K{k}_adversarial"]["completion_rate"] for k in mit_k]
    ax.plot(mit_k, mit_y, ":s", color=P["gray"], lw=1.5, ms=5.5,
            markeredgecolor="white", markeredgewidth=1.0,
            label="+ retry (assumed independent)")
    # Measured escape probability from real resamples of stalled items.
    R = json.load(open(ROOT / "results/hedged_retry_real.json"))
    p_ok = R["stage_success_with_retry"]
    ax.plot(mit_k, [p_ok ** k for k in mit_k], "--s", color=P["accent"], lw=1.9, ms=6,
            markeredgecolor="white", markeredgewidth=1.0, label="+ retry (measured)")
    ax.annotate(f"{p_ok**3:.0%}", (3, p_ok ** 3), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.5, color=P["accent"],
                weight="medium")
    for k, y in zip(Ks, comp("adversarial")):
        ax.annotate(f"{y:.0%}", (k, y), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=8.5, color=P["adv"])
    ax.set_xlabel("pipeline depth (number of chained agents)")
    ax.set_ylabel("end-to-end completion rate")
    ax.set_xticks(Ks); ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG/"fig_liveness.pdf"); fig.savefig(FIG/"fig_liveness.png")
    plt.close(fig)
    print("wrote fig_liveness.pdf (polished)")


if __name__ == "__main__":
    main()
