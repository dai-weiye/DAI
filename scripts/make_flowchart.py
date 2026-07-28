"""
Figure 1 overview / pipeline flowchart for the non-termination paper.
Nature (NPG) palette. Pure matplotlib (no TikZ), outputs PDF+PNG to results/figures/.

Visualizes the study in one glance: a question (clean or adversarially distracted)
goes to a reasoning LLM with a fixed token budget; the model either TERMINATES
(emits an answer, which is graded) or NON-TERMINATES (exhausts the budget and emits
nothing). The non-termination branch is the paper's headline failure mode.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)

# NPG (Nature) palette
NPG = {"red": "#E64B35", "blue": "#4DBBD5", "teal": "#00A087", "navy": "#3C5488",
       "salmon": "#F39B7F", "grey": "#8491B4", "ink": "#2B2B2B"}
plt.rcParams.update({"font.size": 9})


def box(ax, x, y, w, h, text, edge, fill, fs=9, weight="normal", tc=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc or NPG["ink"], weight=weight, zorder=3, linespacing=1.25)


def arrow(ax, xy1, xy2, color, label=None, lx=0, ly=0):
    a = FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.6, color=color, zorder=1,
                        connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)
    if label:
        mx, my = (xy1[0]+xy2[0])/2 + lx, (xy1[1]+xy2[1])/2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                color=color, weight="bold", zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis("off")

    # inputs
    box(ax, 0.15, 2.55, 2.15, 1.0, "Clean\nquestion", NPG["blue"], "#EAF6FA", fs=9)
    box(ax, 0.15, 0.65, 2.15, 1.0, "Adversarial\n(+distractor)", NPG["red"], "#FDEBE8", fs=9)

    # reasoner
    box(ax, 3.35, 1.35, 2.75, 1.5,
        "Reasoning LLM\nbudget $B$ tokens\n(answer only\nafter it stops)",
        NPG["navy"], "#ECEFF5", fs=8.5, weight="bold")
    arrow(ax, (2.30, 3.05), (3.35, 2.55), NPG["blue"])
    arrow(ax, (2.30, 1.15), (3.35, 1.75), NPG["red"])

    # outcomes
    box(ax, 7.15, 2.55, 4.7, 1.05,
        "Terminates → answer graded\neffort–error AUROC $0.46$ (chance)",
        NPG["blue"], "#EAF6FA", fs=8.5)
    box(ax, 7.15, 0.55, 4.7, 1.15,
        "NON-TERMINATION: budget gone,\nno answer  —  $3.3\\times$ under distraction,\npersists at $4\\times$ budget",
        NPG["red"], "#FDEBE8", fs=8.5, weight="bold", tc=NPG["red"])

    arrow(ax, (6.10, 2.35), (7.15, 3.05), NPG["blue"], "stops", lx=-0.05, ly=0.30)
    arrow(ax, (6.10, 1.85), (7.15, 1.15), NPG["red"], "loops,\ndoubts", lx=-0.35, ly=0.32)

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_pipeline.pdf"); fig.savefig(FIG / "fig_pipeline.png", dpi=150)
    plt.close(fig)
    print("wrote fig_pipeline.pdf/.png")


if __name__ == "__main__":
    main()
