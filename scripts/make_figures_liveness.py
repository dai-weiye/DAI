"""
Liveness figure: end-to-end completion vs pipeline depth, clean vs adversarial,
naive vs mitigated. Nature (NPG) palette. Reads results/liveness.json. Offline.
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
L = json.load(open(ROOT/"results/liveness.json"))
Ks = [1, 2, 3, 5]


def comp(cond, mit=False):
    ys = []
    for K in Ks:
        if mit and K > 1:
            ys.append(L["mitigation"][f"K{K}_{cond}"]["completion_rate"])
        else:
            ys.append(L["pipelines"][f"K{K}_{cond}"]["completion_rate"])
    return ys


def main():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(Ks, comp("clean"), "-o", color=NPG["blue"], lw=1.8, ms=6, label="clean (naive)")
    ax.plot(Ks, comp("adversarial"), "-o", color=NPG["red"], lw=1.8, ms=6,
            label="adversarial (naive)")
    # mitigated adversarial (dashed), K>=2
    mit_k = [k for k in Ks if k > 1]
    mit_y = [L["mitigation"][f"K{k}_adversarial"]["completion_rate"] for k in mit_k]
    ax.plot(mit_k, mit_y, "--s", color=NPG["teal"], lw=1.6, ms=5,
            label="adversarial + retry")
    for k, y in zip(Ks, comp("adversarial")):
        ax.text(k, y-0.06, f"{y:.0%}", ha="center", fontsize=8, color=NPG["red"])
    ax.set_xlabel("pipeline depth (number of chained agents)")
    ax.set_ylabel("end-to-end completion rate")
    ax.set_xticks(Ks); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG/"fig_liveness.pdf"); fig.savefig(FIG/"fig_liveness.png", dpi=150)
    plt.close(fig)
    print("wrote fig_liveness.pdf")


if __name__ == "__main__":
    main()
