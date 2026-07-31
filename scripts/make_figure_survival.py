"""Reasoning-length survival analysis (offline, no API cost).

Non-termination is right-censored data: for a run that hits the cap we only know the
reasoning would have run *at least* that long. Treating it as such lets us say something
sharper than "the failure persists at 4x the budget" -- we can estimate how much budget
would actually be needed.

Panel (a) Kaplan-Meier survival of completion length, clean vs adversarial, censored at
          the 8192 cap, on the large replication sample (n=1195). S(cap) is exactly the
          non-termination rate.
Panel (b) Observed GPQA-adversarial non-termination at three real budgets (2048 / 8192 /
          16384) with Wilson intervals, and a power-law fit in the budget. The fit gives
          the budget at which the rate would fall to 5%, which is the quantitative answer
          to "why not just raise the cap?".
"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import apply_style, PALETTE, FILL, style_axes, pct_axis
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "paper"
CAP = 8192

# GPQA-Diamond adversarial, real runs at three caps (Table/Generalization in the paper).
BUDGETS = np.array([2048, 8192, 16384], dtype=float)
K = np.array([42, 29, 6])      # non-terminating runs
N = np.array([60, 60, 20])     # runs at that budget


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def km(times, events):
    """Kaplan-Meier survival. events=1 terminated (event), 0 censored at the cap."""
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    n_at_risk = len(t)
    ts, ss = [0.0], [1.0]
    s = 1.0
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and t[j] == t[i]:
            j += 1
        d = int(e[i:j].sum())
        if d > 0 and n_at_risk > 0:
            s *= (1 - d / n_at_risk)
        ts.append(float(t[i])); ss.append(s)
        n_at_risk -= (j - i)
        i = j
    return np.array(ts), np.array(ss)


def main():
    apply_style()
    # The *_clean file is the one re-scored with the official extractor and grader, so
    # it is what the replication numbers in the paper are computed from.
    recs = [json.loads(l) for l in open(ROOT / "results/expand_records_clean.jsonl")]
    recs = [r for r in recs if r.get("completion_tokens")]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85))

    # --- (a) Kaplan-Meier survival of reasoning length ---
    ax = axes[0]
    summary = {}
    # adversarial fill goes underneath so the clean band stays visible
    for cond, color, fill, z in (("adversarial", PALETTE["adv"], FILL["adv"], 0),
                                 ("clean", PALETTE["clean"], FILL["clean"], 1)):
        sub = [r for r in recs if r["condition"] == cond]
        times = np.array([min(r["completion_tokens"], CAP) for r in sub], dtype=float)
        events = np.array([0 if r["nonterm"] else 1 for r in sub])
        ts, ss = km(times, events)
        ax.fill_between(ts, 0, ss, step="post", color=fill, alpha=0.9, zorder=z)
        ax.step(ts, ss, where="post", color=color, lw=2.0, zorder=3,
                label=f"{cond} (n={len(sub)})")
        tail = float(ss[-1])
        summary[cond] = {"n": len(sub), "S_at_cap": tail,
                         "median": float(np.median(times))}
        ax.annotate(f"{tail:.0%}", xy=(CAP, tail), xytext=(-4, 6),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=color, weight="medium", zorder=4)
    ax.axvline(CAP, color=PALETTE["muted"], lw=0.9, ls=(0, (3, 2)))
    ax.text(CAP, 1.02, "budget", ha="right", va="bottom", fontsize=8.5,
            color=PALETTE["muted"])
    ax.set_xlabel("completion tokens $t$")
    ax.set_ylabel("$P(\\mathrm{length} > t)$")
    ax.set_title("(a) Survival of reasoning length", loc="left")
    ax.set_xlim(0, CAP * 1.02); ax.set_ylim(0, 1.05)
    pct_axis(ax); style_axes(ax, ygrid=True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93))

    # --- (b) how much budget would it actually take? ---
    ax = axes[1]
    rates = K / N
    los = np.array([wilson(k, n)[1] for k, n in zip(K, N)])
    his = np.array([wilson(k, n)[2] for k, n in zip(K, N)])
    ax.errorbar(BUDGETS, rates, yerr=[rates - los, his - rates], fmt="o",
                color=PALETTE["adv"], ms=7, lw=0, elinewidth=1.1, capsize=3.2,
                capthick=1.0, ecolor="#3F4650", markeredgecolor="white",
                markeredgewidth=1.1, zorder=3, label="observed (GPQA, adversarial)")

    # power law rate ~ a * budget^b, weighted by n
    b, loga = np.polyfit(np.log(BUDGETS), np.log(rates), 1, w=np.sqrt(N))
    a = np.exp(loga)
    grid = np.logspace(np.log10(1500), np.log10(4e6), 200)
    ax.plot(grid, a * grid ** b, color=PALETTE["navy"], lw=1.6, ls=(0, (4, 2)),
            zorder=2, label=f"power-law fit ($\\propto B^{{{b:.2f}}}$)")
    b_at_5 = float((0.05 / a) ** (1 / b))
    ax.axhline(0.05, color=PALETTE["muted"], lw=0.9, ls=":")
    ax.plot([b_at_5], [0.05], marker="*", ms=13, color=PALETTE["accent"],
            markeredgecolor="white", markeredgewidth=0.9, zorder=4)
    ax.annotate(f"5% needs\n$\\approx${b_at_5/1e6:.1f}M tokens", xy=(b_at_5, 0.05),
                xytext=(-6, 14), textcoords="offset points", ha="right", fontsize=8.5,
                color=PALETTE["accent"], weight="medium")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("token budget $B$")
    ax.set_ylabel("non-termination rate")
    ax.set_title("(b) Budget required to remove it", loc="left")
    ax.set_ylim(0.02, 1.0)
    style_axes(ax, ygrid=False)
    ax.grid(which="both", axis="both", visible=True, color=PALETTE["grid"], lw=0.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower left", fontsize=8.2)

    fig.tight_layout(w_pad=1.6)
    for name in ("fig_survival.pdf",):
        fig.savefig(OUT / name)
    summary["powerlaw"] = {"exponent": float(b), "coef": float(a),
                           "budget_for_5pct": b_at_5,
                           "observed": {int(x): {"k": int(k), "n": int(n), "rate": float(r)}
                                        for x, k, n, r in zip(BUDGETS, K, N, rates)}}
    (ROOT / "results/survival_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT/'fig_survival.pdf'}")


if __name__ == "__main__":
    main()
