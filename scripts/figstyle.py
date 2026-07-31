"""
Shared journal-quality matplotlib style for all figures. Import and call apply_style()
at the top of every figure script so the whole paper reads as one polished system.

Design choices (Nature/venue-grade):
- Muted, desaturated palette (still colorblind-distinguishable) instead of saturated RGB.
- Clean typography, hairline spines, light dotted y-grid, generous whitespace.
- Thin capped error bars; slim bars with soft edges.
- Fonts embedded as TrueType (Type 42) so PDFs pass camera-ready font checks.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Refined palette: desaturated, harmonious. clean=cool slate-blue, adversarial=muted coral.
PALETTE = {
    "clean":   "#4C7FA3",   # muted steel blue (slightly deepened for print contrast)
    "adv":     "#D2695E",   # muted coral (not fire-engine red)
    "accent":  "#5B9E86",   # sage green
    "navy":    "#3C5488",   # deep navy for emphasis
    "gold":    "#C9A24B",   # warm ochre, 4th series when needed
    "gray":    "#9AA3B2",   # soft gray
    "ink":     "#26292E",   # near-black text
    "muted":   "#6B7280",   # secondary label gray
    "grid":    "#E6E9EF",   # very light grid
    "edge":    "#FFFFFF",   # white bar edge for separation
}
# soft fills for boxes / shaded regions (pair with the like-named stroke color)
FILL = {
    "clean":  "#EAF1F6",
    "adv":    "#FBEDEA",
    "accent": "#EAF4EF",
    "navy":   "#EDEFF4",
}
# backward-compatible keys used by existing scripts
PALETTE["a"] = PALETTE["clean"]
PALETTE["b"] = PALETTE["adv"]
PALETTE["c"] = PALETTE["accent"]
PALETTE["d"] = PALETTE["navy"]


def apply_style():
    plt.rcParams.update({
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "medium",
        "axes.labelsize": 10.5,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.labelcolor": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        # spines: keep only left+bottom, hairline
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#4B525E",
        # a touch of breathing room between axis labels and ticks
        "axes.labelpad": 4.5,
        # grid: light solid horizontal only, sits behind data
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        # ticks: short, outward, subtle
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.pad": 3.5,
        "ytick.major.pad": 3.0,
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.3,
        "legend.labelspacing": 0.4,
        # lines / markers
        "lines.solid_capstyle": "round",
        "lines.antialiased": True,
        "patch.antialiased": True,
        # output quality + font embedding for camera-ready
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,   # embed TrueType (editable, passes font checks)
        "ps.fonttype": 42,
        "pdf.compression": 6,
    })


def style_axes(ax, ygrid=True):
    """Per-axes touch-ups: grid only on y, no x-grid, spines trimmed and softened."""
    ax.grid(axis="y" if ygrid else "both", which="major")
    ax.grid(axis="x", which="both", visible=False)
    ax.tick_params(length=2.8)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#4B525E")


def bar_kw(color):
    """Consistent slim, soft-edged bars."""
    return dict(color=color, edgecolor=PALETTE["edge"], linewidth=0.9, width=0.62)


def line_kw(color, marker="o"):
    """Consistent line+marker series with a white marker halo for separation."""
    return dict(color=color, lw=2.0, marker=marker, ms=6.5,
                markeredgecolor="white", markeredgewidth=1.1,
                solid_capstyle="round", solid_joinstyle="round")


def label_bars(ax, xs, vs, fmt="{:.0%}", dy=None, color=None, fontsize=8.5,
               weight="medium"):
    """Place value labels centered just above each bar top."""
    if dy is None:
        top = max(vs) if len(vs) else 1.0
        dy = top * 0.02
    for x, v in zip(xs, vs):
        ax.text(x, v + dy, fmt.format(v), ha="center", va="bottom",
                fontsize=fontsize, color=color or PALETTE["ink"], weight=weight)


def pct_axis(ax, which="y"):
    """Format an axis as whole-number percentages."""
    f = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(f)


ERRORBAR_KW = dict(elinewidth=1.0, capsize=3.0, capthick=1.0, ecolor="#3F4650")
