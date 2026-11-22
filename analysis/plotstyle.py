"""Shared matplotlib style: validated categorical palette (fixed order), thin marks,
recessive axes/grid, text in ink tokens (never series colours)."""
import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
MODE_COLORS = {1: "#d9e6f7", 2: "#fbe0d4", 3: "#d4efe5", 4: "#fbecc8", 5: "#eeeeec", 6: "#f6d5d5"}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1.6, "legend.fontsize": 8, "legend.frameon": False,
        "font.size": 9, "axes.prop_cycle": mpl.cycler(color=SERIES), "savefig.dpi": 150,
    })
