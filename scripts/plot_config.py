"""
Plot configuration for PROJECT_NAME.

Two themes available:
  - "macaron"     : low-saturation pastels, used in PMS-AOPs (JHM)
  - "nature_water": Nature colorblind-safe palette, used in ML Water (Nature Water)
  - "pillar"      : 5 pillar mapping, used in ML4Env (ES&T)
  - "groundwater" : indigo-to-blossom 5-colour ramp, used in this manuscript
                    (Nature Water Analysis, groundwater ML generalization gap)

Switch by setting THEME below.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

THEME = "groundwater"

# =============================================================================
# Color Palettes — three themes, switch by THEME
# =============================================================================

THEMES = {
    "macaron": {
        "categorical": [
            "#FFB6C1", "#87CEEB", "#98D8C8", "#F7DC6F",
            "#BB8FCE", "#85C1E2", "#F8C471", "#82E0AA",
            "#F0B27A", "#AED6F1",
        ],
        "warning": "#E07B6C",
        "positive": "#7CB88A",
        "heatmap": "Blues",
    },
    "nature_water": {
        "categorical": [
            "#0077BB", "#33BBEE", "#009988", "#EE7733",
            "#CC3311", "#EE3377", "#BBBBBB",
        ],
        "warning": "#CC3311",
        "positive": "#009988",
        "heatmap": "Blues",
    },
    "pillar": {
        "categorical": [
            "#7EAAC4", "#A3C9A8", "#E8B87C", "#C49BBB",
            "#D4897C", "#8DBEB5", "#D4B96A", "#9BAFD4",
            "#C7A07C", "#A8C5D6",
        ],
        "warning": "#E07B6C",
        "positive": "#7CB88A",
        "heatmap": LinearSegmentedColormap.from_list(
            "pillar_div", ["#D4897C", "#FFFFFF", "#A3C9A8"]
        ),
    },
    # Indigo-to-blossom 5-colour ramp (user-specified, 2026-07-08).
    # C1 deepest indigo = primary/emphasis/kill colour; ramp lightens to C5.
    "groundwater": {
        "categorical": [
            "#5A5FA3",  # C1 low-saturation deep indigo — primary series / emphasis
            "#8F98C9",  # C2 soft misty blue — secondary series
            "#BFC0DE",  # C3 pale lavender — auxiliary data
            "#E0D2E5",  # C4 light mauve — weak-contrast background
            "#E7CBE0",  # C5 blossom pink — accent / highlight points
        ],
        "warning": "#5A5FA3",   # emphasis = deepest indigo (the reporting-gap kill colour)
        "positive": "#8F98C9",
        "heatmap": LinearSegmentedColormap.from_list(
            "groundwater_seq", ["#F4EFF6", "#5A5FA3"]
        ),
    },
}

PALETTE = THEMES[THEME]["categorical"]
WARNING = THEMES[THEME]["warning"]
POSITIVE = THEMES[THEME]["positive"]
HEATMAP_CMAP = THEMES[THEME]["heatmap"]

# =============================================================================
# Figure Sizes (inches; matches Nature/ACS column widths)
# =============================================================================

FIGURE_SIZE_SINGLE = (3.5, 2.8)
FIGURE_SIZE_1_5 = (5.5, 4.0)
FIGURE_SIZE_DOUBLE = (7.2, 5.0)
FIGURE_SIZE_FULL = (7.2, 9.0)
# A4-width figures (user standard for this manuscript, 2026-07-08): full A4
# page width is 8.27 in; usable width inside typical margins is ~7.5 in. All
# main figures in this manuscript are drawn at A4 width with large, legible fonts.
A4_WIDTH = 8.27
FIGURE_SIZE_A4 = (8.27, 5.5)
FIGURE_SIZE_A4_TALL = (8.27, 9.5)
DPI = 300

# =============================================================================
# Font Sizes — preview vs submission
# =============================================================================

FONT_SIZES = {
    "panel_label": 10,
    "title": 10,
    "axis_label": 9,
    "tick_label": 8,
    "legend": 7,
    "annotation": 7,
}

# Large-font set for A4-width main figures (user standard, 2026-07-08): every
# label must read clearly at print size. Use this with FIGURE_SIZE_A4* by
# passing it to apply_plot_style(font_sizes=FONT_SIZES_A4).
FONT_SIZES_A4 = {
    "panel_label": 16,
    "title": 15,
    "axis_label": 14,
    "tick_label": 12,
    "legend": 12,
    "annotation": 12,
}

LINE_WIDTHS = {
    "spine": 0.8,
    "grid": 0.5,
    "main": 1.5,
    "secondary": 1.0,
}

# =============================================================================
# Output
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def apply_plot_style(font_sizes=None):
    fs = font_sizes or FONT_SIZES
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": fs["tick_label"],
        "axes.titlesize": fs["title"],
        "axes.labelsize": fs["axis_label"],
        "xtick.labelsize": fs["tick_label"],
        "ytick.labelsize": fs["tick_label"],
        "legend.fontsize": fs["legend"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": LINE_WIDTHS["spine"],
        "axes.grid": False,
        "figure.dpi": DPI,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": DPI,
    })


def add_panel_label(ax, label, font_sizes=None):
    """Add a bold panel label (e.g. (a)) to the top-left of an axis.

    The label is normalized to the parenthesized form: a bare letter such as
    "a" is rendered as "(a)"; pass "(a)" and it is left untouched. This keeps
    panel labels parenthesized regardless of how the caller spells them.
    """
    fs = font_sizes or FONT_SIZES
    text = label.strip()
    if not text.startswith("("):
        text = f"({text})"
    ax.text(-0.12, 1.05, text, transform=ax.transAxes,
            fontsize=fs["panel_label"], fontweight="bold",
            va="top", ha="left")


def save_figure(fig, filename, dpi=None):
    out = FIGURE_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi or DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)
