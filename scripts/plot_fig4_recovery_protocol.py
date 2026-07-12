#!/usr/bin/env python3
"""Figure 4: deployment-aware attribution, ACR, sensitivity, and index recovery.

The six-panel figure reads every quantitative value from the flat manuscript
truth mirror. It does not recompute experiments or hard-code a truth-schema
field count. Panel (a) is an editorial step strip; panels (b)-(d) share the
seven-row layout with a shaded temporal band; the bottom row holds the
baseline-sensitivity pair (e) and the index-recovery heatmap (f).

Usage:
  python3 scripts/plot_fig4_recovery_protocol.py
  python3 scripts/plot_fig4_recovery_protocol.py --check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

from plot_config import (
    A4_WIDTH, FONT_SIZES_A4, HEATMAP_CMAP, PALETTE, apply_plot_style,
    save_figure,
)


ROOT = Path(__file__).resolve().parents[1]
TRUTH_PATH = ROOT / "data" / "processed" / "truth_table.json"
OUTPUT_PATH = ROOT / "figures" / "fig4_recovery_protocol.png"

C1, C2, C3, C4, C5 = PALETTE
GREY_DARK = "#51545B"
GREY = "#92959C"
GREY_LIGHT = "#E7E7EB"
BAND = "#F4EFF6"  # temporal-group band, low end of the groundwater ramp
AF = FONT_SIZES_A4

ROWS = [
    ("nitrate", "Nitrate (US)", "spatial", "R2"),
    ("fluoride", "Fluoride (global)", "spatial", "AUC-0.5"),
    ("arsenic", "Arsenic (US)", "spatial", "AUC-0.5"),
    ("assam_arsenic", "Arsenic (Assam)", "spatial", "AUC-0.5"),
    ("gems_de", "GEMS Germany", "temporal", "R2"),
    ("ncp_cn", "North China Plain", "temporal", "R2"),
    ("grow_16", "GROW 16 countries", "temporal", "R2"),
]
Y_POS = [6.7, 5.7, 4.7, 3.7, 2.1, 1.1, 0.1]

GROUP_STYLE = {
    "spatial": {"color": C1, "marker": "o"},
    "temporal": {"color": C5, "marker": "s"},
}

# white halo so value labels stay legible across reference lines
HALO = [withStroke(linewidth=2.6, foreground="white")]


def load_truth() -> dict[str, object]:
    if not TRUTH_PATH.exists():
        raise SystemExit(f"Missing truth mirror: {TRUTH_PATH}")
    return json.loads(TRUTH_PATH.read_text(encoding="utf-8"))


def required_keys() -> set[str]:
    suffixes = ("", "_ci_lo", "_ci_hi", "_increment", "_ml_skill", "_baseline_skill")
    keys = {f"acr_{key}{suffix}" for key, _, _, _ in ROWS for suffix in suffixes}
    for dataset in ("nitrate", "fluoride", "assam_arsenic"):
        keys.update(f"acr_{dataset}_ksweep_k{k}_pct" for k in (1, 4, 8, 16))
    keys.update({
        "acr_arsenic_granularity_fine_pct", "acr_arsenic_granularity_coarse_pct",
        "acr_arsenic_granularity_fine_regions_n", "acr_arsenic_granularity_coarse_regions_n",
        "index_recovery_linear_ols_r2", "index_recovery_linear_xgb_r2",
        "index_recovery_linear_logistic_auc", "index_recovery_nonlinear_ols_r2",
        "index_recovery_nonlinear_xgb_r2", "index_recovery_nonlinear_logistic_auc",
    })
    return keys


def validate_truth(t: dict[str, object]) -> None:
    missing = sorted(required_keys().difference(t))
    if missing:
        raise SystemExit(f"Missing Figure 4 truth fields: {missing}")
    if not all(math.isfinite(float(t[key])) for key in required_keys()):
        raise SystemExit("Figure 4 truth fields contain non-finite values")
    positive = 0
    for key, _, _, _ in ROWS:
        point = float(t[f"acr_{key}"])
        lo = float(t[f"acr_{key}_ci_lo"])
        hi = float(t[f"acr_{key}_ci_hi"])
        if not lo <= point <= hi:
            raise SystemExit(f"ACR interval does not contain estimate for {key}")
        positive += float(t[f"acr_{key}_increment"]) > 0
    if positive != 6 or float(t["acr_assam_arsenic"]) <= 100:
        raise SystemExit("Expected six positive increments and Assam recovery above 100%")


def panel_label(
    fig, ax, label: str, dx: float = -0.055, dy: float = 0.025,
    absolute_x: float | None = None, absolute_y: float | None = None,
) -> None:
    pos = ax.get_position()
    x = absolute_x if absolute_x is not None else pos.x0 + dx
    y = absolute_y if absolute_y is not None else pos.y1 + dy
    fig.text(x, y, f"({label})",
             fontsize=AF["panel_label"], fontweight="bold",
             ha="left", va="top")


def panel_a(ax):
    """Editorial step strip: rule, numbered disc, heading, grey body."""
    ax.set_axis_off()
    steps = [
        ("1", C4, "Deployment\ninformation", "Neighbours | last level\nformula inputs"),
        ("2", C3, "Deployment-matched\nvalidation", "Withhold locations or\nmove forecast origins"),
        ("3", C2, "Task-matched\ncomparator", "Identical units and\nskill scale"),
        ("4", C1, "Report skill\n+ model increment", "Switch with horizon\nNo ratio without inputs"),
    ]
    width = 0.214
    for i, (num, color, heading, body) in enumerate(steps):
        x = i * 0.262
        ax.plot([x, x + width], [0.98, 0.98], color=color, lw=3.0,
                transform=ax.transAxes, clip_on=False, solid_capstyle="butt")
        ax.scatter([x + 0.020], [0.78], s=430, color=color, zorder=3,
                   transform=ax.transAxes, clip_on=False, edgecolors="none")
        ax.text(x + 0.020, 0.775, num, transform=ax.transAxes, ha="center",
                va="center", fontsize=AF["annotation"], fontweight="bold",
                color="white" if color in (C1, C2) else GREY_DARK)
        ax.text(x, 0.52, heading, transform=ax.transAxes, ha="left", va="top",
                fontsize=AF["annotation"] - 1, fontweight="bold",
                color=GREY_DARK, linespacing=1.15)
        ax.text(x, 0.17, body, transform=ax.transAxes, ha="left", va="top",
                fontsize=AF["annotation"] - 2, color=GREY, linespacing=1.2)
    for i in range(3):
        x0 = i * 0.262 + width + 0.010
        x1 = (i + 1) * 0.262 - 0.010
        ax.annotate("", xy=(x1, 0.78), xytext=(x0, 0.78),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))


def style_shared_rows(ax, show_labels=False):
    ax.set_ylim(-0.55, 7.55)
    ax.axhspan(-0.55, 2.9, color=BAND, zorder=0)
    ax.set_yticks(Y_POS)
    if show_labels:
        ax.set_yticklabels([row[1] for row in ROWS])
        ax.tick_params(axis="y", length=0, labelleft=True,
                       labelsize=AF["tick_label"] - 2)
    else:
        ax.tick_params(axis="y", length=0, labelleft=False)
    ax.spines["left"].set_visible(False)


def panel_b(ax, t):
    for (key, _, group, _), y in zip(ROWS, Y_POS):
        style = GROUP_STYLE[group]
        baseline = float(t[f"acr_{key}_baseline_skill"])
        model = float(t[f"acr_{key}_ml_skill"])
        ax.hlines(y, min(baseline, model), max(baseline, model),
                  color=style["color"], lw=2.1, alpha=0.75)
        ax.plot(baseline, y, marker=style["marker"], ms=7.0, mfc="white",
                mec=style["color"], mew=1.5)
        ax.plot(model, y, marker=style["marker"], ms=7.0, mfc=style["color"],
                mec=GREY_DARK, mew=0.7)
    style_shared_rows(ax, show_labels=True)
    # group tags anchor the shared-row structure (Fig. 3 in-panel style)
    ax.text(0.02, 7.32, "Spatial", color=GREY, fontweight="bold",
            ha="left", va="center", fontsize=AF["annotation"] - 3)
    ax.text(0.02, 2.55, "Temporal", color=GREY, fontweight="bold",
            ha="left", va="center", fontsize=AF["annotation"] - 3)
    ax.set_xlim(0, 0.88)
    ax.set_xlabel("Normalized skill")
    handles = [
        Line2D([], [], color=GREY_DARK, marker="o", mfc="white", lw=0, label="Baseline"),
        Line2D([], [], color=GREY_DARK, marker="o", mfc=GREY_DARK, lw=0, label="Model"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False,
              fontsize=AF["legend"] - 3, handletextpad=0.2,
              columnspacing=0.6, borderaxespad=0.0)


def panel_c(ax, t):
    ax.axvline(100, color=GREY_DARK, lw=1.0, ls="--", zorder=1)
    for (key, _, group, _), y in zip(ROWS, Y_POS):
        style = GROUP_STYLE[group]
        point = float(t[f"acr_{key}"])
        lo = float(t[f"acr_{key}_ci_lo"])
        hi = float(t[f"acr_{key}_ci_hi"])
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]],
                    fmt=style["marker"], ms=7.0, mfc=style["color"],
                    mec=GREY_DARK, mew=0.7, ecolor=style["color"],
                    elinewidth=1.7, capsize=2.5)
        ax.text(point, y + 0.34, f"{point:.1f}", ha="center", va="bottom",
                fontsize=AF["annotation"] - 4, color=GREY_DARK,
                path_effects=HALO, zorder=4)
    style_shared_rows(ax)
    ax.set_xlim(35, 112)
    ax.set_xticks([40, 60, 80, 100])
    ax.set_xlabel("Recovery (%)")


def panel_d(ax, t):
    ax.axvline(0, color=GREY_DARK, lw=1.0)
    for (key, _, group, _), y in zip(ROWS, Y_POS):
        style = GROUP_STYLE[group]
        point = float(t[f"acr_{key}_increment"])
        ax.hlines(y, 0, point, color=style["color"], lw=2.0)
        ax.plot(point, y, marker=style["marker"], ms=7.0,
                mfc=style["color"], mec=GREY_DARK, mew=0.7)
        text = f"+{point:.2f}" if point >= 0 else f"−{abs(point):.2f}"
        ax.text(point, y + 0.34, text, ha="center", va="bottom",
                fontsize=AF["annotation"] - 4, color=GREY_DARK,
                path_effects=HALO, zorder=4)
    style_shared_rows(ax)
    ax.set_xlim(-0.075, 0.24)
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_xlabel("Model increment")


def panel_e(ax_curve, ax_gran, t):
    ks = [1, 4, 8, 16]
    specs = [
        ("nitrate", "Nitrate", C1, "o"),
        ("fluoride", "Fluoride", C2, "s"),
        ("assam_arsenic", "Assam", C5, "D"),
    ]
    for key, label, color, marker in specs:
        values = [float(t[f"acr_{key}_ksweep_k{k}_pct"]) for k in ks]
        ax_curve.plot(ks, values, color=color, marker=marker, ms=6.5,
                      lw=1.8, label=label, mec=GREY_DARK, mew=0.6)
    ax_curve.axhline(100, color=GREY, lw=0.9, ls="--")
    ax_curve.set_xscale("log", base=2)
    ax_curve.set_xticks(ks)
    ax_curve.set_xticklabels([str(k) for k in ks])
    ax_curve.set_xlim(0.8, 19)
    ax_curve.set_xlabel("IDW neighbours, k")
    ax_curve.set_ylabel("Recovery (%)")
    ax_curve.set_ylim(0, 115)
    ax_curve.legend(loc="lower right", frameon=False, fontsize=AF["legend"] - 3,
                    handlelength=1.2, borderaxespad=0.1)

    fine_n = int(t["acr_arsenic_granularity_fine_regions_n"])
    coarse_n = int(t["acr_arsenic_granularity_coarse_regions_n"])
    fine = float(t["acr_arsenic_granularity_fine_pct"])
    coarse = float(t["acr_arsenic_granularity_coarse_pct"])
    ax_gran.plot([0, 1], [fine, coarse], color=C1, lw=1.8, marker="o", ms=7,
                 mec=GREY_DARK, mew=0.7)
    for x, value in ((0, fine), (1, coarse)):
        ax_gran.text(x, value + 0.8, f"{value:.1f}", ha="center", va="bottom",
                     fontsize=AF["annotation"] - 4, color=GREY_DARK)
    ax_gran.set_xlim(-0.55, 1.55)
    ax_gran.set_xticks([0, 1])
    ax_gran.set_xticklabels([str(fine_n), str(coarse_n)])
    ax_gran.set_xlabel("Baseline regions")
    ax_gran.set_ylim(40, 51)
    ax_gran.set_yticks([40, 45, 50])
    ax_gran.set_ylabel("US arsenic recovery (%)")
    ax_gran.spines["left"].set_visible(True)


def panel_f(ax, t):
    values = np.array([
        [float(t["index_recovery_linear_ols_r2"]),
         float(t["index_recovery_linear_xgb_r2"]),
         float(t["index_recovery_linear_logistic_auc"])],
        [float(t["index_recovery_nonlinear_ols_r2"]),
         float(t["index_recovery_nonlinear_xgb_r2"]),
         float(t["index_recovery_nonlinear_logistic_auc"])],
    ])
    ax.set_box_aspect(1)
    ax.imshow(values, vmin=0.5, vmax=1.0, cmap=HEATMAP_CMAP, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > 0.82 else GREY_DARK
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                    fontsize=AF["annotation"], fontweight="bold", color=color)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["OLS\n$R^2$", "XGBoost\n$R^2$", "Logistic\nmacro-AUC"],
                       fontsize=AF["tick_label"] - 2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["WQI", "CCME"],
                       fontsize=AF["tick_label"] - 2)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(1.2)


def build_figure(t):
    apply_plot_style(font_sizes=AF)
    fig = plt.figure(figsize=(A4_WIDTH, 9.8))

    ax_a = fig.add_axes([0.055, 0.855, 0.91, 0.12])
    middle = fig.add_gridspec(
        1, 3, width_ratios=[1.70, 0.98, 0.90],
        left=0.205, right=0.985, top=0.785, bottom=0.405, wspace=0.17,
    )
    ax_b = fig.add_subplot(middle[0, 0])
    ax_c = fig.add_subplot(middle[0, 1], sharey=ax_b)
    ax_d = fig.add_subplot(middle[0, 2], sharey=ax_b)

    bottom = fig.add_gridspec(
        1, 3, width_ratios=[1.25, 0.55, 0.95],
        left=0.11, right=0.97, top=0.295, bottom=0.055, wspace=0.55,
    )
    ax_e_curve = fig.add_subplot(bottom[0, 0])
    ax_e_gran = fig.add_subplot(bottom[0, 1])
    ax_f = fig.add_subplot(bottom[0, 2])

    panel_a(ax_a)
    panel_b(ax_b, t)
    panel_c(ax_c, t)
    panel_d(ax_d, t)
    panel_e(ax_e_curve, ax_e_gran, t)
    panel_f(ax_f, t)

    aligned_left = 0.012
    bottom_label_y = ax_e_curve.get_position().y1 + 0.025
    for ax, label, dx, absolute_x, absolute_y in [
        (ax_a, "a", -0.045, aligned_left, None),
        (ax_b, "b", -0.20, aligned_left, None),
        (ax_c, "c", -0.05, None, None),
        (ax_d, "d", -0.05, None, None),
        (ax_e_curve, "e", -0.08, aligned_left, bottom_label_y),
        (ax_f, "f", -0.12, None, bottom_label_y),
    ]:
        panel_label(fig, ax, label, dx=dx, absolute_x=absolute_x,
                    absolute_y=absolute_y)

    save_figure(fig, OUTPUT_PATH.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Validate all Figure 4 truth inputs without plotting.")
    args = parser.parse_args()
    truth = load_truth()
    validate_truth(truth)
    if args.check:
        print(f"PASS: Figure 4 required fields validated against {len(truth)} truth fields.")
        return
    build_figure(truth)


if __name__ == "__main__":
    main()
