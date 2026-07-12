#!/usr/bin/env python3
"""Figure 1: high-density corpus audit and reporting-gap summary.

Eight panels keep their manuscript letters. Panels (a)-(c) form one
hierarchical composition block (corpus bar feeding the water-quality and
water-level breakdowns through shaded connectors), (d) shows denominator
refinement on a log scale, (e) exact reporting-rate intervals, (f) audit
reliability, (g) evidence coverage as progress bars, and (h) model-family
prevalence as a full-width dumbbell panel. All manuscript numbers are read
from the flat truth mirror emitted by validate_manuscript.py.

Usage:
  python3 scripts/plot_fig1_corpus_audit.py
  python3 scripts/plot_fig1_corpus_audit.py --check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import NullLocator

from plot_config import (
    A4_WIDTH, FONT_SIZES_A4, PALETTE, apply_plot_style, save_figure,
)


ROOT = Path(__file__).resolve().parents[1]
TRUTH_PATH = ROOT / "data" / "processed" / "truth_table.json"
OUTPUT_PATH = ROOT / "figures" / "fig1_corpus_audit.png"

C1, C2, C3, C4, C5 = PALETTE
GREY_DARK = "#51545B"
GREY = "#92959C"
GREY_LIGHT = "#E7E7EB"
AF = FONT_SIZES_A4

WQ_ROWS = [
    ("Index", "mappability_index_recovery_n", C3),
    ("Mapping", "mappability_genuine_n", C1),
    ("Association", "mappability_hydrochem_n", C2),
    ("Mixed", "mappability_mixed_n", C4),
    ("Unclear", "mappability_unclear_n", C5),
]

WL_ROWS = [
    ("Exogenous", "wl_taskmode_exogenous_n", C2),
    ("Local AR", "wl_taskmode_local_autoregressive_n", C1),
    ("Multi-horizon", "wl_taskmode_multi_horizon_n", C3),
    ("Spatial", "wl_taskmode_spatial_transfer_n", C4),
    ("Storage", "wl_taskmode_storage_n", C5),
    ("Gap fill", "wl_taskmode_gap_filling_n", GREY),
    ("Mixed", "wl_taskmode_mixed_n", GREY_LIGHT),
    ("Unclear", "wl_taskmode_unclear_n", "#D5D5DA"),
]

MODEL_ROWS = [
    ("Tree/boost.", "tree_boosting"),
    ("Neural/deep", "neural_deep"),
    ("SVM/kernel", "svm_kernel"),
    ("Linear/stat.", "linear_statistical"),
    ("Hybrid/fuzzy", "hybrid_fuzzy_other"),
    ("kNN/distance", "knn_distance"),
    ("Unmapped", "unmapped_or_unclear"),
    ("Geostatistical", "geostatistical"),
]

BAR_BOTTOM = -0.31  # panel (a) segment bottom edge in data coordinates


def load_truth() -> dict[str, object]:
    if not TRUTH_PATH.exists():
        raise SystemExit(f"Missing truth mirror: {TRUTH_PATH}")
    return json.loads(TRUTH_PATH.read_text(encoding="utf-8"))


def required_keys() -> set[str]:
    keys = {
        "corpus_n", "mappability_wqm_n", "wl_taskmode_audited_n",
        "corpus_quality_mapping_pct", "corpus_level_timeseries_pct", "corpus_other_pct",
        "mappability_genuine_n", "mappability_index_recovery_n", "mappability_hydrochem_n",
        "double_blind_llm_genuine_n", "double_blind_adjudicated_genuine_n",
        "double_blind_adjudicated_genuine_spatial_holdout_n",
        "double_blind_adjudicated_genuine_spatial_holdout_pct",
        "double_blind_adjudicated_genuine_spatial_holdout_ci_lo",
        "double_blind_adjudicated_genuine_spatial_holdout_ci_hi",
        "wl_taskmode_core_n", "wl_taskmode_core_persistence_n",
        "wl_taskmode_core_persistence_pct", "wl_taskmode_core_persistence_ci_lo",
        "wl_taskmode_core_persistence_ci_hi", "wl_taskmode_field_persistence_n",
        "wl_taskmode_field_persistence_pct", "wl_taskmode_field_persistence_ci_lo",
        "wl_taskmode_field_persistence_ci_hi", "audit_full_schema_n",
        "audit_full_schema_fields_n", "audit_full_schema_agree_pct",
        "double_blind_mappability_agree_pct", "double_blind_mappability_kappa",
        "double_blind_spatial_holdout_agree_pct", "double_blind_spatial_holdout_kappa",
        "audit_spatial_holdout_agree_pct", "audit_persistence_agree_pct",
        "coverage_footprint_two_axis_pct", "manual_audit_dedup_n", "manual_audit_dedup_pct",
        "coverage_confirmed_n", "coverage_confirmed_pct", "model_family_breakdown",
    }
    keys.update(key for _, key, _ in WQ_ROWS if key)
    keys.update(key for _, key, _ in WL_ROWS if key)
    return keys


def validate_truth(t: dict[str, object]) -> None:
    missing = sorted(required_keys().difference(t))
    if missing:
        raise SystemExit(f"Missing Figure 1 truth fields: {missing}")
    values = [float(t[key]) for key in required_keys() if key != "model_family_breakdown"]
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("Figure 1 truth fields contain non-finite values")

    wq_sum = sum(int(t[key]) for _, key, _ in WQ_ROWS)
    wl_sum = sum(int(t[key]) for _, key, _ in WL_ROWS if key)
    if wq_sum != int(t["mappability_wqm_n"]):
        raise SystemExit(f"Water-quality task modes sum to {wq_sum}, not 267")
    if wl_sum != int(t["wl_taskmode_audited_n"]):
        raise SystemExit(f"Water-level task modes sum to {wl_sum}, not 179")
    if int(t["audit_full_schema_fields_n"]) != 15:
        raise SystemExit("Expected 15 fields in the full-schema audit")


def panel_label(fig, ax, label: str, dx: float = -0.055, dy: float = 0.025) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 + dx, pos.y1 + dy, f"({label})",
        fontsize=AF["panel_label"], fontweight="bold", ha="left", va="top",
    )


def panel_a(ax, t):
    """Corpus composition bar; returns segment spans for the connectors."""
    corpus = int(t["corpus_n"])
    counts = [
        int(t["mappability_wqm_n"]),
        int(t["wl_taskmode_audited_n"]),
        corpus - int(t["mappability_wqm_n"]) - int(t["wl_taskmode_audited_n"]),
    ]
    labels = ["Water quality", "Water level", "Other"]
    colors = [C1, C2, C4]
    left = 0.0
    spans = []
    for count, label, color in zip(counts, labels, colors):
        pct = count / corpus * 100
        ax.barh(0, pct, left=left, color=color, height=0.62,
                edgecolor="white", linewidth=1.2)
        emphasize = label in {"Water quality", "Water level"}
        ax.text(left + pct / 2, 0, f"{label}\n{count} ({pct:.1f}%)",
                ha="center", va="center", fontsize=AF["annotation"] - 2,
                color="white" if emphasize else GREY_DARK,
                fontweight="bold" if emphasize else "normal", linespacing=1.2)
        spans.append((left, left + pct))
        left += pct
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return spans


def _breakdown_bars(ax, rows, denominator, t):
    ys = np.arange(len(rows))[::-1]
    values = [int(t[key]) for _, key, _ in rows]
    for y, (label, _, color), value in zip(ys, rows, values):
        ax.barh(y, value, color=color, height=0.62,
                edgecolor="white", linewidth=0.6)
        ax.text(value + max(values) * 0.035, y,
                f"{value} ({value / denominator * 100:.1f}%)",
                ha="left", va="center", fontsize=AF["annotation"] - 2,
                color=GREY_DARK)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlim(0, max(values) * 1.60)
    ax.set_xlabel(f"Papers (of {denominator})")
    ax.tick_params(axis="y", length=0, labelsize=AF["tick_label"] - 1)
    ax.spines["left"].set_visible(True)


def panel_b(ax, t):
    _breakdown_bars(ax, WQ_ROWS, int(t["mappability_wqm_n"]), t)


def panel_c(ax, t):
    _breakdown_bars(ax, WL_ROWS, int(t["wl_taskmode_audited_n"]), t)


def _slope_value(ax, x, y, mode):
    if mode == "above":
        ax.text(x, y * 1.30, str(y), ha="center", va="bottom",
                fontsize=AF["annotation"] - 2, color=GREY_DARK)
    elif mode == "below":
        ax.text(x, y / 1.30, str(y), ha="center", va="top",
                fontsize=AF["annotation"] - 2, color=GREY_DARK)
    elif mode == "right":
        ax.text(x + 0.12, y, str(y), ha="left", va="center",
                fontsize=AF["annotation"] - 2, color=GREY_DARK)
    else:  # "left"
        ax.text(x - 0.12, y, str(y), ha="right", va="center",
                fontsize=AF["annotation"] - 2, color=GREY_DARK)


def panel_d(ax, t):
    x_wq = [0, 1, 2, 3]
    y_wq = [
        int(t["mappability_wqm_n"]),
        int(t["double_blind_llm_genuine_n"]),
        int(t["double_blind_adjudicated_genuine_n"]),
        int(t["double_blind_adjudicated_genuine_spatial_holdout_n"]),
    ]
    x_wl = [0, 2, 3]
    y_wl = [
        int(t["wl_taskmode_audited_n"]),
        int(t["wl_taskmode_core_n"]),
        int(t["wl_taskmode_core_persistence_n"]),
    ]
    ax.plot(x_wq, y_wq, color=C1, marker="o", ms=7, lw=2.2, zorder=3)
    ax.plot(x_wl, y_wl, color=C5, marker="s", ms=7, lw=2.2,
            mec=GREY_DARK, mew=0.7, zorder=3)
    for x, y, mode in zip(x_wq, y_wq, ["above", "above", "below", "above"]):
        _slope_value(ax, x, y, mode)
    for x, y, mode in zip(x_wl, y_wl, ["left", "right", "right"]):
        _slope_value(ax, x, y, mode)
    handles = [
        Line2D([], [], color=C1, marker="o", lw=2.2, label="Water quality"),
        Line2D([], [], color=C5, marker="s", lw=2.2, mec=GREY_DARK, mew=0.7,
               label="Water level"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=AF["legend"] - 2, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.0)
    ax.set_yscale("log")
    ax.set_ylim(1.05, 700)
    ax.set_yticks([2, 10, 50, 250])
    ax.set_yticklabels(["2", "10", "50", "250"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.55, 3.35)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Broad", "Automated", "Adjudicated", "Reported"],
                       fontsize=AF["tick_label"] - 3)
    ax.set_ylabel("Papers")


def panel_e(ax, t):
    rows = [
        ("Holdout", float(t["double_blind_adjudicated_genuine_spatial_holdout_pct"]),
         float(t["double_blind_adjudicated_genuine_spatial_holdout_ci_lo"]),
         float(t["double_blind_adjudicated_genuine_spatial_holdout_ci_hi"]), C1, "o", "6/68"),
        ("Core persistence", float(t["wl_taskmode_core_persistence_pct"]),
         float(t["wl_taskmode_core_persistence_ci_lo"]),
         float(t["wl_taskmode_core_persistence_ci_hi"]), C5, "s", "2/91"),
        ("Field-wide", float(t["wl_taskmode_field_persistence_pct"]),
         float(t["wl_taskmode_field_persistence_ci_lo"]),
         float(t["wl_taskmode_field_persistence_ci_hi"]), C5, "s", "2/179"),
    ]
    ys = [2, 1, 0]
    for y, (label, point, lo, hi, color, marker, ratio) in zip(ys, rows):
        open_marker = label == "Field-wide"
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt=marker,
                    ms=8.5, mfc="white" if open_marker else color, mec=GREY_DARK,
                    mew=0.8, ecolor=color, elinewidth=2.0, capsize=3.5)
        ax.text(hi + 0.7, y, f"{ratio} ({point:.1f}%)", va="center", ha="left",
                fontsize=AF["annotation"] - 2)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_ylim(-0.45, 2.45)
    ax.set_xlim(0, 27)
    ax.set_xlabel("Papers reporting comparator (%)")
    ax.tick_params(axis="y", length=0, labelsize=AF["tick_label"] - 1)
    ax.spines["left"].set_visible(True)


def panel_f(ax, t):
    rows = [
        ("All fields", float(t["audit_full_schema_agree_pct"]), "99 papers × 15 fields"),
        ("Task class", float(t["double_blind_mappability_agree_pct"]),
         f"$\\kappa$={float(t['double_blind_mappability_kappa']):.3f}"),
        ("Holdout labels", float(t["double_blind_spatial_holdout_agree_pct"]),
         f"$\\kappa$={float(t['double_blind_spatial_holdout_kappa']):.3f}"),
        ("Holdout field", float(t["audit_spatial_holdout_agree_pct"]), "98/99"),
        ("Persistence", float(t["audit_persistence_agree_pct"]), "99/99"),
    ]
    ax.axvline(100, color=GREY, lw=0.9, ls=":", zorder=0)
    ys = np.arange(len(rows))[::-1]
    for y, (label, point, note) in zip(ys, rows):
        ax.hlines(y, 90, point, color=C2, lw=2.0)
        ax.plot(point, y, "o", color=C1, ms=7.5, mec=GREY_DARK, mew=0.7)
        ax.text(point + 0.25, y, f"{point:.1f}%", va="center", ha="left",
                fontsize=AF["annotation"] - 2)
        ax.text(90.15, y - 0.28, note, va="top", ha="left",
                fontsize=AF["annotation"] - 4, color=GREY_DARK)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlim(90, 101.5)
    ax.set_xticks([90, 95, 100])
    ax.set_xlabel("Agreement (%)")
    ax.tick_params(axis="y", length=0, labelsize=AF["tick_label"] - 1)
    ax.spines["left"].set_visible(True)


def panel_g(ax, t):
    corpus = int(t["corpus_n"])
    rows = [
        ("Task footprint", float(t["coverage_footprint_two_axis_pct"]),
         int(t["mappability_wqm_n"]) + int(t["wl_taskmode_audited_n"]), C2),
        ("Manual audit", float(t["manual_audit_dedup_pct"]),
         int(t["manual_audit_dedup_n"]), C3),
        ("Evidence-backed", float(t["coverage_confirmed_pct"]),
         int(t["coverage_confirmed_n"]), C1),
    ]
    ys = [2, 1, 0]
    for y, (label, point, n, color) in zip(ys, rows):
        ax.barh(y, 100, color=GREY_LIGHT, height=0.58, zorder=1)
        ax.barh(y, point, color=color, height=0.58, zorder=2)
        ax.text(point - 1.8, y, f"{n}/{corpus} ({point:.1f}%)",
                ha="right", va="center", fontsize=AF["annotation"] - 2,
                color="white" if color == C1 else GREY_DARK,
                fontweight="bold" if color == C1 else "normal", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of corpus (%)")
    ax.tick_params(axis="y", length=0, labelsize=AF["tick_label"] - 1)
    ax.spines["left"].set_visible(True)


def panel_h(ax, t):
    breakdown = t["model_family_breakdown"]
    corpus_n = int(t["corpus_n"])
    wq_n = int(t["mappability_wqm_n"])
    ys = np.arange(len(MODEL_ROWS))[::-1]
    for y, (label, key) in zip(ys, MODEL_ROWS):
        corpus_pct = float(breakdown[key][0]) / corpus_n * 100
        wq_pct = float(breakdown[key][1]) / wq_n * 100
        ax.hlines(y, min(corpus_pct, wq_pct), max(corpus_pct, wq_pct),
                  color=GREY, lw=1.6)
        ax.plot(corpus_pct, y, "o", color=C2, ms=7.5, mec=GREY_DARK, mew=0.6)
        ax.plot(wq_pct, y, "s", color=C1, ms=7.5, mec=GREY_DARK, mew=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in MODEL_ROWS])
    ax.set_xlim(0, 80)
    ax.set_xlabel("Papers using model family (%)")
    ax.tick_params(axis="y", length=0, labelsize=AF["tick_label"] - 1)
    ax.spines["left"].set_visible(True)
    handles = [
        Line2D([], [], color=C2, marker="o", lw=0, label="Full corpus"),
        Line2D([], [], color=C1, marker="s", lw=0, label="Water quality"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=AF["legend"] - 1, handletextpad=0.3, borderaxespad=0.1)


def _connect(fig, ax_a, span, ax_child, color):
    """Shaded connector from a panel (a) segment to its breakdown panel."""
    x0, x1 = span
    to_fig = fig.transFigure.inverted().transform
    p0 = to_fig(ax_a.transData.transform((x0, BAR_BOTTOM)))
    p1 = to_fig(ax_a.transData.transform((x1, BAR_BOTTOM)))
    pos = ax_child.get_position()
    poly = Polygon(
        [p0, p1, (pos.x1, pos.y1 + 0.004), (pos.x0, pos.y1 + 0.004)],
        closed=True, facecolor=color, alpha=0.11, edgecolor="none",
        transform=fig.transFigure, zorder=0.4,
    )
    fig.add_artist(poly)


def build_figure(t: dict[str, object]) -> None:
    apply_plot_style(font_sizes=AF)
    fig = plt.figure(figsize=(A4_WIDTH, 10.8))
    gs = fig.add_gridspec(
        5, 2,
        height_ratios=[0.42, 1.30, 1.18, 1.02, 1.18],
        left=0.175, right=0.97, top=0.975, bottom=0.05,
        hspace=0.62, wspace=0.50,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    ax_f = fig.add_subplot(gs[3, 0])
    ax_g = fig.add_subplot(gs[3, 1])
    ax_h = fig.add_subplot(gs[4, :])

    spans = panel_a(ax_a, t)
    panel_b(ax_b, t)
    panel_c(ax_c, t)
    panel_d(ax_d, t)
    panel_e(ax_e, t)
    panel_f(ax_f, t)
    panel_g(ax_g, t)
    panel_h(ax_h, t)

    _connect(fig, ax_a, spans[0], ax_b, C1)
    _connect(fig, ax_a, spans[1], ax_c, C2)

    for ax, label, dx in [
        (ax_a, "a", -0.055), (ax_b, "b", -0.055), (ax_c, "c", -0.16),
        (ax_d, "d", -0.055), (ax_e, "e", -0.16), (ax_f, "f", -0.055),
        (ax_g, "g", -0.16), (ax_h, "h", -0.055),
    ]:
        panel_label(fig, ax, label, dx=dx)
    save_figure(fig, OUTPUT_PATH.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Validate all Figure 1 truth inputs without plotting.")
    args = parser.parse_args()
    truth = load_truth()
    validate_truth(truth)
    if args.check:
        print(
            f"PASS: Figure 1 inputs validated against {len(truth)} truth fields; "
            "water-quality and water-level task modes sum to their denominators."
        )
        return
    build_figure(truth)


if __name__ == "__main__":
    main()
