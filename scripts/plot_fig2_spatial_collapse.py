#!/usr/bin/env python3
"""
Figure 2 - Random-split skill collapses under spatial holdout (Results 2.2).

Per-fold density redesign (v2, 2026-07-09). Eight panels, every panel plots
ALL folds, medians are only overlaid as bold markers. One claim -> one panel:

  (a) [full width] Regression targets per-fold collapse cloud
      nitrate (53 random + 53 spatial), Mississippi specific conductance and
      chloride (21+21 each); negative-R2 region shaded; matched k-fold random
      protocol throughout (single-protocol rule).
  (b) [full width] Classification targets per-fold collapse cloud
      arsenic US (13+13, leave-one-ecoregion), arsenic Assam (14+14, 0.15deg
      block), global fluoride (6+6, leave-one-continent); AUC 0.5 chance line.
  (c) Per-fold ML vs zero-learning free baseline scatter with y=x diagonal
      (random regime; IDW for the five coordinate targets, categorical
      region-mean for arsenic US, drawn as open markers).
  (d) Five-model x six-target random-to-spatial gap heatmap (30 cells, all
      positive).
  (e) Gap vs block size (nitrate 2-8deg, Assam 0.1-0.25deg, Mississippi
      0.5-2.0deg, fluoride 5deg-block vs leave-one-continent).
  (f) Mechanism lollipop: local-density and nearest-distance Spearman per
      coordinate target; +/-0.2 negligible band; only fluoride exits the band.
  (g) Fluoride AP-lift bar summary (truth-table medians, fold-level IQR),
      random vs leave-one-continent (log y, lift=1 chance line).
  (h) Arsenic US AP-lift bar summary, random vs leave-one-ecoregion (log y,
      shared style with g).

No in-panel titles; all wording lives in manuscript/drafts/figure_captions.md.
Summary numbers (medians, gaps, recovery percentages) are read from
truth_table.json (plot-reads-truth-table); per-fold arrays come from the
source CSVs (utf-8-sig: several carry a BOM on the first column).

Usage:
  python3 scripts/plot_fig2_spatial_collapse.py --panel a   # single-panel preview
  python3 scripts/plot_fig2_spatial_collapse.py             # assemble full figure
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import (
    apply_plot_style, add_panel_label, save_figure,
    PALETTE, THEMES, HEATMAP_CMAP, FONT_SIZES_A4, A4_WIDTH,
)

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
TRUTH = PROC / "truth_table.json"

C1, C2, C3, C4, C5 = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4]
PANEL_E_TEAL = THEMES["nature_water"]["categorical"][2]
PANEL_E_ORANGE = THEMES["nature_water"]["categorical"][3]
GREY = "#9AA0A6"
GREY_DARK = "#5F6368"
NEG_SHADE = "#F6E8E8"      # light wash for the worse-than-mean region
AF = FONT_SIZES_A4
JITTER_SEED = 1182         # match the experiment seed for reproducible jitter

# Shared target markers (consistent identity across panels a, b, c, e).
TARGET_MARKER = {
    "nitrate": "o",
    "ms_spc": "v",
    "ms_chloride": "P",
    "arsenic_us": "s",
    "assam": "D",
    "fluoride": "^",
}
TARGET_LABEL = {
    "nitrate": "Nitrate (US)",
    "ms_spc": "Sp. conductance (Miss.)",
    "ms_chloride": "Chloride (Miss.)",
    "arsenic_us": "Arsenic (US)",
    "assam": "Arsenic (Assam)",
    "fluoride": "Fluoride (global)",
}


# =============================================================================
# Data loaders
# =============================================================================

def load_truth():
    return json.load(open(TRUTH))


def _read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def regression_folds():
    """Per-fold R2, matched random k-fold vs spatial block, three targets."""
    nit_rand = [float(r["r2"]) for r in
                _read_csv(PROC / "usgs_nitrate" / "random_kfold_matched_fold_detail.csv")]
    nit_spat = [float(r["r2"]) for r in
                _read_csv(PROC / "usgs_nitrate" / "spatial_block_fold_detail.csv")]
    ms = _read_csv(PROC / "mississippi_salinity_spatial_full_gpu" / "fold_results.csv")

    def ms_r2(target, regime):
        return [float(r["r2"]) for r in ms
                if r["target"] == target and r["regime"] == regime
                and r["model"] == "xgboost"]

    return {
        "nitrate": (nit_rand, nit_spat),
        "ms_spc": (ms_r2("specific_conductance", "random_kfold"),
                   ms_r2("specific_conductance", "spatial_block")),
        "ms_chloride": (ms_r2("chloride", "random_kfold"),
                        ms_r2("chloride", "spatial_block")),
    }


def classification_folds():
    """Per-fold ROC-AUC, matched random k-fold vs spatial holdout."""
    ars = _read_csv(PROC / "usgs_arsenic" / "pr_auc_fold_detail.csv")
    asm = _read_csv(PROC / "assam_arsenic" / "spatial_block_fold_detail.csv")
    flu = _read_csv(PROC / "global_fluoride" / "rare_event_fold_detail.csv")
    return {
        "arsenic_us": (
            [float(r["roc_auc"]) for r in ars if r["regime"] == "random_kfold_matched"],
            [float(r["roc_auc"]) for r in ars if r["regime"] == "leave_one_ecoregion"]),
        "assam": (
            [float(r["ml_auc"]) for r in asm if r["regime"] == "random_kfold"],
            [float(r["ml_auc"]) for r in asm if r["regime"] == "spatial_block"]),
        "fluoride": (
            [float(r["roc_auc"]) for r in flu if r["regime"] == "random_kfold"],
            [float(r["roc_auc"]) for r in flu if r["regime"] == "leave_one_continent"]),
    }


def baseline_folds():
    """Per-fold (baseline_skill, ml_skill) pairs under the random regime.

    Skill convention matches the ACR table: R2 for regression, AUC - 0.5 for
    classification (both are 'skill above the uninformed reference').
    The four acr_folds.csv files already carry these as base_skill/ml_skill;
    Mississippi pairs are matched fold-by-fold from fold_results.csv
    (idw_k8 vs xgboost, random_kfold regime).
    """
    out = {}
    for key, d in (("nitrate", "usgs_nitrate"), ("fluoride", "global_fluoride"),
                   ("arsenic_us", "usgs_arsenic"), ("assam", "assam_arsenic")):
        rows = _read_csv(PROC / d / "acr_folds.csv")
        out[key] = ([float(r["base_skill"]) for r in rows],
                    [float(r["ml_skill"]) for r in rows])
    ms = _read_csv(PROC / "mississippi_salinity_spatial_full_gpu" / "fold_results.csv")
    for key, target in (("ms_spc", "specific_conductance"),
                        ("ms_chloride", "chloride")):
        idw = {r["fold"]: float(r["r2"]) for r in ms
               if r["target"] == target and r["regime"] == "random_kfold"
               and r["model"] == "idw_k8"}
        xgb = {r["fold"]: float(r["r2"]) for r in ms
               if r["target"] == target and r["regime"] == "random_kfold"
               and r["model"] == "xgboost"}
        folds = sorted(idw.keys() & xgb.keys())
        out[key] = ([idw[f] for f in folds], [xgb[f] for f in folds])
    return out


# =============================================================================
# Shared drawing helpers
# =============================================================================

def _collapse_cloud(ax, targets, fold_data, medians, ylim, ylabel,
                    chance_line=None):
    """Paired random-vs-spatial per-fold strip chart with median slope.

    targets   : ordered target keys
    fold_data : {key: (random_folds, spatial_folds)}
    medians   : {key: (random_median, spatial_median)}  # from truth_table
    """
    rng = np.random.default_rng(JITTER_SEED)
    lo, hi = ylim
    dx = 0.22

    if lo < 0:
        ax.axhspan(lo, 0, color=NEG_SHADE, zorder=0)
        ax.axhline(0, color=GREY_DARK, lw=0.8, ls="--", zorder=1)
    if chance_line is not None:
        ax.axhline(chance_line, color=GREY_DARK, lw=0.8, ls="--", zorder=1)

    for i, key in enumerate(targets):
        rand, spat = fold_data[key]
        m = TARGET_MARKER[key]
        for vals, x0, face, edge in (
                (rand, i - dx, "white", GREY),
                (spat, i + dx, C1, C1)):
            vals = np.asarray(vals, dtype=float)
            xs = x0 + rng.uniform(-0.09, 0.09, size=len(vals))
            shown = np.clip(vals, lo, hi)
            ax.scatter(xs, shown, s=34, marker=m, facecolor=face,
                       edgecolor=edge, linewidth=1.0, alpha=0.85, zorder=3)
            # folds below the axis floor: re-mark + print the true value
            for v, x in zip(vals, xs):
                if v < lo:
                    ax.scatter([x], [lo], s=52, marker="v", facecolor=edge,
                               edgecolor=edge, zorder=4)
                    ax.annotate(f"{v:.1f}", (x, lo),
                                textcoords="offset points", xytext=(7, 3),
                                fontsize=AF["annotation"] - 2, color=edge)
        # median slope segment + bold ticks + value labels
        mr, msp = medians[key]
        ax.plot([i - dx, i + dx], [mr, msp], color=GREY_DARK, lw=1.2, zorder=5)
        ax.hlines(mr, i - dx - 0.14, i - dx + 0.14, color=GREY_DARK, lw=3.0,
                  zorder=6)
        ax.hlines(msp, i + dx - 0.14, i + dx + 0.14, color=C1, lw=3.0,
                  zorder=6)
        ax.annotate(f"{mr:.2f}", (i - dx - 0.17, mr), ha="right", va="center",
                    fontsize=AF["annotation"], color=GREY_DARK)
        ax.annotate(f"{msp:.2f}", (i + dx + 0.17, msp), ha="left", va="center",
                    fontsize=AF["annotation"], color=C1, fontweight="bold")

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([TARGET_LABEL[k] for k in targets],
                       fontsize=AF["tick_label"])
    ax.set_xlim(-0.65, len(targets) - 0.35)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel, fontsize=AF["axis_label"])
    ax.tick_params(axis="y", labelsize=AF["tick_label"])


def _regime_legend(ax, loc="upper right"):
    # short forms only; full protocol wording lives in the figure caption
    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor="white",
               markeredgecolor=GREY, markersize=7, label="Random"),
        Line2D([], [], marker="o", ls="none", markerfacecolor=C1,
               markeredgecolor=C1, markersize=7, label="Spatial"),
        Line2D([], [], color=GREY_DARK, lw=3.0, label="Median"),
    ]
    ax.legend(handles=handles, loc=loc, fontsize=AF["legend"], frameon=False,
              handletextpad=0.5, borderaxespad=0.2)


# =============================================================================
# Panels
# =============================================================================

def panel_a(ax, t):
    """Regression targets per-fold collapse cloud."""
    fd = regression_folds()
    medians = {
        "nitrate": (t["nitrate_random_r2"], t["nitrate_spatial_block_r2"]),
        "ms_spc": (t["mississippi_specific_conductance_random_r2"],
                   t["mississippi_specific_conductance_spatial_block_r2"]),
        "ms_chloride": (t["mississippi_chloride_random_r2"],
                        t["mississippi_chloride_spatial_block_r2"]),
    }
    _collapse_cloud(ax, ["nitrate", "ms_spc", "ms_chloride"], fd, medians,
                    ylim=(-1.0, 1.0), ylabel="$R^2$ (per fold)")
    # no in-panel prose: reference-line meanings live in the caption
    _regime_legend(ax, loc="lower left")


def panel_b(ax, t):
    """Classification targets per-fold collapse cloud."""
    fd = classification_folds()
    medians = {
        "arsenic_us": (t["arsenic_random_auc"], t["arsenic_ecoregion_auc"]),
        "assam": (t["assam_arsenic_random_auc"],
                  t["assam_arsenic_spatial_block_auc"]),
        "fluoride": (t["fluoride_random_auc"], t["fluoride_continent_auc"]),
    }
    _collapse_cloud(ax, ["arsenic_us", "assam", "fluoride"], fd, medians,
                    ylim=(0.25, 1.0), ylabel="ROC-AUC (per fold)",
                    chance_line=0.5)
    # no in-panel prose: chance-line meaning lives in the caption
    _regime_legend(ax, loc="lower left")


def panel_c(ax, t):
    """Per-fold ML vs free-baseline skill scatter with y=x diagonal.

    Random regime. IDW baseline for the five coordinate targets (ramp-coloured
    filled markers); the coordinate-free arsenic US target uses a categorical
    region-mean baseline, drawn open grey to flag the different baseline form.
    Points hug the diagonal = the free baseline matches the fitted model.
    Recovery percentages live in the caption (from truth_table), not on the
    figure.
    """
    bf = baseline_folds()
    lim_lo, lim_hi = -0.05, 1.0
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=GREY_DARK, lw=1.0,
            ls="--", zorder=1)

    # ramp colours chosen so overlapping clouds get maximally distinct shades:
    # the mid cluster holds nitrate / assam / fluoride, the top-right cluster
    # holds the two Mississippi targets.
    face = {"nitrate": C1, "assam": C3, "fluoride": C5,
            "ms_spc": C2, "ms_chloride": C4}
    order = ["nitrate", "ms_spc", "ms_chloride", "assam", "fluoride",
             "arsenic_us"]
    for key in order:
        base, ml = bf[key]
        m = TARGET_MARKER[key]
        if key == "arsenic_us":
            fc, ec = "white", GREY_DARK
        else:
            fc, ec = face[key], GREY_DARK
        ax.scatter(base, ml, s=42, marker=m, facecolor=fc, edgecolor=ec,
                   linewidth=0.8, alpha=0.9, zorder=3)

    # short legend forms, no recovery numbers (both live in the caption)
    short = {
        "nitrate": "Nitrate", "ms_spc": "Sp. cond.",
        "ms_chloride": "Chloride", "assam": "Assam arsenic",
        "fluoride": "Fluoride", "arsenic_us": "US arsenic",
    }
    handles = [
        Line2D([], [], marker=TARGET_MARKER[k], ls="none",
               markerfacecolor="white" if k == "arsenic_us" else face[k],
               markeredgecolor=GREY_DARK, markersize=7, label=short[k])
        for k in order]
    # lower-right region is empty after the diagonal split: keep legend away
    # from the main cloud.
    ax.legend(handles=handles, loc="lower right", fontsize=AF["legend"] - 2,
              bbox_to_anchor=(1.08, -0.03),
              frameon=False, handletextpad=0.4, borderaxespad=0.1,
              labelspacing=0.3)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Zero-learning baseline skill (per fold)",
                  fontsize=AF["axis_label"])
    ax.set_ylabel("Model skill (per fold)", fontsize=AF["axis_label"])


def panel_d(ax, t):
    """Five-model x six-target random-to-spatial gap heatmap (30 cells)."""
    rows = _read_csv(PROC / "method3_spatial_extended_panel_gpu.csv")
    model_key = {   # csv model id -> family display slot
        "linear": "Linear",
        "random_forest_xgbrf_gpu": "Random\nforest",
        "xgboost_gpu": "Gradient\nboosting",
        "hist_gbdt": "Histogram\nGBDT",
        "mlp_torch_gpu": "Neural\nnetwork",
    }
    model_order = ["linear", "random_forest_xgbrf_gpu", "xgboost_gpu",
                   "hist_gbdt", "mlp_torch_gpu"]
    ds_key = {"nitrate": "nitrate", "mississippi_specific_conductance": "ms_spc",
              "mississippi_chloride": "ms_chloride", "arsenic": "arsenic_us",
              "assam_arsenic": "assam", "fluoride": "fluoride"}
    ds_order = ["nitrate", "ms_spc", "ms_chloride", "arsenic_us", "assam",
                "fluoride"]
    short = {
        "nitrate": "Nitrate", "ms_spc": "Sp. cond.",
        "ms_chloride": "Chloride", "arsenic_us": "Arsenic US",
        "assam": "Arsenic Assam", "fluoride": "Fluoride",
    }
    gaps = {}
    for r in rows:
        gaps[(ds_key[r["dataset"]], r["model"])] = float(r["gap"])
    grid = np.array([[gaps[(d, m)] for d in ds_order] for m in model_order])

    vmax = t["method3_extended_gap_max"]
    im = ax.imshow(grid, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=AF["annotation"],
                    color="white" if v > 0.55 * vmax else GREY_DARK)
    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels([short[d] for d in ds_order],
                       fontsize=AF["tick_label"] - 3, rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels([model_key[m] for m in model_order],
                       fontsize=AF["tick_label"] - 2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Random-to-spatial gap", fontsize=AF["axis_label"] - 2)
    cb.ax.tick_params(labelsize=AF["tick_label"] - 2)
    cb.outline.set_visible(False)


def blocksize_folds():
    """Per-fold spatial scores at each holdout granularity (raw, no derived
    gaps: the random-median reference lines come from truth_table)."""
    nit = _read_csv(PROC / "usgs_nitrate" / "blocksize_sensitivity_detail.csv")
    flu = _read_csv(PROC / "global_fluoride" / "blocksize_sensitivity_detail.csv")
    ms = _read_csv(PROC / "mississippi_salinity_blocksize_sensitivity"
                   / "fold_results.csv")
    out = {"nitrate": {}, "fluoride": {}, "ms_spc": {}, "ms_chloride": {}}
    for r in nit:
        out["nitrate"].setdefault(float(r["block_deg"]), []).append(float(r["r2"]))
    for r in flu:
        out["fluoride"].setdefault(float(r["block_deg"]), []).append(float(r["auc"]))
    for r in ms:
        if r["model"] != "xgboost" or r["regime"] != "spatial_block":
            continue
        key = "ms_spc" if r["target"] == "specific_conductance" else "ms_chloride"
        out[key].setdefault(float(r["block_deg"]), []).append(float(r["r2"]))
    # Assam: killshot json stores per-blocksize medians (no per-fold dump);
    # its 0.15 deg per-fold cloud comes from spatial_block_fold_detail.csv.
    asm_js = json.load(open(PROC / "assam_arsenic" / "killshot_results.json"))
    asm_medians = {float(x["block_deg"]): float(x["ml_median_auc"])
                   for x in asm_js["blocksize_sensitivity"]}
    asm_folds = [float(r["ml_auc"]) for r in
                 _read_csv(PROC / "assam_arsenic" / "spatial_block_fold_detail.csv")
                 if r["regime"] == "spatial_block"]
    return out, asm_medians, asm_folds


def panel_e(ax, t):
    """Spatial-holdout score vs holdout granularity; per-fold clouds where a
    fold dump exists, medians as bold dashes, random-split medians as dashed
    reference lines. The cloud never climbing back to its reference line is
    the coarsening-does-not-rescue claim. Split axes: AUC on top, R2 below.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fold_data, asm_medians, asm_folds = blocksize_folds()
    rng = np.random.default_rng(JITTER_SEED)

    div = make_axes_locatable(ax)
    ax_top = ax                          # classification (AUC)
    ax_bot = div.append_axes("bottom", size="100%", pad=0.14, sharex=ax)

    def cloud(a, x, vals, key, lo, color=C1):
        vals = np.asarray(vals, float)
        xs = x * np.exp(rng.uniform(-0.06, 0.06, size=len(vals)))
        a.scatter(xs, np.clip(vals, lo, None), s=20, marker=TARGET_MARKER[key],
                  facecolor=color, edgecolor=GREY_DARK, alpha=0.70, linewidth=0.8,
                  zorder=3)
        for v, xx in zip(vals, xs):
            if v < lo:      # off-scale fold: mark at floor + print true value
                a.annotate(f"{v:.1f}", (xx, lo), textcoords="offset points",
                           xytext=(2, 3), fontsize=AF["annotation"] - 3,
                           color=color)
        a.hlines(float(np.median(vals)), x / 1.18, x * 1.18, color=GREY_DARK,
                 lw=2.6, zorder=6)

    # --- top: classification (Assam 0.1-0.25 deg, fluoride 3-10 deg) ---
    for bd, med in sorted(asm_medians.items()):
        if abs(bd - 0.15) < 1e-9:
            cloud(ax_top, bd, asm_folds, "assam", 0.0)
        else:
            ax_top.hlines(med, bd / 1.18, bd * 1.18, color=GREY_DARK, lw=2.6,
                          zorder=6)
    for bd, vals in sorted(fold_data["fluoride"].items()):
        cloud(ax_top, bd, vals, "fluoride", 0.0)
    # random-split medians as per-target reference lines over their x-span
    ax_top.hlines(t["assam_arsenic_random_auc"], 0.085, 0.32, color=GREY,
                  lw=1.4, ls="--", zorder=2)
    ax_top.hlines(t["fluoride_random_auc"], 2.4, 12, color=GREY, lw=1.4,
                  ls="--", zorder=2)
    ax_top.axhline(0.5, color=GREY_DARK, lw=0.7, ls=":", zorder=1)
    ax_top.set_ylim(0.15, 1.02)
    ax_top.set_ylabel("ROC-AUC", fontsize=AF["axis_label"] - 1)
    # direct group annotations (no legend: the clusters are disjoint in x);
    # short forms, full names in the caption
    ax_top.annotate("Arsenic", (0.16, 0.30), ha="center", va="top",
                    fontsize=AF["annotation"] - 1, color=GREY_DARK)
    ax_top.annotate("Fluoride", (5.5, 0.19), ha="center", va="bottom",
                    fontsize=AF["annotation"] - 1, color=GREY_DARK)

    # --- bottom: regression (Mississippi 0.5-2 deg, nitrate 2-8 deg) ---
    lo = -1.0
    ax_bot.axhspan(lo, 0, color=NEG_SHADE, zorder=0)
    ax_bot.axhline(0, color=GREY_DARK, lw=0.7, ls=":", zorder=1)
    # Local high-contrast colours for the dense lower block-size panel. The rest
    # of Fig. 2 keeps the groundwater ramp; here hue is added so readers do not
    # have to rely on marker shape alone.
    bottom_colours = {
        "ms_spc": PANEL_E_TEAL,
        "ms_chloride": PANEL_E_ORANGE,
        "nitrate": C1,
    }
    for key in ("ms_spc", "ms_chloride", "nitrate"):
        for bd, vals in sorted(fold_data[key].items()):
            cloud(ax_bot, bd, vals, key, lo, bottom_colours[key])
    ax_bot.hlines(t["mississippi_specific_conductance_random_r2"], 0.42, 2.4,
                  color=GREY, lw=1.4, ls="--", zorder=2)
    ax_bot.hlines(t["nitrate_random_r2"], 1.7, 9.5, color=GREY, lw=1.4,
                  ls="--", zorder=2)
    ax_bot.set_ylim(lo, 1.0)
    ax_bot.set_ylabel("$R^2$", fontsize=AF["axis_label"] - 1)
    bot_handles = [
        Line2D([], [], marker=TARGET_MARKER["ms_spc"], ls="none",
               markerfacecolor=bottom_colours["ms_spc"], markeredgecolor=GREY_DARK,
               markersize=6,
               label="Sp. cond."),
        Line2D([], [], marker=TARGET_MARKER["ms_chloride"], ls="none",
               markerfacecolor=bottom_colours["ms_chloride"], markeredgecolor=GREY_DARK,
               markersize=6,
               label="Chloride"),
        Line2D([], [], marker=TARGET_MARKER["nitrate"], ls="none",
               markerfacecolor=bottom_colours["nitrate"], markeredgecolor=GREY_DARK,
               markersize=6,
               label="Nitrate"),
        Line2D([], [], color=GREY, lw=1.4, ls="--",
               label="Random median"),
    ]
    # x < 0.4 deg is empty in the bottom panel (regression clouds start at
    # 0.5 deg); anchor the legend there
    ax_bot.legend(handles=bot_handles, loc="lower left",
                  bbox_to_anchor=(0.0, 0.0),
                  fontsize=AF["legend"] - 2, frameon=False,
                  handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)

    # shared log x: set scale once, then pin the tick set (sharex propagates)
    ax_bot.set_xscale("log")
    ax_bot.set_xlim(0.075, 13)
    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8]
    ax_bot.set_xticks(ticks)
    ax_bot.set_xticklabels([f"{v:g}" for v in ticks],
                           fontsize=AF["tick_label"] - 1)
    ax_bot.minorticks_off()
    ax_top.minorticks_off()
    ax_bot.set_xlabel("Spatial block size (degrees)",
                      fontsize=AF["axis_label"] - 1)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.tick_params(axis="x", length=0)


def panel_f(ax, t):
    """Mechanism lollipop: density and distance Spearman per coordinate
    target; +/-0.2 negligible band; only fluoride density exits the band.
    Negative density correlation = error falls where sampling is denser
    (the neighbour-copyable signature)."""
    keys = ["nitrate", "ms_spc", "ms_chloride", "assam", "fluoride"]
    tt_key = {
        "nitrate": "nitrate", "ms_spc": "mississippi_specific_conductance",
        "ms_chloride": "mississippi_chloride", "assam": "assam",
        "fluoride": "fluoride",
    }
    short = {
        "nitrate": "Nitrate (US)", "ms_spc": "Sp. cond. (Miss.)",
        "ms_chloride": "Chloride (Miss.)", "assam": "Arsenic (Assam)",
        "fluoride": "Fluoride (global)",
    }
    dens = [t[f"spatial_mechanism_{tt_key[k]}_density_spearman"] for k in keys]
    dist = [t[f"spatial_mechanism_{tt_key[k]}_distance_spearman"] for k in keys]

    def half_up(v):
        # round half away from zero so the label matches the prose
        # (-0.425 -> -0.43, not the float-default -0.42)
        return np.sign(v) * np.floor(abs(v) * 100 + 0.5) / 100

    ax.axvspan(-0.2, 0.2, color="#F0F0F3", zorder=0)
    ax.axvline(0, color=GREY_DARK, lw=0.8, zorder=1)
    ys = np.arange(len(keys))[::-1]
    off = 0.18
    for y, dn, dt_ in zip(ys, dens, dist):
        emphas = abs(dn) > 0.2
        c = C1 if emphas else GREY_DARK
        ax.plot([0, dn], [y + off, y + off], color=c, lw=2.2, zorder=3)
        ax.scatter([dn], [y + off], s=90, color=c, zorder=4)
        ax.plot([0, dt_], [y - off, y - off], color=GREY, lw=2.2, zorder=3)
        ax.scatter([dt_], [y - off], s=90, facecolor="white",
                   edgecolor=GREY, linewidth=1.6, zorder=4)
        ax.annotate(f"{half_up(dn):.2f}", (dn, y + off),
                    textcoords="offset points",
                    xytext=(-8 if dn < 0 else 8, 0),
                    ha="right" if dn < 0 else "left", va="center",
                    fontsize=AF["annotation"] - 1, color=c,
                    fontweight="bold" if emphas else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([short[k] for k in keys], fontsize=AF["tick_label"] - 1)
    ax.set_ylim(-0.95, len(keys) - 0.35)
    ax.set_xlim(-0.62, 0.62)
    ax.set_xlabel("Spearman correlation with model error",
                  fontsize=AF["axis_label"] - 1)
    ax.annotate("negligible ($|R|$ < 0.2)", (0, -0.85), ha="center",
                va="bottom", fontsize=AF["annotation"] - 1, color=GREY_DARK)
    handles = [
        Line2D([], [], marker="o", ls="-", color=C1, markersize=8,
               label="Local well density"),
        Line2D([], [], marker="o", ls="-", color=GREY,
               markerfacecolor="white", markeredgecolor=GREY, markersize=8,
               label="Nearest-training distance"),
    ]
    # no clean in-plot anchor (all five rows are populated): park the legend
    # above the axes, two columns
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=AF["legend"] - 2, frameon=False, handletextpad=0.5,
              borderaxespad=0.0, labelspacing=0.3, columnspacing=1.2)


def rare_event_lift():
    """Per-fold AP-lift for the two rare-event classifiers."""
    flu = _read_csv(PROC / "global_fluoride" / "rare_event_fold_detail.csv")
    ars = _read_csv(PROC / "usgs_arsenic" / "pr_auc_fold_detail.csv")
    return {
        "fluoride": (
            [float(r["ap_lift"]) for r in flu if r["regime"] == "random_kfold"],
            [float(r["ap_lift"]) for r in flu
             if r["regime"] == "leave_one_continent"]),
        "arsenic_us": (
            [float(r["ap_lift"]) for r in ars
             if r["regime"] == "random_kfold_matched"],
            [float(r["ap_lift"]) for r in ars
             if r["regime"] == "leave_one_ecoregion"]),
    }


def _lift_panel(ax, t, key, med_random, med_spatial):
    """Shared AP-lift summary: bars use truth-table median AP-lift; vertical
    error bars show fold-level IQR. Light jittered points keep the per-fold
    evidence visible without returning to a box plot."""
    rand, spat = [np.asarray(v, dtype=float) for v in rare_event_lift()[key]]
    vals = [rand, spat]
    meds = [float(med_random), float(med_spatial)]
    positions = np.array([0, 1], dtype=float)
    bottom = 0.05
    colours = [C5, C1]
    edges = [GREY_DARK, GREY_DARK]
    point_edges = [GREY_DARK, C1]
    rng = np.random.default_rng(JITTER_SEED)

    for x, v, med, fc, ec, point_ec in zip(
            positions, vals, meds, colours, edges, point_edges):
        q25, q75 = np.percentile(v, [25, 75])
        ax.bar(x, med - bottom, width=0.52, bottom=bottom,
               facecolor=fc, edgecolor=ec, linewidth=1.2, alpha=0.9,
               zorder=2)
        ax.vlines(x, q25, q75, color=GREY_DARK, lw=1.2, zorder=5)
        ax.hlines([q25, q75], x - 0.10, x + 0.10, color=GREY_DARK,
                  lw=1.2, zorder=5)
        xs = x + rng.uniform(-0.11, 0.11, size=len(v))
        ax.scatter(xs, v, s=18, marker="o", facecolor="white",
                   edgecolor=point_ec,
                   linewidth=0.7, alpha=0.65, zorder=4)

    ax.annotate(f"{med_random:g}", (-0.33, med_random), ha="right",
                va="center", fontsize=AF["annotation"], color=GREY_DARK)
    ax.annotate(f"{med_spatial:g}", (1.33, med_spatial), ha="left",
                va="center", fontsize=AF["annotation"], color=C1,
                fontweight="bold")
    ax.axhline(1.0, color=GREY_DARK, lw=0.8, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 80)
    ax.set_xlim(-0.7, 1.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random\nsplit", "Spatial\nholdout"],
                       fontsize=AF["tick_label"] - 1)
    # short ylabel: the full term (average-precision lift) is defined in the
    # caption; the long form is taller than this short panel and collides
    # with the left-edge panel label
    ax.set_ylabel("AP-lift (per fold)", fontsize=AF["axis_label"] - 1)


def panel_g(ax, t):
    """Fluoride AP-lift collapse (base rate 0.9%)."""
    _lift_panel(ax, t, "fluoride",
                t["fluoride_random_ap_lift"], t["fluoride_continent_ap_lift"])


def panel_h(ax, t):
    """Arsenic (US) AP-lift collapse (base rate 13%)."""
    _lift_panel(ax, t, "arsenic_us",
                t["arsenic_random_ap_lift"], t["arsenic_ecoregion_ap_lift"])


PANEL_FUNCS = {"a": panel_a, "b": panel_b, "c": panel_c, "d": panel_d,
               "e": panel_e, "f": panel_f, "g": panel_g, "h": panel_h}
PANEL_SIZES = {"a": (A4_WIDTH, 3.4), "b": (A4_WIDTH, 3.4),
               "c": (A4_WIDTH / 2, 4.2), "d": (A4_WIDTH / 2, 4.2),
               "e": (A4_WIDTH / 2, 4.4), "f": (A4_WIDTH / 2, 3.4),
               "g": (A4_WIDTH / 4, 3.2), "h": (A4_WIDTH / 4, 3.2)}


# =============================================================================
# Entry points
# =============================================================================

def render_single(key):
    apply_plot_style(font_sizes=AF)
    fig, ax = plt.subplots(figsize=PANEL_SIZES[key])
    t = load_truth()
    PANEL_FUNCS[key](ax, t)
    add_panel_label(ax, key, font_sizes=AF)
    save_figure(fig, f"fig2_panels/fig2_panel_{key}.png")


def render_full():
    """Assemble the eight panels: (a)(b) full-width rows, then 2x3 grid.

    Panel labels are placed in FIGURE coordinates so the two columns align
    vertically: left-column labels (a,b,c,e,g) share one x at the left edge,
    right-column labels (d,f,h) sit on the figure midline. A draw pass runs
    first so aspect-equal (c) and the split axes in (e) report their final
    positions.
    """
    apply_plot_style(font_sizes=AF)
    t = load_truth()
    fig = plt.figure(figsize=(A4_WIDTH, 12.8))
    gs = fig.add_gridspec(5, 2, height_ratios=[2.35, 2.35, 3.5, 3.5, 2.5],
                          hspace=0.42, wspace=0.55,
                          left=0.09, right=0.96, top=0.985, bottom=0.045)
    slots = {
        "a": gs[0, :], "b": gs[1, :],
        "c": gs[2, 0], "d": gs[2, 1],
        "e": gs[3, 0], "f": gs[3, 1],
        "g": gs[4, 0], "h": gs[4, 1],
    }
    axes = {}
    for key, slot in slots.items():
        ax = fig.add_subplot(slot)
        PANEL_FUNCS[key](ax, t)
        axes[key] = ax

    fig.canvas.draw()          # finalize aspect/divider-adjusted positions
    LEFT_X, MID_X = 0.005, 0.505
    label_y_offset = {"b": 0.016}
    for key, ax in axes.items():
        x = LEFT_X if key in "abceg" else MID_X
        y = ax.get_position().y1 + label_y_offset.get(key, 0.004)
        fig.text(x, y, f"({key})", fontsize=AF["panel_label"],
                 fontweight="bold", va="bottom", ha="left")
    save_figure(fig, "fig2_spatial_collapse.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", help="render one panel preview (a-h)")
    args = p.parse_args()
    if args.panel:
        render_single(args.panel)
    else:
        render_full()


if __name__ == "__main__":
    main()
