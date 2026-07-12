"""
Fig. 3 -- Free baselines bound water-level forecast skill across protocols,
learners and driver sets.

Eight panels, A4 width, three rows (matches the Fig. 2 information density):

  (a)(b)(c) top row -- rolling-origin per-well median R2 versus forecast
      horizon (1/3/6/12 steps) for GEMS Germany, the North China Plain and the
      GROW 16-country panel. Curves: tuned random forest, last-value
      persistence, seasonal comparator (horizon-independent dotted line).
      The h1 persistence meet-or-beat share is annotated per panel.
  (d) middle left -- same-fold per-unit scatter: persistence R2 versus random
      forest R2 for every analysis unit (131 GEMS grid cells, 62 North China
      Plain grid cells, 16 GROW countries). Points on the diagonal = the free
      baseline fully reproduces the learner. Per-dataset recovery in legend.
  (e) middle right -- six-learner one-step per-well median R2 gain over
      persistence with percentile-bootstrap 95% CIs (B = 2000); dashed line
      marks the largest median gain in the panel (0.037).
  (f) bottom left -- exogenous-driver probe, native short horizon: persistence
      recovery of flexible-model skill in four external datasets (FrenchPiezo, Swiss Groundwater Database, Tuscany drought wells,
      CAMELS-GB), with the 100% meet-or-exceed line.
  (g) bottom centre -- horizon switching in the French and Swiss daily panels:
      median R2 of persistence, Ridge and XGBoost at 7, 28 and 84 days. The
      best comparator changes from persistence (7 d) to cheap linear (28 d);
      by 84 d all floor at zero except Ridge in France.
  (h) bottom right -- feature ablation (Ridge): groundwater-level lags alone
      reproduce most full-feature skill, exogenous drivers alone have zero
      median skill at both 7 and 28 days.

Data sources (READ-ONLY; never re-run the underlying experiments):
  data/processed/rolling_origin_temporal.json                          (a-c)
  data/processed/water_level_multimodel_panel/same_fold_panel.csv      (d)
  data/processed/water_level_multimodel_panel/bootstrap_ci.csv         (e)
  data/processed/truth_table.json  exo_* recovery keys                 (f)
  data/processed/exogenous_driver_external_probe_robustness/
      model_summary.csv                                                (g)
      feature_ablation_summary.csv                                     (h)

Hard conventions (see HANDOFF):
  * Per-well medians ONLY in (a)-(c); pooled GROW R2 is a pooling artefact and
    stays in the main text / Supplementary Table S23.
  * h6 gaps must NOT be framed as multi-step reliability: learner and
    persistence medians are both negative beyond one step (caption aligned
    with Supplementary Table S30).
  * C1 deep indigo = free-baseline emphasis colour, same semantics as Fig. 2.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import (
    apply_plot_style, save_figure,
    PALETTE, FONT_SIZES_A4, A4_WIDTH,
)

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
PANEL_DIR = PROC / "water_level_multimodel_panel"
EXO_DIR = PROC / "exogenous_driver_external_probe_robustness"

C1, C2, C3, C4, C5 = PALETTE
GREY = "#9AA0A6"
GREY_DARK = "#5F6368"
NEG_SHADE = "#F6E8E8"      # same worse-than-baseline wash as Fig. 2
AF = FONT_SIZES_A4

HORIZONS = [1, 3, 6, 12]
XPOS = [0, 1, 2, 3]        # ordinal spacing for the four discrete horizons

DATASETS = {                # rolling_origin_temporal.json keys -> display names
    "gems_de": "GEMS Germany",
    "ncp_cn": "North China Plain",
    "grow_16": "GROW 16 countries",
}
DATASET_SHORT = {"gems_de": "GEMS", "ncp_cn": "NCP", "grow_16": "GROW"}
CSV_DS = {"gems_de": "gems", "ncp_cn": "ncp", "grow_16": "grow"}
DS_MARKER = {"gems_de": "o", "ncp_cn": "s", "grow_16": "D"}
DS_FACE = {"gems_de": C2, "ncp_cn": C5, "grow_16": "white"}

LEARNERS = [                # model keys -> abbreviations defined in the caption
    ("ridge", "RR"),
    ("lasso", "LASSO"),
    ("elasticnet", "EN"),
    ("random_forest", "RF"),
    ("xgboost", "XGB"),
    ("histgbm", "HGB"),
]


# =============================================================================
# Data loaders (read-only)
# =============================================================================

def _read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_rolling():
    return json.load(open(PROC / "rolling_origin_temporal.json"))


def load_truth():
    return json.load(open(PROC / "truth_table.json"))


def load_same_fold():
    """Per-unit persistence and random-forest R2, keyed by dataset."""
    out = {k: {"pers": [], "rf": []} for k in DATASETS}
    key_by_csv = {v: k for k, v in CSV_DS.items()}
    for r in _read_csv(PANEL_DIR / "same_fold_panel.csv"):
        ds = key_by_csv[r["dataset"]]
        out[ds]["pers"].append(float(r["persistence_r2"]))
        out[ds]["rf"].append(float(r["random_forest_r2"]))
    return out


def load_bootstrap_h1():
    """rolling_h1 median gain_r2 CIs, keyed (dataset, model)."""
    out = {}
    for r in _read_csv(PANEL_DIR / "bootstrap_ci.csv"):
        if r["scope"] == "rolling_h1" and r["metric"] == "median_gain_r2":
            out[(r["dataset"], r["model"])] = (
                float(r["point"]), float(r["ci_lo"]), float(r["ci_hi"]))
    return out


def load_exo_switching():
    """Median R2 by (dataset, model, horizon label) from the robustness panel."""
    out = {}
    for r in _read_csv(EXO_DIR / "model_summary.csv"):
        out[(r["dataset"], r["model"], r["horizon_label"])] = float(r["median_r2"])
    return out


def load_exo_ablation():
    """Ridge median R2 by (dataset, feature set, horizon label)."""
    out = {}
    for r in _read_csv(EXO_DIR / "feature_ablation_summary.csv"):
        if r["model"] == "ridge":
            out[(r["dataset"], r["feature_set"], r["horizon_label"])] = \
                float(r["median_r2"])
    return out


def decay_series(rolling, ds_key):
    """Per-well median R2 curves for one dataset: (ml, pers, seasonal const)."""
    ml, pers = [], []
    for h in HORIZONS:
        pw = rolling[ds_key]["horizons"][f"h{h}"]["per_well_median"]
        ml.append(pw["ml_r2"])
        pers.append(pw["persistence_r2"])
    seasonal = rolling[ds_key]["horizons"]["h1"]["per_well_median"][
        "seasonal_persistence_r2"]
    return ml, pers, seasonal


# =============================================================================
# Panels
# =============================================================================

YLIM_DECAY = (-1.32, 1.02)


def _decay_panel(ax, rolling, ds_key, show_ylabel):
    """(a)(b)(c): per-well median R2 vs horizon, one dataset."""
    ml, pers, seasonal = decay_series(rolling, ds_key)

    ax.axhspan(YLIM_DECAY[0], 0, color=NEG_SHADE, zorder=0)
    ax.axhline(0, color=GREY_DARK, lw=0.8, ls="--", zorder=1)
    ax.axhline(seasonal, color=GREY, lw=1.6, ls=":", zorder=2)

    ax.plot(XPOS, pers, color=C1, lw=2.0, marker="s", markersize=5.5, zorder=4)
    ax.plot(XPOS, ml, color=GREY_DARK, lw=2.0, marker="o", markersize=5.5,
            zorder=3)

    ax.set_xticks(XPOS)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(*YLIM_DECAY)
    ax.set_xlabel("Horizon (time steps)")
    if show_ylabel:
        ax.set_ylabel("Per-well median $R^2$")
    else:
        ax.set_yticklabels([])
    ax.text(0.96, 0.965, DATASETS[ds_key],
            transform=ax.transAxes, ha="right", va="top",
            fontsize=AF["annotation"] - 2, color=GREY_DARK)


def decay_legend_handles():
    """Shared figure-level legend for the top row."""
    return [
        Line2D([], [], color=GREY_DARK, lw=2.0, marker="o",
               markersize=6, label="Random forest"),
        Line2D([], [], color=C1, lw=2.0, marker="s",
               markersize=6, label="Persistence (last value)"),
        Line2D([], [], color=GREY, lw=1.6, ls=":",
               label="Seasonal (same month last year)"),
    ]


def panel_a(ax, data):
    _decay_panel(ax, data["rolling"], "gems_de", True)


def panel_b(ax, data):
    _decay_panel(ax, data["rolling"], "ncp_cn", False)


def panel_c(ax, data):
    _decay_panel(ax, data["rolling"], "grow_16", False)


def panel_d(ax, data):
    """Same-fold per-unit scatter: persistence vs random forest R2."""
    sf = data["same_fold"]
    lim_lo, lim_hi = -0.20, 1.02
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=GREY_DARK, lw=1.0,
            ls="--", zorder=1)
    for ds_key in DATASETS:
        ax.scatter(sf[ds_key]["pers"], sf[ds_key]["rf"], s=34,
                   marker=DS_MARKER[ds_key], facecolor=DS_FACE[ds_key],
                   edgecolor=GREY_DARK, linewidth=0.8, alpha=0.80, zorder=3,
                   label=DATASET_SHORT[ds_key])
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Persistence $R^2$ (per unit)")
    ax.set_ylabel("Random forest $R^2$ (per unit)")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2,
              borderaxespad=0.1, fontsize=AF["legend"] - 1,
              alignment="left")


def panel_e(ax, data):
    """Six-learner one-step median gain over persistence, bootstrap 95% CI."""
    boot = data["boot"]
    ROW, GAP = 1.0, 3.0          # row spacing; extra gap holding group header
    y = 0.0
    ytick_pos, ytick_lab = [], []
    ceiling = max(v[0] for v in boot.values())

    for ds_key, ds_name in DATASETS.items():
        ax.text(0.01, y, ds_name, fontsize=AF["annotation"] - 2,
                fontweight="bold", va="center", ha="left", color=GREY_DARK,
                transform=ax.get_yaxis_transform())
        y += ROW
        for m_key, m_name in LEARNERS:
            point, lo, hi = boot[(CSV_DS[ds_key], m_key)]
            ax.errorbar(point, y, xerr=[[point - lo], [hi - point]],
                        fmt="o", markersize=5, color=GREY_DARK,
                        ecolor=GREY_DARK, elinewidth=1.3, capsize=2.2,
                        zorder=3)
            ytick_pos.append(y)
            ytick_lab.append(m_name)
            y += ROW
        y += GAP - ROW

    ax.axvline(0, color=C1, lw=1.8, zorder=2)
    ax.axvline(ceiling, color=GREY, lw=1.2, ls="--", zorder=1)
    ax.text(ceiling, -0.9, f"ceiling {ceiling:.3f}", ha="center", va="bottom",
            fontsize=AF["annotation"] - 2, color=GREY_DARK)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=AF["annotation"] - 2)
    ax.set_ylim(y - GAP + 1.3, -1.4)          # inverted: first group on top
    ax.set_xlim(-0.030, 0.050)
    ax.set_xlabel("One-step median $R^2$ gain over persistence",
                  fontsize=AF["axis_label"] - 1)
    ax.tick_params(axis="y", length=0)


def panel_f(ax, data):
    """Exogenous probe, native short horizon: persistence recovery, 4 sets."""
    t = data["truth"]
    order = [
        ("exo_french_recovery_pct", "France\n7 d", "exo_french_n"),
        ("exo_swiss_recovery_pct", "Swiss\n7 d", "exo_swiss_n"),
        ("exo_tuscany_recovery_pct", "Tuscany\n1 mo", "exo_tuscany_n"),
        ("exo_camelsgb_recovery_pct", "CAMELS\n1 mo", "exo_camelsgb_n"),
    ]
    vals = [t[k] for k, _, _ in order]
    xs = range(len(vals))
    ax.bar(xs, vals, width=0.58, color=C5, edgecolor=GREY_DARK,
           linewidth=0.7, zorder=3)
    ax.axhline(100, color=GREY_DARK, lw=1.0, ls="--", zorder=4)
    for x, v in zip(xs, vals):
        ax.text(x, v + 2, f"{v:.1f}", ha="center", va="bottom",
                fontsize=AF["annotation"] - 1, color=GREY_DARK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab, _ in order],
                       fontsize=AF["annotation"] - 3)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Persistence recovery of\nflexible-model skill (%)")


def panel_g(ax, data):
    """French/Swiss horizon switching: persistence -> Ridge -> zero."""
    exo = data["exo_switch"]
    hz = ["h7_days", "h28_days", "h84_days"]
    xp = [0, 1, 2]
    series = [("persistence", C1, "s"), ("ridge", C2, "^"),
              ("xgboost", GREY_DARK, "o")]
    for ds, ls, filled in (("french_piezoforecast", "-", True),
                           ("swiss_groundwater_database", "--", False)):
        for model, color, marker in series:
            ys = [exo[(ds, model, h)] for h in hz]
            ax.plot(xp, ys, color=color, ls=ls, lw=1.8, marker=marker,
                    markersize=5.5,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color, zorder=3)
    ax.axhline(0, color=GREY_DARK, lw=0.8, ls="--", zorder=1)
    ax.set_xticks(xp)
    ax.set_xticklabels(["7", "28", "84"])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(-0.06, 1.0)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Median $R^2$")
    model_handles = [
        Line2D([], [], color=C1, lw=1.8, marker="s", markersize=5,
               label="Persistence"),
        Line2D([], [], color=C2, lw=1.8, marker="^", markersize=5,
               label="Ridge"),
        Line2D([], [], color=GREY_DARK, lw=1.8, marker="o", markersize=5,
               label="XGB"),
    ]
    country_handles = [
        Line2D([], [], color=GREY_DARK, lw=1.5, ls="-", label="France"),
        Line2D([], [], color=GREY_DARK, lw=1.5, ls="--", label="Switzerland"),
    ]
    model_legend = ax.legend(
        handles=model_handles, loc="lower center", frameon=False, ncol=3,
        bbox_to_anchor=(0.5, 1.02), fontsize=AF["legend"] - 3,
        handlelength=1.2, borderaxespad=0.0, labelspacing=0.2,
        handletextpad=0.3, columnspacing=0.7,
    )
    country_legend = ax.legend(
        handles=country_handles, loc="upper right", frameon=False, ncol=1,
        fontsize=AF["legend"] - 3, handlelength=1.4, borderaxespad=0.15,
        labelspacing=0.15, handletextpad=0.35,
    )
    ax.add_artist(model_legend)


def panel_h(ax, data):
    """Feature ablation (Ridge): lags carry the skill, drivers alone do not."""
    ab = data["exo_abl"]
    groups = [("french_piezoforecast", "h7_days", "France\n7 d"),
              ("french_piezoforecast", "h28_days", "France\n28 d"),
              ("swiss_groundwater_database", "h7_days", "Switz.\n7 d"),
              ("swiss_groundwater_database", "h28_days", "Switz.\n28 d")]
    sets = [("gwl_plus_drivers", "Full", GREY_DARK),
            ("gwl_lags_only", "GWL lags", C1),
            ("drivers_only", "Drivers", C4)]
    width = 0.26
    for j, (fs, label, color) in enumerate(sets):
        xs = [i + (j - 1) * width for i in range(len(groups))]
        ys = [ab[(ds, fs, h)] for ds, h, _ in groups]
        ax.bar(xs, ys, width=width * 0.92, color=color, zorder=3, label=label,
               edgecolor=GREY_DARK, linewidth=0.6)
        if fs == "drivers_only":            # zero bars: make absence explicit
            for x in xs:
                ax.text(x, 0.012, "0", ha="center", va="bottom",
                        fontsize=AF["annotation"] - 2, color=GREY_DARK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lab for _, _, lab in groups],
                       fontsize=AF["annotation"] - 3)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Ridge median $R^2$")
    ax.legend(loc="lower center", frameon=False, fontsize=AF["legend"] - 3,
              ncol=3, bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0,
              labelspacing=0.2,
              handlelength=0.9, handletextpad=0.25, columnspacing=0.55)


PANEL_FUNCS = {"a": panel_a, "b": panel_b, "c": panel_c, "d": panel_d,
               "e": panel_e, "f": panel_f, "g": panel_g, "h": panel_h}
PANEL_SIZES = {"a": (A4_WIDTH / 3 + 0.4, 2.9), "b": (A4_WIDTH / 3 + 0.4, 2.9),
               "c": (A4_WIDTH / 3 + 0.4, 2.9),
               "d": (A4_WIDTH / 2, 4.4), "e": (A4_WIDTH / 2, 4.4),
               "f": (A4_WIDTH / 3 + 0.4, 3.0), "g": (A4_WIDTH / 3 + 0.4, 3.0),
               "h": (A4_WIDTH / 3 + 0.4, 3.0)}


# =============================================================================
# Entry points
# =============================================================================

def _load_all():
    return {
        "rolling": load_rolling(),
        "truth": load_truth(),
        "same_fold": load_same_fold(),
        "boot": load_bootstrap_h1(),
        "exo_switch": load_exo_switching(),
        "exo_abl": load_exo_ablation(),
    }


def render_single(key):
    apply_plot_style(font_sizes=AF)
    data = _load_all()
    fig, ax = plt.subplots(figsize=PANEL_SIZES[key])
    PANEL_FUNCS[key](ax, data)
    save_figure(fig, f"fig3_panels/fig3_panel_{key}.png")


def render_full():
    """Assemble: (a)(b)(c) decay row, (d)(e) protocol row, (f)(g)(h) exo row.

    Panel labels are placed in figure coordinates after a draw pass (Fig. 2
    assembly pattern) so labels track each axis' final position, including the
    aspect-equal scatter in (d).
    """
    apply_plot_style(font_sizes=AF)
    data = _load_all()
    fig = plt.figure(figsize=(A4_WIDTH, 11.8))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.6, 1.05],
                          hspace=0.66, wspace=0.90,
                          left=0.10, right=0.96, top=0.965, bottom=0.055)
    slots = {"a": gs[0, 0:2], "b": gs[0, 2:4], "c": gs[0, 4:6],
             "d": gs[1, 0:3], "e": gs[1, 3:6],
             "f": gs[2, 0:2], "g": gs[2, 2:4], "h": gs[2, 4:6]}
    axes = {}
    for key, slot in slots.items():
        ax = fig.add_subplot(slot)
        PANEL_FUNCS[key](ax, data)
        axes[key] = ax

    fig.canvas.draw()
    legend_y = (
        min(axes[key].get_position().y0 for key in ("a", "b", "c"))
        + max(axes[key].get_position().y1 for key in ("d", "e"))
    ) / 2
    fig.legend(handles=decay_legend_handles(), loc="center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, legend_y),
               columnspacing=1.8, handletextpad=0.5)

    for key, ax in axes.items():
        pos = ax.get_position()
        fig.text(pos.x0 - 0.058, pos.y1 + 0.006, f"({key})",
                 fontsize=AF["panel_label"], fontweight="bold",
                 va="bottom", ha="left")
    save_figure(fig, "fig3_horizon_decay.png")


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
