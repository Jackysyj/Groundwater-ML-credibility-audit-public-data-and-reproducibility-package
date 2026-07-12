#!/usr/bin/env python3
"""
Water-level MULTI-MODEL panel + sensitivity suite for the three corpus
datasets (GEMS Germany, North China Plain, GROW 16 countries).

Why: manuscript section 2.3 had a learner-dependence asymmetry. The exogenous
driver panels (French/Swiss) carry a six-learner panel, but the corpus
water-level datasets were RandomForest-only in BOTH protocols (same-fold
random split and chronological rolling-origin). This script closes the gap.

HARD RULE (2026-07-05 lag-misalignment lesson): the supervised constructions
are IMPORTED from the headline scripts, never rewritten:
  - same-fold GEMS : spatial_cv_experiment_gems_wells (clusters + make_supervised)
  - same-fold NCP  : spatial_cv_experiment_ncp_wells  (clusters + make_supervised)
  - same-fold GROW : persistence_baseline_grow        (per-country supervised)
  - rolling-origin : rolling_origin_temporal          (build_supervised + loaders)
So ML features end AT z[i], target = z[i+h], persistence = z[i]: identical
information cutoff for every learner and the baseline.

Learners mirror the exogenous six-learner panel (exogenous_driver_robustness):
ridge / lasso / elasticnet / random_forest / xgboost / histgbm. RandomForest
keeps the DATASET-MATCHED headline hyperparameters (GEMS/NCP same-fold and all
rolling: 200 trees depth 6; GROW same-fold: 150 trees depth 6) so the RF rows
reproduce the published numbers exactly and anchor the panel. XGBoost runs on
GPU (tree_method=hist, device=cuda).

Blocks (each writes its own CSV; existing CSV = skip, --force to redo):
  A same_fold_panel.csv   six learners + persistence, same folds as headline
                          (GEMS 131 cells x10 seeds, NCP 62 cells x10 seeds,
                           GROW 16 countries x5 seeds)
  B rolling_panel.csv     six learners + persistence at h=1,3,6,12; pooled and
                          per-well median R2, paired gain vs persistence,
                          wins n/N (per-well filter >=5 test rows, as headline)
  C lag_sensitivity.csv   rolling h1, N_LAGS in {1,3,6,12}, persistence/ridge/xgboost
  D cutoff_sensitivity.csv rolling h1, cutoff quantile in {0.70,0.80,0.90},
                          persistence/ridge/xgboost
  E seed_stability.csv    rolling h1, RF + XGBoost, seeds 0-4
  F per_well_metrics_h{1,6}.csv  rolling per-well paired R2 + RMSE tables,
                          persistence + 6 learners; RMSE skill =
                          1 - RMSE_model / RMSE_persistence per well
  G bootstrap_ci.csv      percentile bootstrap 95% CI (B=2000) for median
                          paired gains: same-fold resamples units from
                          same_fold_panel.csv, rolling resamples wells from
                          the block-F tables (R2 gain + RMSE skill)
Then summary.json (incl. automatic reconciliation of RF rows against
rolling_origin_temporal.json and the two same-fold headline CSVs) and
analysis_summary.md.

Output dir: data/processed/water_level_multimodel_panel/
"""
import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
PROC = ROOT / "data" / "processed"
OUT_DIR = PROC / "water_level_multimodel_panel"
OUT_DIR.mkdir(exist_ok=True)

PANEL_MODELS = ["ridge", "lasso", "elasticnet", "random_forest", "xgboost", "histgbm"]
ROLL_HORIZONS = [1, 3, 6, 12]
LAG_SETTINGS = [1, 3, 6, 12]
CUTOFF_QUANTILES = [0.70, 0.80, 0.90]
SEED_STABILITY_SEEDS = [0, 1, 2, 3, 4]
F_HORIZONS = [1, 6]          # block F: one-step + six-step per-well tables
BOOT_B = 2000                # block G: bootstrap replicates
BOOT_SEED = 42
DATASETS = ["gems", "ncp", "grow"]

# dataset-matched RandomForest headline settings (anchor rows must reproduce)
RF_SAMEFOLD = {"gems": dict(n_estimators=200, max_depth=6),
               "ncp": dict(n_estimators=200, max_depth=6),
               "grow": dict(n_estimators=150, max_depth=6)}
SAMEFOLD_SEEDS = {"gems": 10, "ncp": 10, "grow": 5}


def _import(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rot = _import("rolling_origin_temporal")
sf_gems = _import("spatial_cv_experiment_gems_wells")
sf_ncp = _import("spatial_cv_experiment_ncp_wells")
sf_grow = _import("persistence_baseline_grow")

ROLL_LOADERS = {"gems": rot._load_gems, "ncp": rot._load_ncp, "grow": rot._load_grow}


def make_model(name, seed, rf_params, device="cuda"):
    """Learner factory mirroring exogenous_driver_robustness.make_model.
    No imputer: lag frames are NaN-free by construction (asserted upstream)."""
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "lasso":
        return Pipeline([("scale", StandardScaler()),
                         ("model", Lasso(alpha=0.001, max_iter=5000, random_state=seed))])
    if name == "elasticnet":
        return Pipeline([("scale", StandardScaler()),
                         ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000,
                                              random_state=seed))])
    if name == "random_forest":
        return RandomForestRegressor(**rf_params, random_state=seed, n_jobs=-1)
    if name == "xgboost":
        return XGBRegressor(n_estimators=180, max_depth=5, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                            reg_lambda=1.0, objective="reg:squarederror",
                            tree_method="hist", device=device, random_state=seed,
                            n_jobs=-1, verbosity=0)
    if name == "histgbm":
        return HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05,
                                             max_leaf_nodes=31, l2_regularization=0.05,
                                             random_state=seed)
    raise ValueError(name)


# --------------------------------------------------------------------------
# Block A: same-fold multi-model panel
# --------------------------------------------------------------------------

def _find_eligible_deterministic(mod, df):
    """Deterministic re-implementation of mod.find_eligible_clusters.

    The headline function iterates a SET in its greedy overlap-drop loop, so
    tie-breaks depend on PYTHONHASHSEED: NCP realizes 61 or 62 cells depending
    on the hash seed (GEMS/GROW are unaffected in practice). Sorting the
    candidate iteration order makes the tie-break deterministic and, for NCP,
    reproduces EXACTLY the 62-cell set of the headline CSV (verified 2026-07-10;
    8 cells realize a different well subset on ties, but the panel medians
    agree with the headline to within 0.002). Only the iteration order
    changes; thresholds and make_supervised are untouched."""
    eligible = {}
    for cell, cell_df in df.groupby("grid_cell"):
        wells = sorted(cell_df["well_id"].unique().tolist())
        if len(wells) < mod.MIN_WELLS_FOR_LOWO:
            continue
        well_months = {w: set(cell_df.loc[cell_df["well_id"] == w, "year_month"])
                       for w in wells}
        current = set(wells)
        while len(current) >= mod.MIN_WELLS_FOR_LOWO:
            common = set.intersection(*[well_months[w] for w in current])
            if len(common) >= mod.MIN_OVERLAP_MONTHS:
                break
            best_drop, best_gain = None, -1
            for w in sorted(current):  # deterministic tie-break
                rest = current - {w}
                if len(rest) < mod.MIN_WELLS_FOR_LOWO:
                    continue
                gain = len(set.intersection(*[well_months[x] for x in rest]))
                if gain > best_gain:
                    best_gain, best_drop = gain, w
            if best_drop is None:
                break
            current = current - {best_drop}
        else:
            continue
        common = set.intersection(*[well_months[w] for w in current])
        if len(common) < mod.MIN_OVERLAP_MONTHS or len(current) < mod.MIN_WELLS_FOR_LOWO:
            continue
        wells_kept = sorted(current)
        max_wells = getattr(mod, "MAX_WELLS_PER_CLUSTER", None)
        if max_wells and len(wells_kept) > max_wells:
            ranked = sorted(wells_kept,
                            key=lambda w: (-len(well_months[w] & common), w))
            wells_kept = sorted(ranked[:max_wells])
            common = set.intersection(*[well_months[w] for w in wells_kept])
        eligible[cell] = {"wells": wells_kept, "common_months": sorted(common)}
    return eligible


def _samefold_units_gems_ncp(mod):
    """Yield (unit_id, sup, X, y) per eligible grid cell, exactly as headline
    except for the deterministic eligibility tie-break (see above)."""
    df = mod.load_panel()
    eligible = _find_eligible_deterministic(mod, df)
    for cell, info in sorted(eligible.items(), key=lambda kv: -len(kv[1]["wells"])):
        sup, X, y = mod.make_supervised(df, info["wells"], info["common_months"])
        if sup is None or len(sup) < mod.MIN_ROWS_PER_CLUSTER:
            continue
        yield str(cell), sup, X, y


def _samefold_units_grow():
    """Yield (country, sup, X, y) per country, exactly as persistence_baseline_grow."""
    for p in sorted(sf_grow.PANEL_DIR.glob("*.parquet")):
        sup = sf_grow.supervised(pd.read_parquet(p))
        if len(sup) < 200:
            continue
        X = sup[[c for c in sup.columns if c.startswith("lag_")]]
        # normalize persistence column name across the three source scripts
        sup = sup.rename(columns={"y_pers": "y_persistence"})
        yield p.stem, sup, X, sup["y"]


def block_a(datasets, force):
    out_path = OUT_DIR / "same_fold_panel.csv"
    if out_path.exists() and not force:
        print(f"[A] exists, skip: {out_path}", flush=True)
        return
    rows = []
    for ds in datasets:
        n_seeds = SAMEFOLD_SEEDS[ds]
        units = (_samefold_units_grow() if ds == "grow"
                 else _samefold_units_gems_ncp(sf_gems if ds == "gems" else sf_ncp))
        for k, (unit, sup, X, y) in enumerate(units, 1):
            assert not X.isna().any().any(), f"NaN in features {ds}/{unit}"
            yv = np.asarray(y)
            persv = sup["y_persistence"].values
            idx = np.arange(len(yv))
            scores = {m: [] for m in PANEL_MODELS}
            pers_scores = []
            for seed in range(n_seeds):
                # identical fold to headline: split integer index at this seed
                itr, ite = train_test_split(idx, test_size=0.2, random_state=seed)
                pers_scores.append(r2_score(yv[ite], persv[ite]))
                for m in PANEL_MODELS:
                    model = make_model(m, seed, RF_SAMEFOLD[ds])
                    model.fit(X.iloc[itr], yv[itr])
                    scores[m].append(r2_score(yv[ite], model.predict(X.iloc[ite])))
            rec = {"dataset": ds, "unit_id": unit, "n_rows": len(sup),
                   "n_seeds": n_seeds,
                   "persistence_r2": float(np.mean(pers_scores))}
            for m in PANEL_MODELS:
                rec[f"{m}_r2"] = float(np.mean(scores[m]))
            rows.append(rec)
            print(f"[A] {ds} {k} {unit}: rows={len(sup)} "
                  + " ".join(f"{m}={rec[f'{m}_r2']:.3f}" for m in PANEL_MODELS)
                  + f" pers={rec['persistence_r2']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)  # checkpoint per dataset
    print(f"[A] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------
# Rolling-origin shared helpers (blocks B-E)
# --------------------------------------------------------------------------

_DF_CACHE, _FRAME_CACHE = {}, {}


def _roll_df(ds):
    if ds not in _DF_CACHE:
        df = ROLL_LOADERS[ds]()
        df["year_month"] = df["year_month"].astype(int)
        _DF_CACHE[ds] = df
    return _DF_CACHE[ds]


def _roll_frames(ds, horizon, n_lags=3, cutoff_q=0.80):
    """Cached rot.build_supervised call. n_lags/MIN_TRAIN_PER_WELL are module
    globals in rot; set them for the call, then restore (documented monkey-patch,
    the alternative is rewriting the pipeline, which is forbidden)."""
    key = (ds, horizon, n_lags, cutoff_q)
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    df = _roll_df(ds)
    cutoff = int(np.quantile(df["year_month"].unique(), cutoff_q))
    old_lags, old_min = rot.N_LAGS, rot.MIN_TRAIN_PER_WELL
    rot.N_LAGS, rot.MIN_TRAIN_PER_WELL = n_lags, n_lags + 2
    try:
        tr, te = rot.build_supervised(df, cutoff, horizon)
    finally:
        rot.N_LAGS, rot.MIN_TRAIN_PER_WELL = old_lags, old_min
    _FRAME_CACHE[key] = (tr, te, cutoff)
    print(f"[frame] {ds} h={horizon} lags={n_lags} q={cutoff_q}: "
          f"tr={len(tr)} te={len(te)} cutoff={cutoff}", flush=True)
    return tr, te, cutoff


def _perwell_stats(te, pred):
    """Per-well median R2 + paired stats vs persistence, mirroring
    rot.eval_horizon (filter: >=5 test rows per well)."""
    te = te.copy()
    te["_pred"] = pred
    recs = []
    for wid, gg in te.groupby("well_id"):
        if len(gg) < 5:
            continue
        recs.append({"ml": r2_score(gg["y"], gg["_pred"]),
                     "pers": r2_score(gg["y"], gg["y_persistence"])})
    pw = pd.DataFrame(recs)
    if not len(pw):
        return {}
    return {"perwell_median_r2": float(pw["ml"].median()),
            "perwell_median_pers_r2": float(pw["pers"].median()),
            "perwell_median_gain_vs_pers": float((pw["ml"] - pw["pers"]).median()),
            "wins_ge_persistence_n": int((pw["ml"] >= pw["pers"]).sum()),
            "perwell_n_wells": int(len(pw))}


def _fit_eval_rolling(ds, tr, te, model_name, seed=0):
    feat = [c for c in tr.columns if c.startswith("lag_z_")]
    assert not tr[feat].isna().any().any()
    if model_name == "persistence":
        pred = te["y_persistence"].values
    else:
        # rolling RF anchor uses the headline hyperparameters from rot.RF
        rf_params = {"n_estimators": rot.RF["n_estimators"],
                     "max_depth": rot.RF["max_depth"]}
        model = make_model(model_name, seed, rf_params)
        model.fit(tr[feat], tr["y"])
        pred = np.asarray(model.predict(te[feat]))
    rec = {"pooled_r2": float(r2_score(te["y"], pred))}
    rec.update(_perwell_stats(te, pred))
    return rec


def block_b(datasets, horizons, models, force):
    out_path = OUT_DIR / "rolling_panel.csv"
    if out_path.exists() and not force:
        print(f"[B] exists, skip: {out_path}", flush=True)
        return
    rows = []
    for ds in datasets:
        for h in horizons:
            tr, te, cutoff = _roll_frames(ds, h)
            for m in ["persistence"] + models:
                rec = {"dataset": ds, "horizon": h, "model": m,
                       "n_train_rows": len(tr), "n_test_rows": len(te),
                       "cutoff_year_month": cutoff, "n_lags": 3,
                       **_fit_eval_rolling(ds, tr, te, m)}
                rows.append(rec)
                print(f"[B] {ds} h={h} {m}: pooled={rec['pooled_r2']:.4f} "
                      f"pw={rec.get('perwell_median_r2', float('nan')):.4f} "
                      f"gain={rec.get('perwell_median_gain_vs_pers', float('nan')):.4f} "
                      f"wins={rec.get('wins_ge_persistence_n')}/{rec.get('perwell_n_wells')}",
                      flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)  # checkpoint per frame
        # free horizon frames for this dataset (h1 kept for C/D/E via re-build)
        for key in [k for k in _FRAME_CACHE if k[0] == ds and k[1] != 1]:
            del _FRAME_CACHE[key]
    print(f"[B] wrote {out_path}", flush=True)


def block_c(datasets, force):
    out_path = OUT_DIR / "lag_sensitivity.csv"
    if out_path.exists() and not force:
        print(f"[C] exists, skip: {out_path}", flush=True)
        return
    rows = []
    for ds in datasets:
        for lags in LAG_SETTINGS:
            tr, te, _ = _roll_frames(ds, 1, n_lags=lags)
            for m in ["persistence", "ridge", "xgboost"]:
                rec = {"dataset": ds, "horizon": 1, "n_lags": lags, "model": m,
                       "n_train_rows": len(tr), "n_test_rows": len(te),
                       **_fit_eval_rolling(ds, tr, te, m)}
                rows.append(rec)
                print(f"[C] {ds} lags={lags} {m}: pooled={rec['pooled_r2']:.4f} "
                      f"pw={rec.get('perwell_median_r2', float('nan')):.4f}", flush=True)
            if lags != 3:  # keep the default frame for D/E
                del _FRAME_CACHE[(ds, 1, lags, 0.80)]
            pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[C] wrote {out_path}", flush=True)


def block_d(datasets, force):
    out_path = OUT_DIR / "cutoff_sensitivity.csv"
    if out_path.exists() and not force:
        print(f"[D] exists, skip: {out_path}", flush=True)
        return
    rows = []
    for ds in datasets:
        for q in CUTOFF_QUANTILES:
            tr, te, cutoff = _roll_frames(ds, 1, cutoff_q=q)
            for m in ["persistence", "ridge", "xgboost"]:
                rec = {"dataset": ds, "horizon": 1, "cutoff_quantile": q,
                       "cutoff_year_month": cutoff,
                       "n_train_rows": len(tr), "n_test_rows": len(te),
                       "model": m, **_fit_eval_rolling(ds, tr, te, m)}
                rows.append(rec)
                print(f"[D] {ds} q={q} {m}: pooled={rec['pooled_r2']:.4f} "
                      f"pw={rec.get('perwell_median_r2', float('nan')):.4f}", flush=True)
            if q != 0.80:
                del _FRAME_CACHE[(ds, 1, 3, q)]
            pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[D] wrote {out_path}", flush=True)


def block_e(datasets, force):
    out_path = OUT_DIR / "seed_stability.csv"
    if out_path.exists() and not force:
        print(f"[E] exists, skip: {out_path}", flush=True)
        return
    rows = []
    for ds in datasets:
        tr, te, _ = _roll_frames(ds, 1)
        for m in ["random_forest", "xgboost"]:
            for seed in SEED_STABILITY_SEEDS:
                rec = {"dataset": ds, "horizon": 1, "model": m, "seed": seed,
                       **_fit_eval_rolling(ds, tr, te, m, seed=seed)}
                rows.append(rec)
                print(f"[E] {ds} {m} seed={seed}: pooled={rec['pooled_r2']:.4f} "
                      f"pw={rec['perwell_median_r2']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[E] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------
# Block F: per-well paired R2 + RMSE tables (rolling, h1 + h6)
# --------------------------------------------------------------------------

def _rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def block_f(datasets, force):
    """Per-well paired metrics under the rolling protocol at h1 and h6.
    One row per (dataset, model, well): R2, RMSE, paired persistence values,
    gain_r2 = r2 - pers_r2 and rmse_skill = 1 - rmse / pers_rmse.
    Same >=5 test-row filter as _perwell_stats. Persistence appears only as
    the paired pers_* columns (its own gain/skill would be identically 0)."""
    for h in F_HORIZONS:
        out_path = OUT_DIR / f"per_well_metrics_h{h}.csv"
        if out_path.exists() and not force:
            print(f"[F] exists, skip: {out_path}", flush=True)
            continue
        rows = []
        for ds in datasets:
            tr, te, cutoff = _roll_frames(ds, h)
            feat = [c for c in tr.columns if c.startswith("lag_z_")]
            assert not tr[feat].isna().any().any()
            te2 = te.copy()
            rf_params = {"n_estimators": rot.RF["n_estimators"],
                         "max_depth": rot.RF["max_depth"]}
            for m in PANEL_MODELS:
                model = make_model(m, 0, rf_params)
                model.fit(tr[feat], tr["y"])
                te2[f"_pred_{m}"] = np.asarray(model.predict(te[feat]))
                print(f"[F] {ds} h={h} {m}: fitted", flush=True)
            for wid, gg in te2.groupby("well_id"):
                if len(gg) < 5:
                    continue
                y = gg["y"].values
                pers_r2 = r2_score(y, gg["y_persistence"])
                pers_rmse = _rmse(y, gg["y_persistence"])
                for m in PANEL_MODELS:
                    r2 = r2_score(y, gg[f"_pred_{m}"])
                    rmse = _rmse(y, gg[f"_pred_{m}"])
                    rows.append({
                        "dataset": ds, "horizon": h, "model": m,
                        "well_id": wid, "n_test_rows": len(gg),
                        "r2": r2, "rmse": rmse,
                        "pers_r2": pers_r2, "pers_rmse": pers_rmse,
                        "gain_r2": r2 - pers_r2,
                        "rmse_skill": (1.0 - rmse / pers_rmse)
                                      if pers_rmse > 0 else np.nan,
                    })
            pd.DataFrame(rows).to_csv(out_path, index=False)  # checkpoint
            n_wells = len({r['well_id'] for r in rows if r['dataset'] == ds})
            print(f"[F] {ds} h={h}: {n_wells} wells written", flush=True)
        print(f"[F] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------
# Block G: percentile bootstrap 95% CI for median paired gains (B=2000)
# --------------------------------------------------------------------------

def _boot_median_ci(vals, rng, B=BOOT_B):
    """Percentile bootstrap CI for the median of paired per-unit values."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    meds = np.median(vals[idx], axis=1)
    return (float(np.median(vals)),
            float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)),
            int(len(vals)))


def block_g(datasets, force):
    """Bootstrap CIs. Same-fold: resample units (grid cells / countries) from
    same_fold_panel.csv. Rolling: resample wells from the block-F tables.
    Deterministic: single rng seeded BOOT_SEED, fixed iteration order."""
    out_path = OUT_DIR / "bootstrap_ci.csv"
    if out_path.exists() and not force:
        print(f"[G] exists, skip: {out_path}", flush=True)
        return
    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    sf = pd.read_csv(OUT_DIR / "same_fold_panel.csv")
    for ds in [d for d in DATASETS if d in datasets]:
        g = sf[sf["dataset"] == ds]
        for m in PANEL_MODELS:
            point, lo, hi, n = _boot_median_ci(g[f"{m}_r2"] - g["persistence_r2"], rng)
            rows.append({"scope": "same_fold", "dataset": ds, "horizon": None,
                         "model": m, "metric": "median_gain_r2",
                         "point": point, "ci_lo": lo, "ci_hi": hi,
                         "n_units": n, "B": BOOT_B})
            print(f"[G] same_fold {ds} {m}: {point:.4f} [{lo:.4f}, {hi:.4f}] n={n}",
                  flush=True)

    for h in F_HORIZONS:
        p = OUT_DIR / f"per_well_metrics_h{h}.csv"
        if not p.exists():
            print(f"[G] missing {p}, run block F first", flush=True)
            continue
        pw = pd.read_csv(p)
        for ds in [d for d in DATASETS if d in datasets]:
            for m in PANEL_MODELS:
                sub = pw[(pw["dataset"] == ds) & (pw["model"] == m)]
                if sub.empty:
                    continue
                for metric, col in [("median_gain_r2", "gain_r2"),
                                    ("median_rmse_skill", "rmse_skill")]:
                    point, lo, hi, n = _boot_median_ci(sub[col], rng)
                    rows.append({"scope": f"rolling_h{h}", "dataset": ds,
                                 "horizon": h, "model": m, "metric": metric,
                                 "point": point, "ci_lo": lo, "ci_hi": hi,
                                 "n_units": n, "B": BOOT_B})
                    print(f"[G] rolling_h{h} {ds} {m} {metric}: "
                          f"{point:.4f} [{lo:.4f}, {hi:.4f}] n={n}", flush=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[G] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------
# Summary + reconciliation
# --------------------------------------------------------------------------

def summarize():
    summary = {"panel_models": PANEL_MODELS,
               "protocols": {"same_fold": "identical folds/seeds as headline scripts",
                             "rolling_origin": "rot.build_supervised, cutoff q=0.80"}}

    sf = pd.read_csv(OUT_DIR / "same_fold_panel.csv")
    sf_sum = {}
    for ds, g in sf.groupby("dataset"):
        d = {"n_units": int(len(g)),
             "persistence_median_r2": round(float(g["persistence_r2"].median()), 4)}
        for m in PANEL_MODELS:
            med = float(g[f"{m}_r2"].median())
            d[f"{m}_median_r2"] = round(med, 4)
            d[f"{m}_recovery_pct"] = round(d["persistence_median_r2"] / med * 100, 1) if med > 0 else None
            d[f"{m}_median_paired_gain_vs_pers"] = round(
                float((g[f"{m}_r2"] - g["persistence_r2"]).median()), 4)
        best = max(PANEL_MODELS, key=lambda m: d[f"{m}_median_r2"])
        d["best_model"] = best
        d["best_model_median_r2"] = d[f"{best}_median_r2"]
        d["recovery_vs_best_pct"] = d[f"{best}_recovery_pct"]
        sf_sum[ds] = d
    summary["same_fold"] = sf_sum

    rp = pd.read_csv(OUT_DIR / "rolling_panel.csv")
    roll_sum = {}
    for (ds, h), g in rp.groupby(["dataset", "horizon"]):
        pers = g[g["model"] == "persistence"].iloc[0]
        ml = g[g["model"] != "persistence"]
        best = ml.loc[ml["perwell_median_r2"].idxmax()]
        roll_sum[f"{ds}_h{h}"] = {
            "persistence_pooled_r2": round(float(pers["pooled_r2"]), 4),
            "persistence_perwell_r2": round(float(pers["perwell_median_r2"]), 4),
            "best_model": str(best["model"]),
            "best_pooled_r2": round(float(best["pooled_r2"]), 4),
            "best_perwell_r2": round(float(best["perwell_median_r2"]), 4),
            "best_perwell_gain_vs_pers": round(float(best["perwell_median_gain_vs_pers"]), 4),
            "max_perwell_gain_over_pers": round(float(ml["perwell_median_gain_vs_pers"].max()), 4),
        }
    summary["rolling"] = roll_sum

    # reconciliation: RF rows must reproduce the headline numbers
    rec = {}
    old = json.load(open(PROC / "rolling_origin_temporal.json"))
    name_map = {"gems": "gems_de", "ncp": "ncp_cn", "grow": "grow_16"}
    for ds in rp["dataset"].unique():
        for h in rp[rp["dataset"] == ds]["horizon"].unique():
            new_rf = rp[(rp["dataset"] == ds) & (rp["horizon"] == h)
                        & (rp["model"] == "random_forest")]
            if new_rf.empty:
                continue
            old_h = old[name_map[ds]]["horizons"][f"h{int(h)}"]
            diff = abs(float(new_rf.iloc[0]["pooled_r2"]) - old_h["pooled"]["ml_r2"])
            rec[f"{ds}_h{int(h)}_rf_pooled_absdiff"] = round(diff, 4)
    summary["reconcile_rf_vs_headline_json"] = rec
    summary["reconcile_max_absdiff"] = max(rec.values()) if rec else None

    if (OUT_DIR / "seed_stability.csv").exists():
        ss = pd.read_csv(OUT_DIR / "seed_stability.csv")
        sd = ss.groupby(["dataset", "model"])["perwell_median_r2"].std()
        summary["seed_stability_perwell_sd_max"] = round(float(sd.max()), 4)
        summary["seed_stability_pooled_sd_max"] = round(
            float(ss.groupby(["dataset", "model"])["pooled_r2"].std().max()), 4)

    for name in ["lag_sensitivity", "cutoff_sensitivity"]:
        p = OUT_DIR / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            ml = df[df["model"] != "persistence"]
            summary[f"{name}_max_perwell_gain_over_pers"] = round(
                float(ml["perwell_median_gain_vs_pers"].max()), 4)

    # block F: per-well RMSE-skill medians
    f_sum = {}
    for h in F_HORIZONS:
        p = OUT_DIR / f"per_well_metrics_h{h}.csv"
        if not p.exists():
            continue
        pw = pd.read_csv(p)
        for (ds, m), g in pw.groupby(["dataset", "model"]):
            f_sum[f"{ds}_h{h}_{m}"] = {
                "n_wells": int(len(g)),
                "median_gain_r2": round(float(g["gain_r2"].median()), 4),
                "median_rmse_skill": round(float(g["rmse_skill"].median()), 4),
            }
    if f_sum:
        summary["per_well_metrics"] = f_sum
        skills = [v["median_rmse_skill"] for v in f_sum.values()]
        summary["rmse_skill_median_max"] = max(skills)
        summary["rmse_skill_median_min"] = min(skills)

    # block G: bootstrap CIs
    p = OUT_DIR / "bootstrap_ci.csv"
    if p.exists():
        bc = pd.read_csv(p)
        g_sum = {}
        for _, r in bc.iterrows():
            g_sum[f"{r['scope']}_{r['dataset']}_{r['model']}_{r['metric']}"] = {
                "point": round(float(r["point"]), 4),
                "ci95": [round(float(r["ci_lo"]), 4), round(float(r["ci_hi"]), 4)],
                "n_units": int(r["n_units"]),
            }
        summary["bootstrap_ci"] = g_sum
        h1g = bc[(bc["scope"] == "rolling_h1") & (bc["metric"] == "median_gain_r2")]
        if len(h1g):
            summary["bootstrap_rolling_h1_gain_ci_hi_max"] = round(
                float(h1g["ci_hi"].max()), 4)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[S] wrote {OUT_DIR / 'summary.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="ABCDEFGS",
                    help="subset of ABCDEFGS (S=summary)")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--horizons", default=",".join(map(str, ROLL_HORIZONS)))
    ap.add_argument("--models", default=",".join(PANEL_MODELS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    datasets = [d for d in args.datasets.split(",") if d]
    horizons = [int(h) for h in args.horizons.split(",") if h]
    models = [m for m in args.models.split(",") if m]

    if "A" in args.blocks:
        block_a(datasets, args.force)
    if "B" in args.blocks:
        block_b(datasets, horizons, models, args.force)
    if "C" in args.blocks:
        block_c(datasets, args.force)
    if "D" in args.blocks:
        block_d(datasets, args.force)
    if "E" in args.blocks:
        block_e(datasets, args.force)
    if "F" in args.blocks:
        block_f(datasets, args.force)
    if "G" in args.blocks:
        block_g(datasets, args.force)
    if "S" in args.blocks:
        summarize()


if __name__ == "__main__":
    main()
