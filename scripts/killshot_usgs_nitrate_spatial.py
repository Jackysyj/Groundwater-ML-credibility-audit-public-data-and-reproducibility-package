#!/usr/bin/env python3
"""
WATER-QUALITY-MAPPING SPATIAL KILL-SHOT.

Tests the reframed thesis (task type, not dataset/region, governs spatial-
leakage severity): water-quality MAPPING tasks must extrapolate to
unsampled locations and therefore leak severely under random splitting,
UNLIKE lagged-self water-level tasks (shown immune across 16 GROW
countries, gap < 0.05).

Dataset: Ransom et al. 2021 (STOTEN 10.1016/j.scitotenv.2021.151065), USGS
Data Release. 12,082 wells across the conterminous US, measured nitrate
(target LNO3 = ln nitrate), with the authors' RFE-selected predictor set
(well characteristics, hydrology, soil, geology, land use, N inputs). This
is a canonical published national-scale nitrate mapping model; the authors
reported training R2 = 0.83 and a RANDOM hold-out R2 = 0.49.

We reuse the AUTHORS' EXACT features and target and XGBoost hyper-parameters
and change ONLY the validation geometry, so the contrast is clean:
  1. random_holdout   - the authors' own spl flag (reproduce ~0.49); the
                        number literature reports and readers trust
  2. spatial_block_cv - leave-one-spatial-block-out over a lon/lat grid
                        (default 4 deg cells); held-out block is a compact
                        region never seen in training = the real deployment
                        setting for a contamination map
  3. random_kfold     - k-fold with random assignment, matched fold count,
                        as an apples-to-apples optimistic reference for (2)

If spatial-block R2 collapses far below the random R2 (large positive gap),
water-quality mapping is confirmed to leak severely under random splitting,
completing the task-type contrast with the immune water-level task. Numbers
are written to manuscript_stats.json-ready JSON.

Red-line note: predictors and target are the authors' physical variables;
the outcome is a directly measured R2 difference. No composite score.

Output: data/processed/usgs_nitrate/killshot_results.json
        data/processed/usgs_nitrate/spatial_block_fold_detail.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = (ROOT / "data" / "raw" / "water_quality_retest" /
        "ransom2021_usgs_conus_nitrate" / "National_NO3" / "Inputs" /
        "training_and_holdout_data.txt")
OUT_DIR = ROOT / "data" / "processed" / "usgs_nitrate"
OUT_JSON = OUT_DIR / "killshot_results.json"
OUT_FOLDS = OUT_DIR / "spatial_block_fold_detail.csv"

TARGET = "LNO3"
LAT, LON = "DEC_LAT_VA", "DEC_LONG_VA"
BLOCK_DEG = 4.0            # spatial block size in degrees (~440 km lat)
MIN_TEST_WELLS = 50        # a block must hold >= this many wells to be a fold
SEED = 1182                # authors' seed

# Authors' RFE-selected final variables (from Source code/xgb_model.R).
FINAL_VARS = [
    "DEPTH","us_ppt1981_mmyr","WTDEPL_m","PET_mmyr","TOP","trans","DrnClass_9_mean",
    "rech48","dep_no3_1985","AVG_NO4_mean","LP8","pden_1990","dep_nh4_1992",
    "X1982_LU50","wc_avg","DSD8","LR_arsenic","TWI","nfarm_1974","us_tave198_degC",
    "HYDCLASS_mean","BFI48","avg_bd_mean","avg_om_mean","AWS25_mean","TOP5_Mg",
    "X1974_LU43","SubsurfContactTime","STRM_DIST","ET_Reitz","dep_nh4_1985",
    "DrnClass_4_mean","X1974_LU50","dep_no3_1992","C_Mn","DSD2","LP7","nconf_1982",
    "runoff_Reitz","nucnf_1992","DSD1","DrnClass_8_mean","C_C_Tot","DSD7","LP6","LP5",
    "StreamDensity","tt_total","LP2","LP9","hydgrp_B_mean","X1992_LU50","DSD9",
    "avg_awc_mean","X1982_LU43","AVG_NO10_mean","DSD6","pct_wells","DrnClass_6_mean",
    "TOP5_Fe","nucnf_1982","dtw","LP3","LP1","hydgrp_CD_mean","avg_kfact_mean",
    "DrnClass_3_mean","avg_silt_mean","avg_kv_mean","C_As","DSD3","TOP5_Mn","LP4",
    "DSD4","C_Mg",
    # one categorical dummy the authors kept; approximate with the raw cat below
]

# Authors' oneSE XGBoost params (from xgb_model.R).
XGB_PARAMS = dict(
    booster="gbtree", max_depth=6, eta=0.01, gamma=10,
    colsample_bytree=1.0, min_child_weight=10, subsample=0.5,
    objective="reg:squarederror", eval_metric="rmse",
)
NROUNDS = 1500


def load_data():
    df = pd.read_csv(DATA, sep="\t")
    # keep only numeric final vars that exist (drop the one categorical dummy
    # placeholder; 75 numeric predictors remain, matching the model intent)
    feats = [v for v in FINAL_VARS if v in df.columns
             and np.issubdtype(df[v].dtype, np.number)]
    keep = feats + [TARGET, LAT, LON, "spl"]
    df = df[keep].copy()
    df = df[df[TARGET].notna() & df[LAT].notna() & df[LON].notna()]
    return df, feats


def fit_eval(train, test, feats):
    dtr = xgb.DMatrix(train[feats].values, label=train[TARGET].values,
                      missing=np.nan)
    dte = xgb.DMatrix(test[feats].values, label=test[TARGET].values,
                      missing=np.nan)
    m = xgb.train(XGB_PARAMS, dtr, num_boost_round=NROUNDS, verbose_eval=False)
    pred = m.predict(dte)
    return (float(r2_score(test[TARGET], pred)),
            float(mean_squared_error(test[TARGET], pred) ** 0.5))


def random_holdout(df, feats):
    """Authors' own random split (spl TRUE=train / FALSE=holdout)."""
    tr = df[df["spl"] == True]
    te = df[df["spl"] == False]
    r2, rmse = fit_eval(tr, te, feats)
    return {"r2": r2, "rmse": rmse, "n_train": len(tr), "n_test": len(te)}


def assign_blocks(df):
    gx = np.floor(df[LON] / BLOCK_DEG).astype(int)
    gy = np.floor(df[LAT] / BLOCK_DEG).astype(int)
    return (gx.astype(str) + "_" + gy.astype(str)).values


def spatial_block_cv(df, feats):
    """Leave-one-spatial-block-out. Each block with >= MIN_TEST_WELLS is a
    held-out fold; train on all wells outside it."""
    df = df.assign(block=assign_blocks(df))
    sizes = df["block"].value_counts()
    fold_blocks = sizes[sizes >= MIN_TEST_WELLS].index.tolist()
    rows, r2s, rmses, ns = [], [], [], []
    for b in fold_blocks:
        te = df[df["block"] == b]
        tr = df[df["block"] != b]
        r2, rmse = fit_eval(tr, te, feats)
        rows.append({"block": b, "n_test": len(te), "r2": r2, "rmse": rmse})
        r2s.append(r2); rmses.append(rmse); ns.append(len(te))
        print(f"  block {b}: n_test={len(te)}, R2={r2:.3f}", flush=True)
    detail = pd.DataFrame(rows)
    w = np.array(ns) / sum(ns)
    return {
        "n_folds": len(fold_blocks),
        "block_deg": BLOCK_DEG,
        "median_r2": float(np.median(r2s)),
        "mean_r2_weighted": float(np.average(r2s, weights=w)),
        "frac_negative_r2": float(np.mean(np.array(r2s) < 0)),
        "pooled_test_wells": int(sum(ns)),
    }, detail


def random_kfold_matched(df, feats, k):
    """Random k-fold with the same number of folds as the spatial CV, as an
    optimistic apples-to-apples reference (same data, random assignment)."""
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, k, size=len(df))
    r2s, ns = [], []
    for f in range(k):
        te = df[fold == f]; tr = df[fold != f]
        if len(te) < MIN_TEST_WELLS:
            continue
        r2, _ = fit_eval(tr, te, feats)
        r2s.append(r2); ns.append(len(te))
    w = np.array(ns) / sum(ns)
    return {"n_folds": len(r2s), "median_r2": float(np.median(r2s)),
            "mean_r2_weighted": float(np.average(r2s, weights=w))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feats = load_data()
    print(f"wells: {len(df)}, numeric features: {len(feats)}", flush=True)

    print("\n[1] random holdout (authors' spl) ...", flush=True)
    rh = random_holdout(df, feats)
    print(f"    random holdout R2 = {rh['r2']:.3f} "
          f"(authors reported ~0.49)", flush=True)

    print("\n[2] spatial block CV ...", flush=True)
    sb, detail = spatial_block_cv(df, feats)
    detail.to_csv(OUT_FOLDS, index=False, encoding="utf-8-sig")

    print("\n[3] random k-fold matched reference ...", flush=True)
    rk = random_kfold_matched(df, feats, k=sb["n_folds"])

    gap = rk["median_r2"] - sb["median_r2"]
    results = {
        "dataset": "Ransom et al. 2021 USGS CONUS nitrate, 12082 wells",
        "task_type": "water_quality_mapping (spatial extrapolation)",
        "target": "LNO3 (ln nitrate)",
        "n_features_numeric": len(feats),
        "random_holdout_authors_spl": rh,
        "spatial_block_cv": sb,
        "random_kfold_matched": rk,
        "gap_randomkfold_minus_spatialblock_median_r2": float(gap),
        "verdict_note": ("Compare against the water-LEVEL task (16 GROW countries, "
                         "median gap < 0.05, no collapse). A large positive gap here "
                         "confirms task-type governs spatial-leakage severity."),
    }
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print("\n=== WATER-QUALITY MAPPING SPATIAL KILL-SHOT ===")
    print(f"  random holdout (authors)      R2 = {rh['r2']:.3f}")
    print(f"  random k-fold (matched)  median R2 = {rk['median_r2']:.3f}")
    print(f"  SPATIAL BLOCK CV         median R2 = {sb['median_r2']:.3f}  "
          f"(neg-R2 folds: {sb['frac_negative_r2']:.0%})")
    print(f"  GAP (randomkfold - spatialblock) = {gap:.3f}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
