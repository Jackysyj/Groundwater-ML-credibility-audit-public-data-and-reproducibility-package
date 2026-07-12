#!/usr/bin/env python3
"""
WATER-QUALITY-MAPPING SPATIAL KILL-SHOT, third contaminant (fluoride),
GLOBAL scale.

Third companion to the nitrate (Ransom 2021) and arsenic (Lombard 2021)
spatial kill-shots. Fluoride adds (a) a third contaminant, (b) a genuinely
GLOBAL, six-continent sample, and (c) a continent-level spatial holdout,
which directly answers "can a model trained on some continents predict an
unseen one" -- the strongest form of spatial extrapolation.

Dataset: F_sample.rda from Podgorski & Berg 2022 (Nat Commun global
fluoride, code+sample deposited on ERIC/open, doi 10.25678/0006GQ). 37,132
wells across 6 continents, with lat/lon, the authors' binary target depBIN
(fluoride > 1.5 mg/L WHO guideline), and the 10 predictor variables the
authors' global_F_vars.r actually activates (climate + terrain, all 100%
populated).

HONEST CONSTRAINT: fluoride exceedance is rare here (0.85% global), so some
continents/blocks have few positives and AUC is noisy there; we report
per-fold n and positives, weight by n, and treat this as a GLOBAL-scale
CORROBORATION of the nitrate/arsenic result, not the primary evidence.

Design (target depBIN, metric AUC):
  1. random_kfold          - random k-fold reference (optimistic)
  2. leave_one_continent   - train on all but one continent, test on it
  3. spatial_block_cv      - 5-deg lon/lat blocks with enough positives
A drop from random to leave-one-continent / spatial-block confirms fluoride
mapping leaks spatially like nitrate and arsenic.

Red-line note: authors' physical predictors, measured exceedance target,
measured AUC difference. No composite score.

Output: data/processed/global_fluoride/killshot_results.json
        data/processed/global_fluoride/continent_fold_detail.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
RDA = (ROOT / "data" / "raw" / "water_quality_retest" /
       "podgorski2022_global_fluoride" / "code" / "F_sample.rda")
OUT_DIR = ROOT / "data" / "processed" / "global_fluoride"
OUT_JSON = OUT_DIR / "killshot_results.json"
OUT_FOLDS = OUT_DIR / "continent_fold_detail.csv"

TARGET = "depBIN"
FEATURES = [
    "aet_yr_250mPixel",
    "alpha_Priestley_Taylor_coef_250mPixel",
    "aridity_petDIVprec_WorldClim2_250mPixel",
    "dtm_rough.magnitude_merit.dem_m_250m_s0..0cm_2018_v1.0",
    "dtm_rough.scale_merit.dem_m_250m_s0..0cm_2018_v1.0",
    "dtm_vbf_merit.dem_m_250m_s0..0cm_2017_v1.0",
    "pet_annual_250mPixel",
    "PHIHOX_M_sl7_250m",
    "prec_worldClim2_Jahresdurchschnitt_250mPixel",
    "temperature_250mPixel",
]
BLOCK_DEG = 5.0
MIN_TEST = 100          # a fold needs >= this many wells
MIN_POS = 5            # and >= this many positives to compute a stable AUC

XGB_PARAMS = dict(
    booster="gbtree", max_depth=5, eta=0.05, subsample=0.7,
    colsample_bytree=0.7, min_child_weight=5,
    objective="binary:logistic", eval_metric="auc",
)
NROUNDS = 500


def load():
    df = pyreadr.read_r(RDA)["myDF"]
    df[TARGET] = df[TARGET].astype(int)
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    # CRITICAL: the raw file uses -9999 as a missing-data sentinel; all 10
    # feature columns contain some (temperature has 249). Left as -9999 they
    # would be trained as real values. Convert to NaN so XGBoost's
    # missing=np.nan handles them properly. 257 rows (0.7%) affected.
    for f in FEATURES:
        df[f] = df[f].replace(-9999, np.nan).replace(-9999.0, np.nan)
    return df


def fit_auc(train, test):
    dtr = xgb.DMatrix(train[FEATURES].values, label=train[TARGET].values, missing=np.nan)
    dte = xgb.DMatrix(test[FEATURES].values, label=test[TARGET].values, missing=np.nan)
    m = xgb.train(XGB_PARAMS, dtr, num_boost_round=NROUNDS, verbose_eval=False)
    if test[TARGET].nunique() < 2:
        return np.nan
    return float(roc_auc_score(test[TARGET].values, m.predict(dte)))


def leave_one_continent(df):
    rows, aucs, ns = [], [], []
    for c in sorted(df["Continent"].dropna().unique()):
        te = df[df["Continent"] == c]
        tr = df[df["Continent"] != c]
        if len(te) < MIN_TEST or te[TARGET].sum() < MIN_POS:
            print(f"  skip {c}: n={len(te)}, pos={int(te[TARGET].sum())}", flush=True)
            continue
        auc = fit_auc(tr, te)
        rows.append({"continent": c, "n_test": int(len(te)),
                     "n_pos": int(te[TARGET].sum()),
                     "exceed_rate": float(te[TARGET].mean()), "auc": auc})
        aucs.append(auc); ns.append(len(te))
        print(f"  {c}: n={len(te)}, pos={int(te[TARGET].sum())}, AUC={auc:.3f}", flush=True)
    w = np.array(ns) / sum(ns)
    return {"n_folds": len(aucs), "median_auc": float(np.median(aucs)),
            "mean_auc_weighted": float(np.average(aucs, weights=w)),
            "min_auc": float(np.min(aucs))}, pd.DataFrame(rows)


def spatial_block_cv(df):
    d = df.copy()
    d["block"] = (np.floor(d["lon"] / BLOCK_DEG).astype(int).astype(str) + "_" +
                  np.floor(d["lat"] / BLOCK_DEG).astype(int).astype(str))
    aucs, ns = [], []
    for b, g in d.groupby("block"):
        if len(g) < MIN_TEST or g[TARGET].sum() < MIN_POS:
            continue
        te = g
        tr = d[d["block"] != b]
        auc = fit_auc(tr, te)
        aucs.append(auc); ns.append(len(te))
    w = np.array(ns) / sum(ns)
    return {"n_folds": len(aucs), "median_auc": float(np.median(aucs)),
            "mean_auc_weighted": float(np.average(aucs, weights=w))}


def random_kfold(df, k):
    rng = np.random.default_rng(42)
    fold = rng.integers(0, k, size=len(df))
    aucs, ns = [], []
    for f in range(k):
        te = df[fold == f]; tr = df[fold != f]
        if len(te) < MIN_TEST or te[TARGET].sum() < MIN_POS:
            continue
        aucs.append(fit_auc(tr, te)); ns.append(len(te))
    w = np.array(ns) / sum(ns)
    return {"n_folds": len(aucs), "median_auc": float(np.median(aucs)),
            "mean_auc_weighted": float(np.average(aucs, weights=w))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"wells: {len(df)}, global exceedance rate: {df[TARGET].mean():.4f}, "
          f"continents: {df['Continent'].nunique()}", flush=True)

    print("\n[1] leave-one-continent ...", flush=True)
    loc, detail = leave_one_continent(df)
    detail.to_csv(OUT_FOLDS, index=False, encoding="utf-8-sig")

    print("\n[2] spatial block CV (5 deg) ...", flush=True)
    sb = spatial_block_cv(df)
    print(f"    spatial block median AUC = {sb['median_auc']:.3f} ({sb['n_folds']} blocks)", flush=True)

    print("\n[3] random k-fold reference ...", flush=True)
    rk = random_kfold(df, k=max(loc["n_folds"], 5))
    print(f"    random k-fold median AUC = {rk['median_auc']:.3f}", flush=True)

    results = {
        "dataset": "Podgorski & Berg 2022 global fluoride sample (F_sample.rda, ERIC/open 10.25678/0006GQ)",
        "task_type": "water_quality_mapping, classification (fluoride > 1.5 mg/L WHO guideline), GLOBAL 6-continent",
        "target": TARGET, "metric": "ROC-AUC", "n_features": len(FEATURES),
        "n_wells": int(len(df)), "global_exceedance_rate": float(df[TARGET].mean()),
        "imbalance_caveat": ("Fluoride exceedance is rare (0.85%); per-continent positives range 9-127, so "
            "small-continent AUCs are noisy. Treated as global-scale corroboration, not primary evidence."),
        "random_kfold": rk,
        "leave_one_continent": loc,
        "spatial_block_cv": sb,
        "gap_randomkfold_minus_continent_median_auc": float(rk["median_auc"] - loc["median_auc"]),
        "gap_randomkfold_minus_block_median_auc": float(rk["median_auc"] - sb["median_auc"]),
    }
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print("\n=== GLOBAL FLUORIDE MAPPING SPATIAL KILL-SHOT ===")
    print(f"  random k-fold        median AUC = {rk['median_auc']:.3f}")
    print(f"  spatial block (5deg) median AUC = {sb['median_auc']:.3f}")
    print(f"  leave-one-CONTINENT  median AUC = {loc['median_auc']:.3f} (min {loc['min_auc']:.3f})")
    print(f"  GAP (randomkfold - continent) = {results['gap_randomkfold_minus_continent_median_auc']:.3f}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
