#!/usr/bin/env python3
"""
WATER-QUALITY-MAPPING SPATIAL KILL-SHOT, second contaminant (arsenic).

Companion to killshot_usgs_nitrate_spatial.py. Tests whether the severe
spatial-leakage of water-quality MAPPING generalizes across contaminants
(nitrate -> arsenic), so the task-type thesis is not a nitrate artefact.

Dataset: Lombard et al. 2021 (ES&T 10.1021/acs.est.0c05239), USGS Data
Release (ScienceBase 5f2d4ce382ceae4cb3c2e1d6). AsModelInput.csv: 20,450
private wells, arsenic exceedance targets (bas1/bas5/bas10 binary, As3Cat
3-class) with the authors' own train/holdout split flags (spl*), and 85
predictor variables (geology, aquifer unit, soil geochemistry, climate,
ecoregion). Classification task, so skill is measured by ROC-AUC, not R2.

IMPORTANT DATA CONSTRAINT (itself a finding): the public file has NO
per-well coordinates and NO well ID, so a lon/lat spatial-block CV (as used
for nitrate) is impossible. Instead we use the EPA Level III ECOREGION
(na_* one-hot columns) as the spatial holdout unit: leave-one-ecoregion-out
is a standard environmental-blocking form of spatial CV (Roberts et al.
2017). Ecoregions are compact geographic-climatic regions, so holding one
out is a genuine spatial-extrapolation test. The ecoregion columns are
REMOVED from the feature set when they define the holdout, so the model
cannot simply read off the held-out group.

Design (target = bas10, arsenic > 10 ug/L MCL exceedance; AUC):
  1. authors_random_holdout - authors' spl10 flag (their reported skill)
  2. random_kfold_matched   - random k-fold, k = #ecoregions, same subset
  3. leave_one_ecoregion    - train on all but one ecoregion, test on it
A large AUC drop from random to leave-one-ecoregion confirms arsenic
mapping leaks spatially like nitrate.

Red-line note: predictors are the authors' physical variables; targets are
measured exceedances; outcome is a measured AUC difference. No composite score.

Output: data/processed/usgs_arsenic/killshot_results.json
        data/processed/usgs_arsenic/ecoregion_fold_detail.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = (ROOT / "data" / "raw" / "water_quality_retest" /
        "lombard2021_usgs_conus_arsenic" / "AsModelInput.csv")
OUT_DIR = ROOT / "data" / "processed" / "usgs_arsenic"
OUT_JSON = OUT_DIR / "killshot_results.json"
OUT_FOLDS = OUT_DIR / "ecoregion_fold_detail.csv"

TARGET = "bas10"                 # arsenic > 10 ug/L (EPA MCL) exceedance, binary
SPLIT = "spl10"                  # authors' train(TRUE)/holdout(FALSE) flag for bas10
TARGETS = ["bas1", "bas5", "bas10", "As3Cat"]
SPLITS = ["spl1", "spl5", "spl10", "spl3cat"]
MIN_TEST = 60                    # an ecoregion needs >= this many wells to be a fold

XGB_PARAMS = dict(
    booster="gbtree", max_depth=5, eta=0.05, subsample=0.7,
    colsample_bytree=0.7, min_child_weight=5,
    objective="binary:logistic", eval_metric="auc",
)
NROUNDS = 600


def load():
    d = pd.read_csv(DATA)
    na_cols = [c for c in d.columns if c.startswith("na_")]
    # feature set: everything except targets, splits, and the ecoregion
    # one-hot columns (removed so the held-out group is not directly readable)
    feats = [c for c in d.columns if c not in TARGETS + SPLITS + na_cols]
    feats = [c for c in feats if np.issubdtype(pd.to_numeric(d[c], errors="coerce").dtype, np.number)]
    # keep wells with a defined ecoregion (one-hot sums to 1) and a target
    d = d[d[na_cols].sum(axis=1) > 0].copy()
    d = d[d[TARGET].notna()].copy()
    d["eco"] = d[na_cols].idxmax(axis=1)
    return d, feats


def fit_auc(train, test, feats):
    dtr = xgb.DMatrix(train[feats].values, label=train[TARGET].values, missing=np.nan)
    dte = xgb.DMatrix(test[feats].values, label=test[TARGET].values, missing=np.nan)
    m = xgb.train(XGB_PARAMS, dtr, num_boost_round=NROUNDS, verbose_eval=False)
    pred = m.predict(dte)
    if test[TARGET].nunique() < 2:
        return np.nan
    return float(roc_auc_score(test[TARGET].values, pred))


def authors_random_holdout(d, feats):
    tr = d[d[SPLIT] == True]
    te = d[d[SPLIT] == False]
    return {"auc": fit_auc(tr, te, feats), "n_train": int(len(tr)), "n_test": int(len(te))}


def leave_one_ecoregion(d, feats):
    ecos = sorted(d["eco"].unique())
    rows, aucs, ns = [], [], []
    for e in ecos:
        te = d[d["eco"] == e]
        tr = d[d["eco"] != e]
        if len(te) < MIN_TEST or te[TARGET].nunique() < 2:
            continue
        auc = fit_auc(tr, te, feats)
        rows.append({"ecoregion": e, "n_test": int(len(te)),
                     "exceed_rate": float(te[TARGET].mean()), "auc": auc})
        aucs.append(auc); ns.append(len(te))
        print(f"  eco {e}: n_test={len(te)}, exceed={te[TARGET].mean():.3f}, AUC={auc:.3f}", flush=True)
    detail = pd.DataFrame(rows)
    w = np.array(ns) / sum(ns)
    return {
        "n_folds": len(aucs),
        "median_auc": float(np.median(aucs)),
        "mean_auc_weighted": float(np.average(aucs, weights=w)),
        "min_auc": float(np.min(aucs)),
        "frac_auc_below_0p6": float(np.mean(np.array(aucs) < 0.6)),
    }, detail


def random_kfold_matched(d, feats, k):
    rng = np.random.default_rng(42)
    fold = rng.integers(0, k, size=len(d))
    aucs, ns = [], []
    for f in range(k):
        te = d[fold == f]; tr = d[fold != f]
        if len(te) < MIN_TEST or te[TARGET].nunique() < 2:
            continue
        aucs.append(fit_auc(tr, te, feats)); ns.append(len(te))
    w = np.array(ns) / sum(ns)
    return {"n_folds": len(aucs), "median_auc": float(np.median(aucs)),
            "mean_auc_weighted": float(np.average(aucs, weights=w))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d, feats = load()
    print(f"wells (ecoregion-defined): {len(d)}, features: {len(feats)}, "
          f"ecoregions: {d['eco'].nunique()}", flush=True)
    print(f"target {TARGET} prevalence: {d[TARGET].mean():.3f}", flush=True)

    print("\n[1] authors' random holdout (spl10) ...", flush=True)
    rh = authors_random_holdout(d, feats)
    print(f"    random holdout AUC = {rh['auc']:.3f}", flush=True)

    print("\n[2] leave-one-ecoregion-out ...", flush=True)
    leo, detail = leave_one_ecoregion(d, feats)
    detail.to_csv(OUT_FOLDS, index=False, encoding="utf-8-sig")

    print("\n[3] random k-fold matched reference ...", flush=True)
    rk = random_kfold_matched(d, feats, k=leo["n_folds"])

    gap = rk["median_auc"] - leo["median_auc"]
    results = {
        "dataset": "Lombard et al. 2021 USGS CONUS arsenic private wells, AsModelInput.csv",
        "task_type": "water_quality_mapping (spatial extrapolation), classification",
        "target": "bas10 (arsenic > 10 ug/L MCL exceedance)",
        "skill_metric": "ROC-AUC",
        "spatial_holdout_unit": "EPA Level III ecoregion (leave-one-ecoregion-out)",
        "coordinate_constraint_note": ("Public file has no per-well coordinates or ID; "
            "lon/lat block CV impossible. Ecoregion is the environmental-blocking spatial "
            "holdout. This coordinate suppression is itself part of the reproducibility finding."),
        "n_features": len(feats),
        "n_wells": int(len(d)),
        "authors_random_holdout": rh,
        "leave_one_ecoregion": leo,
        "random_kfold_matched": rk,
        "gap_randomkfold_minus_ecoregion_median_auc": float(gap),
        "contrast_note": ("Nitrate mapping (Ransom 2021) collapsed under spatial block CV "
            "(R2 0.49 -> 0.15). If arsenic AUC also drops markedly from random to leave-one-"
            "ecoregion, water-quality mapping leakage is confirmed across contaminants."),
    }
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print("\n=== ARSENIC MAPPING SPATIAL KILL-SHOT ===")
    print(f"  authors random holdout AUC   = {rh['auc']:.3f}")
    print(f"  random k-fold matched median = {rk['median_auc']:.3f}")
    print(f"  leave-one-ECOREGION   median = {leo['median_auc']:.3f} "
          f"(min {leo['min_auc']:.3f}, folds<0.6: {leo['frac_auc_below_0p6']:.0%})")
    print(f"  GAP (randomkfold - ecoregion) = {gap:.3f}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
