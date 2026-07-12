#!/usr/bin/env python3
"""Index-recovery branch demonstrator for the Free-Baseline Recovery family.

This is the ALGEBRAIC branch companion to the autocorrelation branch (ACR).
It answers ONE falsifiable question per external dataset: when a Water Quality
Index (WQI) target is a deterministic FORMULA of the very predictors fed to the
model, how much of the "ML skill" is the model merely recovering that formula?

The branch criterion is NOT "OLS approximately equals ML". It is:
    the target is a deterministic function (the index formula) of the predictors.
When that holds, high R^2 is tautological, regardless of the learner:
  - LINEAR index (weighted arithmetic WQI): OLS recovers it exactly (R^2 -> 1),
    and complex ML adds nothing (often less).
  - NON-LINEAR index (CCME piecewise/asymptotic): OLS is a poor fit but a
    flexible learner (XGBoost) still reaches R^2 -> 1. The high ML skill is
    recovery of a known deterministic formula, NOT prediction of any
    environmental signal at an unsampled location.

Both datasets are EXTERNAL DEMONSTRATORS. They are NOT in the audited 539-paper
corpus and are NOT counted in any prevalence denominator. They illustrate the
mechanism of the index_recovery class only. This mirrors the circularity
exclusion applied to the re-test source datasets (Podgorski 514, Nath 441).

Data provenance (both open-license, downloaded 2026-07-06):
  * india_gw_wqi.csv    : SunilVarma24/Water-Quality-Prediction (GitHub),
                          India groundwater, weighted-arithmetic WQI + 14 named
                          components. LINEAR-index demonstrator.
  * nature_canada_ccme.csv : Karim et al. Sci Data 12, 391 (2025),
                          figshare 10.6084/m9.figshare.27800394 (CC BY 4.0),
                          Canada subset, CCME WQI + 8 named components.
                          NON-LINEAR-index demonstrator.

Usage:
  python scripts/index_recovery_demo.py          # recompute, write results JSON
  python scripts/index_recovery_demo.py --check   # drift guard vs stored JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "index_recovery_demo"
OUT = DATA / "index_recovery_demo_results.json"

SEED = 42
N_FOLDS = 5

# India weighted-arithmetic WQI: the 14 named component columns.
INDIA_COMPONENTS = ["pH", "EC", "CO3", "HCO3", "Cl", "SO4", "NO3",
                    "TH", "Ca", "Mg", "Na", "K", "F", "TDS"]
# Canada CCME WQI: the 8 harmonized parameters. Continuous score is CCME_Values
# (0-100); CCME_WQI is the categorical label.
CANADA_COMPONENTS = ["Ammonia (mg/l)", "Biochemical Oxygen Demand (mg/l)",
                     "Dissolved Oxygen (mg/l)", "Orthophosphate (mg/l)",
                     "pH (ph units)", "Temperature (cel)",
                     "Nitrogen (mg/l)", "Nitrate (mg/l)"]


def _cv_r2(model, X, y):
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    return float(cross_val_score(model, X, y, cv=cv, scoring="r2").mean())


def _models():
    ols = make_pipeline(StandardScaler(), LinearRegression())
    quad = make_pipeline(StandardScaler(),
                         PolynomialFeatures(2, include_bias=False),
                         LinearRegression())
    xgb = XGBRegressor(n_estimators=400, max_depth=5,
                       tree_method="hist", device="cuda", verbosity=0,
                       random_state=SEED)
    return ols, quad, xgb


def run_india():
    df = pd.read_csv(DATA / "india_gw_wqi.csv")
    d = df[INDIA_COMPONENTS + ["WQI"]].apply(pd.to_numeric, errors="coerce").dropna()
    X, y = d[INDIA_COMPONENTS].values, d["WQI"].values
    ols, _, xgb = _models()
    r2_ols, r2_xgb = _cv_r2(ols, X, y), _cv_r2(xgb, X, y)
    return {
        "dataset": "SunilVarma India groundwater",
        "index_type": "weighted-arithmetic WQI (LINEAR formula of components)",
        "n": int(len(y)),
        "n_components": len(INDIA_COMPONENTS),
        "components": INDIA_COMPONENTS,
        "wqi_range": [round(float(y.min()), 2), round(float(y.max()), 2)],
        "r2_ols_linear": round(r2_ols, 4),
        "r2_xgboost": round(r2_xgb, 4),
        "ml_increment_over_linear": round(r2_xgb - r2_ols, 4),
        "reading": ("Linear index: OLS recovers the formula exactly (R^2~1) and "
                    "XGBoost adds a NEGATIVE increment. The reported 'ML skill' is "
                    "pure formula recovery; the ML apparatus contributes nothing."),
    }


def run_canada():
    df = pd.read_csv(DATA / "nature_canada_ccme.csv")
    d = df[CANADA_COMPONENTS + ["CCME_Values"]].apply(pd.to_numeric, errors="coerce").dropna()
    X, y = d[CANADA_COMPONENTS].values, d["CCME_Values"].values
    ols, quad, xgb = _models()
    r2_ols, r2_quad, r2_xgb = _cv_r2(ols, X, y), _cv_r2(quad, X, y), _cv_r2(xgb, X, y)
    return {
        "dataset": "Nature Sci Data Canada (CCME)",
        "index_type": "CCME WQI (NON-LINEAR piecewise/asymptotic formula of components)",
        "n": int(len(y)),
        "n_components": len(CANADA_COMPONENTS),
        "components": CANADA_COMPONENTS,
        "wqi_range": [round(float(y.min()), 2), round(float(y.max()), 2)],
        "r2_ols_linear": round(r2_ols, 4),
        "r2_ols_quadratic": round(r2_quad, 4),
        "r2_xgboost": round(r2_xgb, 4),
        "ml_increment_over_linear": round(r2_xgb - r2_ols, 4),
        "reading": ("Non-linear index: OLS is a poor fit (R^2~0.53) so XGBoost shows a "
                    "large apparent 'gain'. But that gain is the flexible learner "
                    "recovering the known CCME formula, NOT predicting environmental "
                    "signal at an unsampled site. This is the guard against reading "
                    "'ML beats linear -> ML is useful' in an index-recovery task: the "
                    "diagnostic is the target-is-a-formula-of-predictors relation, "
                    "not the OLS-vs-ML gap."),
    }


def compute():
    return {
        "branch": "index_recovery (algebraic branch of Free-Baseline Recovery family)",
        "criterion": ("The target is a deterministic FORMULA of the predictors. "
                      "When true, high R^2 is tautological regardless of learner; "
                      "the metric is the relation, not OLS-vs-ML."),
        "scope": ("EXTERNAL DEMONSTRATORS ONLY. Not in the 539-paper audited corpus; "
                  "not counted in any prevalence denominator. Illustrate the "
                  "index_recovery mechanism (35.9% of the water-quality mapping "
                  "literature by the 2026-07-06 predictor re-audit) without "
                  "circularity."),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "demonstrators": {
            "linear_index": run_india(),
            "nonlinear_index": run_canada(),
        },
        "redline_note": ("Per-dataset directly measured R^2 of two learners on the "
                         "same folds. No composite score, never regressed against an "
                         "outcome. Constructive contribution is the family unification "
                         "+ criterion, not a new scalar to rank papers by."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against stored JSON (drift guard)")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="R^2 drift tolerance for --check")
    args = ap.parse_args()

    fresh = compute()

    if args.check:
        if not OUT.exists():
            print(f"FAIL: {OUT} missing; run without --check first")
            sys.exit(1)
        stored = json.load(open(OUT))
        bad = []
        for br in ("linear_index", "nonlinear_index"):
            for k in ("r2_ols_linear", "r2_xgboost"):
                a = fresh["demonstrators"][br][k]
                b = stored["demonstrators"][br].get(k)
                if b is None or abs(a - b) > args.tol:
                    bad.append(f"{br}.{k}: recompute={a} stored={b}")
        if bad:
            print("DRIFT DETECTED:")
            for x in bad:
                print("  " + x)
            sys.exit(1)
        print(f"PASS: index-recovery demo within tol {args.tol} of stored JSON")
        return

    OUT.write_text(json.dumps(fresh, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")
    for br in ("linear_index", "nonlinear_index"):
        d = fresh["demonstrators"][br]
        print(f"\n[{br}] {d['dataset']}  ({d['index_type']})")
        print(f"  n={d['n']}  OLS R^2={d['r2_ols_linear']}  XGB R^2={d['r2_xgboost']}  "
              f"ML increment={d['ml_increment_over_linear']:+}")


if __name__ == "__main__":
    main()
