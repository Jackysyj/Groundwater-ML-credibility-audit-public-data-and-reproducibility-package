#!/usr/bin/env python3
"""Classification-target companion for the index-recovery branch.

Makes the index-recovery branch SYMMETRIC with the spatial branch: the spatial
branch covers both regression (nitrate R^2) and classification (arsenic/fluoride/
Assam AUC); the index-recovery branch previously covered only regression
(SunilVarma / Canada continuous WQI). This script adds the classification target
on the SAME two datasets, so region/components/formula are held constant and ONLY
the target representation changes (continuous index -> its class label).

The circularity is the same, one step further: components -> index (formula) ->
class (thresholds on the index). Predicting the class from the components is still
recovering a deterministic function of the inputs.

Protocol is IDENTICAL to index_recovery_demo.py / index_recovery_robustness.py
(SEED=42, seeds=[42,7,101,2024,31337], 5-fold, same component lists, same 4 axes:
A multi-model / B multi-seed / C subsample / D leave-one-component-out). Only the
metric changes to macro one-vs-rest ROC-AUC (matching the spatial branch's AUC)
plus accuracy. skill = AUC - 0.5 is reported for ACR-comparability.

External demonstrators only; not in corpus, not in any denominator.
Usage: python scripts/index_recovery_classification.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "index_recovery_demo"
OUT = DATA / "index_recovery_classification.json"

SEED = 42
SEEDS = [42, 7, 101, 2024, 31337]
N_FOLDS = 5

INDIA_COMPONENTS = ["pH", "EC", "CO3", "HCO3", "Cl", "SO4", "NO3",
                    "TH", "Ca", "Mg", "Na", "K", "F", "TDS"]
CANADA_COMPONENTS = ["Ammonia (mg/l)", "Biochemical Oxygen Demand (mg/l)",
                     "Dissolved Oxygen (mg/l)", "Orthophosphate (mg/l)",
                     "pH (ph units)", "Temperature (cel)",
                     "Nitrogen (mg/l)", "Nitrate (mg/l)"]


def _load(path, comps, target):
    df = pd.read_csv(DATA / path)
    keep = comps + [target]
    d = df[keep].copy()
    for c in comps:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    X = d[comps].values
    y = LabelEncoder().fit_transform(d[target].astype(str).values)
    return X, y


def _cv_auc(model, X, y, seed):
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    return float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc_ovr").mean())


def _cv_acc(model, X, y, seed):
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    return float(cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean())


def _zoo(seed):
    return {
        "Logistic_linear": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000)),
        "LDA_linear": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "XGBoost_gpu": XGBClassifier(n_estimators=400, max_depth=5, tree_method="hist",
                                     device="cuda", verbosity=0, random_state=seed,
                                     eval_metric="mlogloss"),
        "HistGBM": HistGradientBoostingClassifier(random_state=seed),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=8)),
        "MLP_neural": make_pipeline(StandardScaler(),
                                    MLPClassifier(hidden_layer_sizes=(64, 32),
                                                  max_iter=400, random_state=seed)),
    }


def run(label, path, comps, target, index_kind):
    X, y = _load(path, comps, target)
    A = {name: round(_cv_auc(m, X, y, SEED), 4) for name, m in _zoo(SEED).items()}
    linear_auc = A["Logistic_linear"]
    flex = {k: v for k, v in A.items() if k not in ("Logistic_linear", "LDA_linear")}
    # B multi-seed on the two anchors (linear logistic vs flexible xgboost)
    lg, xg = [], []
    for s in SEEDS:
        lg.append(_cv_auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, s))
        xg.append(_cv_auc(XGBClassifier(n_estimators=400, max_depth=5, tree_method="hist",
                                        device="cuda", verbosity=0, random_state=s,
                                        eval_metric="mlogloss"), X, y, s))
    # C subsample (logistic AUC vs n)
    rng = np.random.RandomState(SEED)
    sizes = [n for n in (500, 2000, 5000, 10000) if n < len(y)] + [len(y)]
    csub = []
    for n in sizes:
        idx = rng.choice(len(y), size=n, replace=False) if n < len(y) else np.arange(len(y))
        csub.append({"n": int(n),
                     "logistic_auc": round(_cv_auc(
                         make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                         X[idx], y[idx], SEED), 4)})
    # D leave-one-component-out (logistic AUC)
    full = _cv_auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, SEED)
    drops = []
    for j, c in enumerate(comps):
        r = _cv_auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                    np.delete(X, j, axis=1), y, SEED)
        drops.append({"dropped": c, "auc": round(r, 4), "delta_vs_full": round(r - full, 4)})
    drops.sort(key=lambda d: d["delta_vs_full"])
    return {
        "dataset": label, "index_kind": index_kind, "n": int(len(y)),
        "n_classes": int(len(set(y))), "n_components": len(comps),
        "metric": "macro one-vs-rest ROC-AUC (skill = AUC-0.5, ACR-comparable)",
        "A_multimodel_auc": A,
        "A_linear_logistic_auc": linear_auc,
        "A_flexible_min_auc": round(min(flex.values()), 4),
        "A_flexible_max_auc": round(max(flex.values()), 4),
        "A_accuracy_logistic": round(_cv_acc(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, SEED), 4),
        "B_multiseed": {"seeds": SEEDS,
                        "logistic_mean": round(float(np.mean(lg)), 4),
                        "logistic_sd": round(float(np.std(lg)), 4),
                        "xgb_mean": round(float(np.mean(xg)), 4),
                        "xgb_sd": round(float(np.std(xg)), 4)},
        "C_subsample_logistic": csub,
        "D_leave_one_component_out": {"full_logistic_auc": round(full, 4),
                                      "per_component": drops,
                                      "max_single_drop": drops[0]["delta_vs_full"]},
    }


def compute():
    return {
        "purpose": "Classification-target companion for the index-recovery branch, "
                   "making it symmetric with the spatial branch (regression + "
                   "classification). Same data as the regression demonstrators; only "
                   "the target changes (continuous index -> its class label). "
                   "External demonstrators only; not in corpus / not in any denominator.",
        "protocol_note": "IDENTICAL protocol to index_recovery_demo/robustness "
                         "(SEED=42, seeds=[42,7,101,2024,31337], 5-fold, same components, "
                         "4 axes). Metric = macro OVR ROC-AUC (matches spatial branch).",
        "seed": SEED, "seeds": SEEDS, "n_folds": N_FOLDS,
        "linear_index": run("SunilVarma India groundwater", "india_gw_wqi.csv",
                            INDIA_COMPONENTS, "Water Quality Classification", "linear"),
        "nonlinear_index": run("Nature Sci Data Canada (CCME)", "nature_canada_ccme.csv",
                              CANADA_COMPONENTS, "CCME_WQI", "nonlinear"),
        "redline_note": "Directly measured per-dataset AUC across learners/seeds/subsamples. "
                        "No composite score, never regressed against an outcome.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()
    fresh = compute()
    if args.check:
        if not OUT.exists():
            print(f"FAIL: {OUT} missing; run without --check first"); sys.exit(1)
        stored = json.load(open(OUT))
        bad = []
        for br in ("linear_index", "nonlinear_index"):
            for k in ("A_linear_logistic_auc", "A_flexible_max_auc"):
                a, b = fresh[br][k], stored[br].get(k)
                if b is None or abs(a - b) > args.tol:
                    bad.append(f"{br}.{k}: recompute={a} stored={b}")
        if bad:
            print("DRIFT DETECTED:"); [print("  " + x) for x in bad]; sys.exit(1)
        print(f"PASS: classification demo within tol {args.tol}"); return
    OUT.write_text(json.dumps(fresh, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}\n")
    for br in ("linear_index", "nonlinear_index"):
        d = fresh[br]
        print(f"=== {d['dataset']} ({d['index_kind']} index, n={d['n']}, {d['n_classes']} classes) ===")
        print("  [A] multi-model macro-AUC (SEED=42):")
        for m, v in d["A_multimodel_auc"].items():
            print(f"       {m:16s} {v:.4f}")
        b = d["B_multiseed"]
        print(f"  [B] 5-seed: Logistic {b['logistic_mean']:.4f}+/-{b['logistic_sd']:.4f}  "
              f"XGB {b['xgb_mean']:.4f}+/-{b['xgb_sd']:.4f}")
        print("  [C] logistic AUC vs n: " +
              "  ".join(f"n{r['n']}={r['logistic_auc']:.3f}" for r in d["C_subsample_logistic"]))
        dd = d["D_leave_one_component_out"]; w = dd["per_component"][0]
        print(f"  [D] full logistic AUC={dd['full_logistic_auc']:.4f}; most damaging = "
              f"{w['dropped']} ({w['delta_vs_full']:+.4f})\n")


if __name__ == "__main__":
    main()
