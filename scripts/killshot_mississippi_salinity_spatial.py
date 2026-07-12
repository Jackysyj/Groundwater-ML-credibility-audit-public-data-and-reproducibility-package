#!/usr/bin/env python3
"""
Mississippi Embayment salinity spatial-holdout probe.

Purpose
-------
Add a continuous water-quality regression dataset to the spatial kill-shot
family. The dataset is the USGS Mississippi Embayment salinity data release
(10.5066/P9WBFR1T), with point-level wells, latitude/longitude, author model
features, and continuous salinity targets:

  - specific conductance (p00095; model target log_p00095)
  - chloride (p00940; model target log_p00940)

The core contrast mirrors the existing nitrate/arsenic/fluoride scripts:
same target and predictors, but random/interpolative validation is compared
against compact spatial-block holdout. A zero-learning IDW baseline is also
scored on the same folds for a later ACR/FBR-style comparison.

This script writes processed outputs only. It does not update
manuscript_stats.json or data/processed/truth_table.json.

Smoke mode is intentionally CPU-light so it can run while a GPU-heavy job is
active:

    python3 scripts/killshot_mississippi_salinity_spatial.py --smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "spatial_wq_external_candidates" / "mississippi_embayment_salinity_usgs_P9WBFR1T"
DEFAULT_OUT = ROOT / "data" / "processed" / "mississippi_salinity_spatial"
EXP_VARS = RAW_DIR / "ExpVars.csv"

LAT = "Lat"
LON = "Long"
LAYER = "Layer2"
SEED = 20260707
EARTH_KM = 6371.0


@dataclass(frozen=True)
class TargetSpec:
    key: str
    zip_name: str
    member: str
    value_col: str
    log_col: str
    expvar_flag_col: str
    units: str


TARGETS = {
    "specific_conductance": TargetSpec(
        key="specific_conductance",
        zip_name="specific_conductance.zip",
        member="specific_conductance/Inputwells_p00095.txt",
        value_col="p00095",
        log_col="log_p00095",
        expvar_flag_col="SC_Model",
        units="uS/cm",
    ),
    "chloride": TargetSpec(
        key="chloride",
        zip_name="chloride.zip",
        member="chloride/Inputwells_p00940.txt",
        value_col="p00940",
        log_col="log_p00940",
        expvar_flag_col="Cl_Model",
        units="mg/L",
    ),
}

LEAK_OR_META_COLS = {
    LAT, LON, "YEAR", "mtype", "mPred", "mPred_o",
    "p00095", "log_p00095", "p00940", "log_p00940",
}


def read_target(spec: TargetSpec, smoke: bool = False, smoke_n: int = 1200) -> tuple[pd.DataFrame, list[str]]:
    """Read one target from the USGS zip and return a clean dataframe plus
    author-selected feature names from ExpVars.csv."""
    zpath = RAW_DIR / spec.zip_name
    with zipfile.ZipFile(zpath) as z:
        df = pd.read_csv(z.open(spec.member))

    exp = pd.read_csv(EXP_VARS)
    selected = exp.loc[exp[spec.expvar_flag_col].fillna("").astype(str).str.lower().eq("x"), "VarName"].tolist()
    features = [c for c in selected if c in df.columns and c not in LEAK_OR_META_COLS]

    # Layer is a mappable hydrogeologic unit and useful for layer sensitivity,
    # but the author archive models raster predictions layer-by-layer. Keep it
    # as an optional one-hot feature only when present in the well table.
    if LAYER in df.columns and LAYER not in features:
        features.append(LAYER)

    keep = sorted(set(features + [LAT, LON, LAYER, "mtype", "mPred", "mPred_o", spec.value_col, spec.log_col]) & set(df.columns))
    df = df[keep].copy()
    df = df[df[spec.log_col].notna() & df[LAT].notna() & df[LON].notna()].copy()
    df["_target"] = df[spec.log_col].astype(float)
    df["_target_raw"] = df[spec.value_col].astype(float)

    if smoke and len(df) > smoke_n:
        # Spatially unbiased smoke sample. It is for pipeline validation only,
        # not for manuscript numbers.
        df = df.sample(n=smoke_n, random_state=SEED).sort_index().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df, features


def encode_features(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = df[features].copy()
    x = pd.get_dummies(x, columns=[c for c in x.columns if x[c].dtype == "object"], dummy_na=True)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x, x.columns.tolist()


def block_id(df: pd.DataFrame, block_deg: float) -> pd.Series:
    gx = np.floor(df[LON].astype(float) / block_deg).astype(int)
    gy = np.floor(df[LAT].astype(float) / block_deg).astype(int)
    return gx.astype(str) + "_" + gy.astype(str)


def make_spatial_splits(df: pd.DataFrame, block_deg: float, min_test: int, max_folds: int | None) -> list[tuple[np.ndarray, np.ndarray, str]]:
    blocks = block_id(df, block_deg)
    sizes = blocks.value_counts()
    valid = sizes[sizes >= min_test].index.tolist()
    # Largest folds first gives a stable smoke test and avoids tiny noisy folds.
    valid = sorted(valid, key=lambda b: sizes[b], reverse=True)
    if max_folds:
        valid = valid[:max_folds]
    splits = []
    for b in valid:
        te = np.flatnonzero((blocks == b).to_numpy())
        tr = np.flatnonzero((blocks != b).to_numpy())
        splits.append((tr, te, str(b)))
    return splits


def make_random_splits(n: int, k: int, min_test: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    chunks = np.array_split(perm, k)
    splits = []
    all_idx = np.arange(n)
    for i, te in enumerate(chunks):
        if len(te) < min_test:
            continue
        mask = np.ones(n, dtype=bool)
        mask[te] = False
        splits.append((all_idx[mask], te, f"random_{i}"))
    return splits


def make_authors_holdout_split(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, str]]:
    if "mtype" not in df.columns:
        return []
    m = df["mtype"].astype(str).str.lower()
    train = np.flatnonzero(m.eq("training").to_numpy())
    test = np.flatnonzero(m.eq("holdout").to_numpy())
    if len(train) == 0 or len(test) == 0:
        return []
    return [(train, test, "authors_holdout")]


def xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la = np.radians(lat.astype(float))
    lo = np.radians(lon.astype(float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def idw_predict(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, k: int) -> np.ndarray:
    tr_xyz = xyz(df.iloc[train_idx][LAT].to_numpy(), df.iloc[train_idx][LON].to_numpy())
    te_xyz = xyz(df.iloc[test_idx][LAT].to_numpy(), df.iloc[test_idx][LON].to_numpy())
    y = df.iloc[train_idx]["_target"].to_numpy(dtype=float)
    kk = min(k, max(1, len(train_idx) - 1))
    preds = np.empty(len(test_idx), dtype=float)
    step = 1000
    for i in range(0, len(test_idx), step):
        chunk = te_xyz[i:i + step]
        d2 = ((chunk[:, None, :] - tr_xyz[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        di = np.take_along_axis(d2, idx, axis=1) ** 0.5 * EARTH_KM
        w = 1.0 / np.clip(di, 1e-4, None)
        preds[i:i + step] = (y[idx] * w).sum(axis=1) / w.sum(axis=1)
    return preds


def fit_predict_model(model: str, x: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                      device: str, nrounds: int, nthread: int, seed: int) -> np.ndarray:
    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y[train_idx]

    if model == "ridge":
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(x_train, y_train)
        return pipe.predict(x_test)

    if model == "histgbm":
        est = HistGradientBoostingRegressor(max_iter=max(50, min(nrounds, 300)), learning_rate=0.05,
                                            max_leaf_nodes=31, random_state=seed)
        est.fit(x_train, y_train)
        return est.predict(x_test)

    if model == "xgboost":
        params = {
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 5,
            "eta": 0.04,
            "subsample": 0.75,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "tree_method": "hist",
            "device": device,
            "nthread": nthread,
            "seed": seed,
        }
        dtr = xgb.DMatrix(x_train.to_numpy(dtype=np.float32), label=y_train, missing=np.nan)
        dte = xgb.DMatrix(x_test.to_numpy(dtype=np.float32), missing=np.nan)
        booster = xgb.train(params, dtr, num_boost_round=nrounds, verbose_eval=False)
        return booster.predict(dte)

    raise ValueError(f"Unknown model: {model}")


def score_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, pred))),
    }


def run_splits(target: str, df: pd.DataFrame, x: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray, str]],
               regime: str, models: list[str], args: argparse.Namespace) -> list[dict]:
    y = df["_target"].to_numpy(dtype=float)
    rows = []
    for train_idx, test_idx, fold in splits:
        y_true = y[test_idx]

        pred = idw_predict(df, train_idx, test_idx, k=args.idw_k)
        score = score_predictions(y_true, pred)
        rows.append({
            "target": target, "regime": regime, "fold": fold, "model": f"idw_k{args.idw_k}",
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
            "r2": score["r2"], "rmse": score["rmse"],
        })

        if regime == "authors_holdout" and "mPred" in df.columns:
            published = pd.to_numeric(df.iloc[test_idx]["mPred"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(published)
            if ok.sum() > 0:
                score = score_predictions(y_true[ok], published[ok])
                rows.append({
                    "target": target, "regime": regime, "fold": fold, "model": "published_mPred",
                    "n_train": int(len(train_idx)), "n_test": int(ok.sum()),
                    "r2": score["r2"], "rmse": score["rmse"],
                })

        for model in models:
            pred = fit_predict_model(model, x, y, train_idx, test_idx, args.device, args.nrounds, args.nthread, args.seed)
            score = score_predictions(y_true, pred)
            rows.append({
                "target": target, "regime": regime, "fold": fold, "model": model,
                "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                "r2": score["r2"], "rmse": score["rmse"],
            })
            print(f"{target:20s} {regime:15s} {fold:14s} {model:8s} "
                  f"n_test={len(test_idx):4d} R2={score['r2']:.3f}", flush=True)
    return rows


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    out = []
    for (target, regime, model), g in df.groupby(["target", "regime", "model"], sort=True):
        weights = g["n_test"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        out.append({
            "target": target,
            "regime": regime,
            "model": model,
            "n_folds": int(len(g)),
            "pooled_n_test": int(g["n_test"].sum()),
            "median_r2": float(g["r2"].median()),
            "weighted_mean_r2": float(np.average(g["r2"], weights=weights)),
            "median_rmse": float(g["rmse"].median()),
        })
    return pd.DataFrame(out)


def build_inventory(target_records: list[dict], args: argparse.Namespace) -> dict:
    return {
        "dataset": "Mississippi Embayment salinity, USGS data release 10.5066/P9WBFR1T",
        "raw_dir": str(RAW_DIR.relative_to(ROOT)),
        "output_dir": relpath(args.output_dir),
        "mode": "smoke" if args.smoke else "full",
        "device": args.device,
        "targets": target_records,
        "validation": {
            "random_kfold": "random folds matched to the number of valid spatial-block folds",
            "spatial_block": {"block_deg": args.block_deg, "min_test": args.min_test},
            "authors_holdout": "USGS archive mtype Training/Holdout split, when present",
            "idw_baseline": f"inverse-distance weighted zero-learning baseline, k={args.idw_k}",
        },
        "status": "smoke output is for pipeline validation only; full results not yet promoted to manuscript truth table",
    }


def relpath(path: Path) -> str:
    """Display a path relative to the project root when possible."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="specific_conductance,chloride",
                    help="comma-separated target keys")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=1200)
    ap.add_argument("--block-deg", type=float, default=0.5)
    ap.add_argument("--min-test", type=int, default=80)
    ap.add_argument("--max-spatial-folds", type=int, default=None)
    ap.add_argument("--models", default=None,
                    help="comma-separated models: xgboost,ridge,histgbm")
    ap.add_argument("--nrounds", type=int, default=None)
    ap.add_argument("--nthread", type=int, default=None)
    ap.add_argument("--idw-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.smoke:
        args.nrounds = args.nrounds or 80
        args.nthread = args.nthread or 2
        args.max_spatial_folds = args.max_spatial_folds or 3
        args.min_test = min(args.min_test, 50)
        default_models = "xgboost,ridge"
    else:
        args.nrounds = args.nrounds or 500
        args.nthread = args.nthread or 8
        default_models = "xgboost,ridge,histgbm"

    models = [m.strip() for m in (args.models or default_models).split(",") if m.strip()]
    bad = set(models) - {"xgboost", "ridge", "histgbm"}
    if bad:
        raise SystemExit(f"Unsupported model(s): {sorted(bad)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    inventory_targets: list[dict] = []
    for key in [t.strip() for t in args.targets.split(",") if t.strip()]:
        if key not in TARGETS:
            raise SystemExit(f"Unknown target {key}; allowed: {sorted(TARGETS)}")
        spec = TARGETS[key]
        df, features = read_target(spec, smoke=args.smoke, smoke_n=args.smoke_n)
        x, encoded_features = encode_features(df, features)
        spatial_splits = make_spatial_splits(df, args.block_deg, args.min_test, args.max_spatial_folds)
        k = max(2, len(spatial_splits))
        random_splits = make_random_splits(len(df), k=k, min_test=args.min_test, seed=args.seed)
        authors_splits = make_authors_holdout_split(df)

        inventory_targets.append({
            "target": key,
            "value_col": spec.value_col,
            "model_target": spec.log_col,
            "units": spec.units,
            "n_rows": int(len(df)),
            "n_unique_coordinates": int(df[[LAT, LON]].drop_duplicates().shape[0]),
            "n_author_features": int(len(features)),
            "n_encoded_features": int(len(encoded_features)),
            "layers": df[LAYER].value_counts(dropna=False).to_dict() if LAYER in df.columns else {},
            "target_raw_summary": {k: float(v) for k, v in df["_target_raw"].describe(percentiles=[0.05, 0.5, 0.95]).to_dict().items()},
            "target_log_summary": {k: float(v) for k, v in df["_target"].describe(percentiles=[0.05, 0.5, 0.95]).to_dict().items()},
            "n_spatial_folds": int(len(spatial_splits)),
        })

        print(f"\n## {key}: n={len(df)}, features={len(features)} encoded={len(encoded_features)}, "
              f"spatial_folds={len(spatial_splits)}, device={args.device}, smoke={args.smoke}", flush=True)
        if authors_splits:
            all_rows.extend(run_splits(key, df, x, authors_splits, "authors_holdout", models, args))
        all_rows.extend(run_splits(key, df, x, random_splits, "random_kfold", models, args))
        all_rows.extend(run_splits(key, df, x, spatial_splits, "spatial_block", models, args))

    fold_df = pd.DataFrame(all_rows)
    summary_df = summarize(all_rows)

    fold_csv = args.output_dir / "fold_results.csv"
    summary_csv = args.output_dir / "target_summary.csv"
    inv_json = args.output_dir / "dataset_inventory.json"
    results_json = args.output_dir / "mississippi_salinity_spatial_results.json"

    fold_df.to_csv(fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    inventory = build_inventory(inventory_targets, args)
    json.dump(inventory, open(inv_json, "w"), indent=2)

    headline = {}
    for target in summary_df["target"].unique() if not summary_df.empty else []:
        xgb_rand = summary_df[(summary_df.target == target) & (summary_df.regime == "random_kfold") & (summary_df.model == "xgboost")]
        xgb_spat = summary_df[(summary_df.target == target) & (summary_df.regime == "spatial_block") & (summary_df.model == "xgboost")]
        idw_rand = summary_df[(summary_df.target == target) & (summary_df.regime == "random_kfold") & (summary_df.model.str.startswith("idw_"))]
        if not xgb_rand.empty and not xgb_spat.empty:
            ml_r = float(xgb_rand.iloc[0]["median_r2"])
            ml_s = float(xgb_spat.iloc[0]["median_r2"])
            item = {
                "random_kfold_xgboost_median_r2": ml_r,
                "spatial_block_xgboost_median_r2": ml_s,
                "gap_random_minus_spatial": ml_r - ml_s,
            }
            if not idw_rand.empty and ml_r > 0:
                idw_r = float(idw_rand.iloc[0]["median_r2"])
                item["random_kfold_idw_median_r2"] = idw_r
                item["idw_recovery_pct_random_regime"] = max(idw_r, 0.0) / max(ml_r, 1e-9) * 100.0
            headline[target] = item

    json.dump({
        "dataset": inventory["dataset"],
        "mode": inventory["mode"],
        "device": args.device,
        "headline": headline,
        "files": {
            "dataset_inventory": relpath(inv_json),
            "fold_results": relpath(fold_csv),
            "target_summary": relpath(summary_csv),
        },
        "manuscript_status": "not promoted; smoke run if mode=smoke",
    }, open(results_json, "w"), indent=2)

    print("\nWrote:")
    print(f"  {inv_json}")
    print(f"  {fold_csv}")
    print(f"  {summary_csv}")
    print(f"  {results_json}")
    if headline:
        print("\nHeadline preview:")
        for target, vals in headline.items():
            print(f"  {target}: random XGB R2={vals['random_kfold_xgboost_median_r2']:.3f}, "
                  f"spatial XGB R2={vals['spatial_block_xgboost_median_r2']:.3f}, "
                  f"gap={vals['gap_random_minus_spatial']:.3f}")


if __name__ == "__main__":
    main()
