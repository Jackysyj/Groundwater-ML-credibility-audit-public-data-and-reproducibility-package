#!/usr/bin/env python3
"""Robustness suite for external exogenous-driver groundwater-level probes.

This script is intentionally exploratory: it writes only processed robustness
outputs and does not update manuscript_stats.json. It reuses the French/Swiss
loaders from exogenous_driver_external_probe.py, then expands the comparison
across model families, forecast horizons, feature ablations, lag windows, and
random seeds.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep process-level parallelism predictable. Several sklearn/XGBoost backends
# otherwise spawn extra OpenMP/BLAS threads inside every worker.
for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from exogenous_driver_external_probe import (
    MIN_DAILY_ROWS,
    MIN_DAILY_TEST,
    MIN_MONTHLY_ROWS,
    MIN_MONTHLY_TEST,
    RAW,
    ROOT,
    SeriesSpec,
    floor0,
    load_french,
    load_swiss,
    numeric,
    safe_r2,
)

warnings.filterwarnings("ignore")

DEFAULT_OUT = ROOT / "data" / "processed" / "exogenous_driver_external_probe_robustness"
BASE_SEED = 20260706
DEFAULT_SEEDS = [20260706, 20260707, 20260708, 20260709, 20260710]
DEFAULT_SEEDS_TEXT = ",".join(str(s) for s in DEFAULT_SEEDS)
TEST_FRAC = 0.20

RESULT_KEY = [
    "analysis_block",
    "dataset",
    "well_id",
    "horizon_steps",
    "horizon_label",
    "n_lags",
    "feature_set",
    "model",
    "seed",
]


@dataclass(frozen=True)
class EvalConfig:
    analysis_block: str
    horizon: int
    n_lags: int
    feature_set: str
    model: str
    seed: int


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_supervised(df: pd.DataFrame, gwl_col: str, driver_cols: list[str], horizon: int, n_lags: int) -> pd.DataFrame:
    d = df[["date", gwl_col] + driver_cols].copy()
    d = d.sort_values("date").reset_index(drop=True)
    d[gwl_col] = numeric(d[gwl_col])
    for c in driver_cols:
        d[c] = numeric(d[c])

    out = pd.DataFrame({"date": d["date"], "y": d[gwl_col].shift(-horizon), "persistence": d[gwl_col]})
    for lag in range(n_lags + 1):
        out[f"gwl_lag{lag}"] = d[gwl_col].shift(lag)
        for c in driver_cols:
            out[f"{c}_lag{lag}"] = d[c].shift(lag)

    gwl_lag_cols = [f"gwl_lag{lag}" for lag in range(n_lags + 1)]
    driver_lag_cols = [f"{c}_lag{lag}" for lag in range(n_lags + 1) for c in driver_cols]
    out = out[["date", "y", "persistence"] + gwl_lag_cols + driver_lag_cols]
    out = out.dropna(subset=["y", "persistence"] + gwl_lag_cols)
    return out


def feature_columns(sup: pd.DataFrame, feature_set: str) -> list[str]:
    gwl_cols = [c for c in sup.columns if c.startswith("gwl_lag")]
    driver_cols = [c for c in sup.columns if c not in {"date", "y", "persistence"} and not c.startswith("gwl_lag")]
    if feature_set == "gwl_lags_only":
        return gwl_cols
    if feature_set == "drivers_only":
        return driver_cols
    if feature_set == "gwl_plus_drivers":
        return gwl_cols + driver_cols
    raise ValueError(f"unknown feature_set: {feature_set}")


def make_model(name: str, seed: int, args: argparse.Namespace) -> Any:
    if name == "ridge":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    if name == "elasticnet":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=seed)),
            ]
        )
    if name == "lasso":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=5000, random_state=seed)),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=args.rf_estimators,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if name == "histgbm":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=args.hgb_iter,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=args.xgb_estimators,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=3,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        tree_method="hist",
                        device=args.ml_device,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=0,
                    ),
                ),
            ]
        )
    if name == "mlp":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        early_stopping=True,
                        max_iter=args.mlp_iter,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model: {name}")


def configs_for_profile(args: argparse.Namespace) -> list[EvalConfig]:
    seeds = parse_ints(args.seeds)
    base_seed = seeds[0]
    horizons = parse_ints(args.horizons_days)
    lag_settings = parse_ints(args.lag_settings)
    base_horizons = [h for h in [7, 28, 84] if h in horizons]
    base_lag = 4 if 4 in lag_settings else lag_settings[0]

    all_models = ["persistence", "ridge", "elasticnet", "lasso", "random_forest", "histgbm", "xgboost"]
    if args.include_mlp:
        all_models.append("mlp")

    cfgs: dict[tuple[Any, ...], EvalConfig] = {}

    def add(block: str, horizon: int, n_lags: int, feature_set: str, model: str, seed: int) -> None:
        cfg = EvalConfig(block, horizon, n_lags, feature_set, model, seed)
        key = (block, horizon, n_lags, feature_set, model, seed)
        cfgs[key] = cfg

    # Multi-model panel: all main model families at the standard lag/feature setup.
    for h in base_horizons:
        for model in all_models:
            model_seeds = seeds if model in {"random_forest", "xgboost"} else [base_seed]
            if model == "mlp":
                model_seeds = seeds if args.mlp_multiseed else [base_seed]
            for seed in model_seeds:
                add("multimodel_panel", h, base_lag, "gwl_plus_drivers", model, seed)

    # Horizon sensitivity: persistence/Ridge/XGBoost at the standard lag/feature setup.
    for h in horizons:
        for model in ["persistence", "ridge", "xgboost"]:
            add("horizon_sensitivity", h, base_lag, "gwl_plus_drivers", model, base_seed)

    # Feature ablation: separate local-history and driver-only skill from the combined feature set.
    for h in base_horizons:
        for feature_set in ["gwl_lags_only", "drivers_only", "gwl_plus_drivers"]:
            for model in ["persistence", "ridge", "xgboost"]:
                add("feature_ablation", h, base_lag, feature_set, model, base_seed)

    # Lag sensitivity: test whether conclusions depend on the history-window length.
    for h in base_horizons:
        for n_lags in lag_settings:
            for model in ["persistence", "ridge", "xgboost"]:
                add("lag_sensitivity", h, n_lags, "gwl_plus_drivers", model, base_seed)

    return list(cfgs.values())


def eval_one_config(
    spec: SeriesSpec,
    cfg: EvalConfig,
    args: argparse.Namespace,
    supervised_cache: dict[tuple[int, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]],
) -> dict[str, Any] | None:
    min_rows = MIN_MONTHLY_ROWS if spec.frequency == "monthly" else MIN_DAILY_ROWS
    min_test = MIN_MONTHLY_TEST if spec.frequency == "monthly" else MIN_DAILY_TEST
    d = spec.df.sort_values("date").reset_index(drop=True)
    if args.max_rows_per_well and len(d) > args.max_rows_per_well:
        d = d.iloc[-args.max_rows_per_well :].reset_index(drop=True)

    cache_key = (cfg.horizon, cfg.n_lags)
    if cache_key not in supervised_cache:
        sup = make_supervised(d, spec.gwl_col, spec.driver_cols, cfg.horizon, cfg.n_lags)
        if len(sup) < min_rows:
            return None
        n_test = max(min_test, int(math.ceil(len(sup) * TEST_FRAC)))
        if len(sup) <= n_test + 30:
            return None
        train = sup.iloc[:-n_test].copy()
        test = sup.iloc[-n_test:].copy()
        y_train = train["y"].to_numpy(dtype=float)
        y_test = test["y"].to_numpy(dtype=float)
        if np.nanstd(y_train) == 0 or np.nanstd(y_test) == 0:
            return None
        supervised_cache[cache_key] = (sup, train, test, y_train, y_test)

    sup, train, test, y_train, y_test = supervised_cache[cache_key]
    persistence_pred = test["persistence"].to_numpy(dtype=float)
    persistence_raw = safe_r2(y_test, persistence_pred)
    if cfg.model == "persistence":
        raw = persistence_raw
        r2 = floor0(raw)
    else:
        cols = feature_columns(sup, cfg.feature_set)
        if not cols:
            return None
        model = make_model(cfg.model, cfg.seed, args)
        model.fit(train[cols], y_train)
        pred = model.predict(test[cols])
        raw = safe_r2(y_test, pred)
        r2 = floor0(raw)

    persistence_r2 = floor0(persistence_raw)
    return {
        "analysis_block": cfg.analysis_block,
        "dataset": spec.dataset,
        "well_id": spec.well_id,
        "frequency": spec.frequency,
        "source": spec.source,
        "horizon_steps": cfg.horizon,
        "horizon_label": f"h{cfg.horizon}_{'months' if spec.frequency == 'monthly' else 'days'}",
        "n_lags": cfg.n_lags,
        "feature_set": cfg.feature_set,
        "model": cfg.model,
        "seed": cfg.seed,
        "n_supervised": int(len(sup)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_driver_cols": int(len(spec.driver_cols)),
        "driver_cols": "|".join(spec.driver_cols),
        "r2_raw": raw,
        "r2": r2,
        "persistence_r2_raw": persistence_raw,
        "persistence_r2": persistence_r2,
        "model_minus_persistence": r2 - persistence_r2,
        "model_ge_persistence": bool(r2 >= persistence_r2),
        "gpu_device": args.ml_device if cfg.model == "xgboost" else "cpu_or_not_applicable",
    }


def eval_spec(args_tuple: tuple[SeriesSpec, list[EvalConfig], argparse.Namespace, set[tuple[Any, ...]]]) -> list[dict[str, Any]]:
    spec, cfgs, args, completed = args_tuple
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for cfg in cfgs:
        key = (
            cfg.analysis_block,
            spec.dataset,
            spec.well_id,
            cfg.horizon,
            f"h{cfg.horizon}_{'months' if spec.frequency == 'monthly' else 'days'}",
            cfg.n_lags,
            cfg.feature_set,
            cfg.model,
            cfg.seed,
        )
        if key in completed:
            continue
        row = eval_one_config(spec, cfg, args, cache)
        if row is not None:
            rows.append(row)
    return rows


def summarize_block(per: pd.DataFrame, block: str) -> pd.DataFrame:
    gcols = ["analysis_block", "dataset", "frequency", "horizon_steps", "horizon_label", "n_lags", "feature_set", "model"]
    rows = []
    for keys, g in per[per["analysis_block"] == block].groupby(gcols, dropna=False):
        r2 = g["r2"].replace([np.inf, -np.inf], np.nan)
        pers = g["persistence_r2"].replace([np.inf, -np.inf], np.nan)
        r2_med = float(np.nanmedian(r2)) if r2.notna().any() else float("nan")
        pers_med = float(np.nanmedian(pers)) if pers.notna().any() else float("nan")
        rows.append(
            {
                **dict(zip(gcols, keys, strict=True)),
                "n_rows": int(len(g)),
                "n_wells": int(g[["dataset", "well_id"]].drop_duplicates().shape[0]),
                "n_seeds": int(g["seed"].nunique()),
                "median_r2": r2_med,
                "median_persistence_r2": pers_med,
                "median_model_minus_persistence": float(np.nanmedian(g["model_minus_persistence"])),
                "model_ge_persistence_pct": float(g["model_ge_persistence"].mean() * 100.0),
                "recovery_vs_model_pct": float(100.0 * pers_med / r2_med) if r2_med > 0 else float("nan"),
                "median_n_test": float(np.nanmedian(g["n_test"])),
            }
        )
    return pd.DataFrame(rows).sort_values(gcols) if rows else pd.DataFrame()


def write_outputs(out_dir: Path, inventory: dict[str, Any], per: pd.DataFrame, args: argparse.Namespace, cfgs: list[EvalConfig]) -> None:
    per = per.drop_duplicates(subset=RESULT_KEY, keep="last").sort_values(RESULT_KEY)
    per.to_csv(out_dir / "fold_or_well_model_results.csv", index=False)

    model_summary = summarize_block(per, "multimodel_panel")
    feature_ablation = summarize_block(per, "feature_ablation")
    horizon_lag = pd.concat(
        [summarize_block(per, "horizon_sensitivity"), summarize_block(per, "lag_sensitivity")],
        ignore_index=True,
    )
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)
    feature_ablation.to_csv(out_dir / "feature_ablation_summary.csv", index=False)
    horizon_lag.to_csv(out_dir / "horizon_lag_sensitivity.csv", index=False)

    config_records = [cfg.__dict__ for cfg in cfgs]
    result = {
        "status": "computed_exploratory_not_truth_table",
        "analysis": "external exogenous-driver groundwater-level robustness and multi-model suite",
        "placement_status": "undecided",
        "outputs": {
            "fold_or_well_model_results": "fold_or_well_model_results.csv",
            "model_summary": "model_summary.csv",
            "feature_ablation_summary": "feature_ablation_summary.csv",
            "horizon_lag_sensitivity": "horizon_lag_sensitivity.csv",
        },
        "run_parameters": {
            "datasets": args.datasets,
            "profile": args.profile,
            "max_wells_large": args.max_wells_large,
            "max_rows_per_well": args.max_rows_per_well,
            "workers": args.workers,
            "seeds": parse_ints(args.seeds),
            "horizons_days": parse_ints(args.horizons_days),
            "lag_settings": parse_ints(args.lag_settings),
            "test_frac": TEST_FRAC,
            "ml_device": args.ml_device,
            "xgb_estimators": args.xgb_estimators,
            "rf_estimators": args.rf_estimators,
            "hgb_iter": args.hgb_iter,
            "include_mlp": args.include_mlp,
            "mlp_multiseed": args.mlp_multiseed,
        },
        "modeling_notes": [
            "XGBoost uses tree_method=hist and device=cuda by default.",
            "RandomForest, HistGBM, linear models, and optional MLP are CPU/sklearn models.",
            "Skill is per-well chronological-holdout R2 floored at 0 before summaries.",
            "Feature ablation compares gwl_lags_only, drivers_only, and gwl_plus_drivers under the same split.",
            "This file is not validator-wired and must not be cited as a truth-table source until promoted.",
        ],
        "inventory": inventory,
        "n_configurations": len(config_records),
        "configurations": config_records,
        "summary_preview": {
            "model_summary": model_summary.to_dict(orient="records"),
            "feature_ablation_summary": feature_ablation.to_dict(orient="records"),
            "horizon_lag_sensitivity": horizon_lag.to_dict(orient="records"),
        },
    }
    (out_dir / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False))
    (out_dir / "exogenous_driver_robustness_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


def completed_keys(out_dir: Path) -> tuple[pd.DataFrame, set[tuple[Any, ...]]]:
    path = out_dir / "fold_or_well_model_results.csv"
    if not path.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(path)
    if df.empty:
        return df, set()
    keys = set(tuple(x) for x in df[RESULT_KEY].itertuples(index=False, name=None))
    return df, keys


def validate_outputs(out_dir: Path) -> None:
    required = [
        "dataset_inventory.json",
        "fold_or_well_model_results.csv",
        "model_summary.csv",
        "feature_ablation_summary.csv",
        "horizon_lag_sensitivity.csv",
        "exogenous_driver_robustness_results.json",
    ]
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        raise SystemExit("missing outputs: " + ", ".join(missing))
    per = pd.read_csv(out_dir / "fold_or_well_model_results.csv")
    if per.empty:
        raise SystemExit("empty fold_or_well_model_results.csv")
    dup = per[RESULT_KEY].duplicated()
    if dup.any():
        raise SystemExit(f"duplicate result keys: {int(dup.sum())}")
    blocks = set(per["analysis_block"].unique())
    expected = {"multimodel_panel", "horizon_sensitivity", "feature_ablation", "lag_sensitivity"}
    missing_blocks = expected - blocks
    if missing_blocks:
        raise SystemExit("missing analysis blocks: " + ", ".join(sorted(missing_blocks)))
    if "xgboost" not in set(per["model"]):
        raise SystemExit("missing xgboost rows")
    if "ridge" not in set(per["model"]):
        raise SystemExit("missing ridge rows")
    if "persistence" not in set(per["model"]):
        raise SystemExit("missing persistence rows")
    bad = per[(per["feature_set"] == "drivers_only") & per["model"].ne("persistence") & per["n_driver_cols"].le(0)]
    if not bad.empty:
        raise SystemExit("drivers_only model rows without driver columns")
    print("CHECK PASS: exogenous-driver robustness outputs exist and are internally consistent.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="french,swiss", help="comma-separated dataset keys: french,swiss")
    ap.add_argument("--profile", choices=["smoke", "full"], default="full")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-wells-large", type=int, default=0, help="0 means all eligible wells")
    ap.add_argument("--max-rows-per-well", type=int, default=5000, help="keep most recent N rows per well; 0 means all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--ml-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS_TEXT)
    ap.add_argument("--horizons-days", default="7,14,28,56,84")
    ap.add_argument("--lag-settings", default="1,4,8,14")
    ap.add_argument("--xgb-estimators", type=int, default=180)
    ap.add_argument("--rf-estimators", type=int, default=160)
    ap.add_argument("--hgb-iter", type=int, default=180)
    ap.add_argument("--include-mlp", action="store_true")
    ap.add_argument("--mlp-multiseed", action="store_true")
    ap.add_argument("--mlp-iter", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        validate_outputs(out_dir)
        return

    if args.profile == "smoke" and args.max_wells_large == 0:
        args.max_wells_large = 20
        if args.seeds == DEFAULT_SEEDS_TEXT:
            args.seeds = "20260706,20260707"
        if args.horizons_days == "7,14,28,56,84":
            args.horizons_days = "7,28"
        if args.lag_settings == "1,4,8,14":
            args.lag_settings = "1,4"
        args.xgb_estimators = min(args.xgb_estimators, 50)
        args.rf_estimators = min(args.rf_estimators, 50)
        args.hgb_iter = min(args.hgb_iter, 50)

    requested = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
    inventory: dict[str, Any] = {
        "raw_root": str(RAW.relative_to(ROOT)),
        "profile": args.profile,
    }
    specs: list[SeriesSpec] = []
    if "french" in requested:
        s, inv = load_french(args.max_wells_large)
        specs.extend(s)
        inventory["french_piezoforecast"] = inv
        print(f"FrenchPiezo: {len(s)} selected wells", flush=True)
    if "swiss" in requested:
        s, inv = load_swiss(args.max_wells_large)
        specs.extend(s)
        inventory["swiss_groundwater_database"] = inv
        print(f"Swiss: {len(s)} selected wells", flush=True)

    if not specs:
        raise SystemExit("no requested datasets loaded")

    cfgs = configs_for_profile(args)
    existing, completed = completed_keys(out_dir) if args.resume else (pd.DataFrame(), set())
    all_rows = existing.to_dict(orient="records") if not existing.empty else []
    print(f"configurations: {len(cfgs)}; existing rows: {len(all_rows)}", flush=True)

    tasks = [(spec, cfgs, args, completed) for spec in specs]
    total = len(tasks)
    if args.workers <= 1:
        for i, task in enumerate(tasks, start=1):
            all_rows.extend(eval_spec(task))
            if i == 1 or i % 10 == 0 or i == total:
                print(f"processed {i}/{total} wells; rows={len(all_rows)}", flush=True)
            if args.checkpoint_every and i % args.checkpoint_every == 0:
                write_outputs(out_dir, inventory, pd.DataFrame(all_rows), args, cfgs)
                print(f"checkpoint wrote {len(all_rows)} rows", flush=True)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(eval_spec, task) for task in tasks]
            for i, fut in enumerate(cf.as_completed(futures), start=1):
                all_rows.extend(fut.result())
                if i == 1 or i % 10 == 0 or i == total:
                    print(f"processed {i}/{total} wells; rows={len(all_rows)}", flush=True)
                if args.checkpoint_every and i % args.checkpoint_every == 0:
                    write_outputs(out_dir, inventory, pd.DataFrame(all_rows), args, cfgs)
                    print(f"checkpoint wrote {len(all_rows)} rows", flush=True)

    if not all_rows:
        raise SystemExit("no robustness rows produced")
    write_outputs(out_dir, inventory, pd.DataFrame(all_rows), args, cfgs)
    validate_outputs(out_dir)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
