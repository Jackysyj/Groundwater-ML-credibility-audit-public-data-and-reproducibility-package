#!/usr/bin/env python3
"""
Mississippi Embayment salinity spatial block-size sensitivity.

This complements the canonical 0.5 degree Mississippi spatial holdout by
coarsening the block grid while keeping the target, predictors, XGBoost model,
IDW baseline, and matched random-fold protocol fixed. The canonical 0.5 degree
fold rows are reused from the validated full GPU output to avoid rerunning the
main headline experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT_DIR = PROC / "mississippi_salinity_blocksize_sensitivity"
CANON_DIR = PROC / "mississippi_salinity_spatial_full_gpu"
CANON_RESULTS = CANON_DIR / "mississippi_salinity_spatial_results.json"
CANON_FOLDS = CANON_DIR / "fold_results.csv"

INCLUDED_DEFAULT = (0.5, 0.75, 1.0, 1.5, 2.0)
SCREENED_DIAGNOSTIC = (0.25,)
TARGET_ORDER = ("specific_conductance", "chloride")


def parse_block_degs(text: str) -> list[float]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def model_rows_from_canonical(block_deg: float) -> pd.DataFrame:
    if abs(block_deg - 0.5) > 1e-9:
        return pd.DataFrame()
    if not CANON_FOLDS.exists():
        return pd.DataFrame()
    df = pd.read_csv(CANON_FOLDS)
    keep = df["regime"].isin(["random_kfold", "spatial_block"]) & (
        df["model"].isin(["xgboost", "idw_k8"]))
    out = df.loc[keep].copy()
    out.insert(0, "block_deg", block_deg)
    out["source"] = "canonical_full_gpu_0p5deg"
    return out


def run_block_deg(target: str, block_deg: float, args: argparse.Namespace) -> pd.DataFrame:
    import killshot_mississippi_salinity_spatial as ks

    if abs(block_deg - 0.5) < 1e-9 and not args.force_recompute_05:
        canon = model_rows_from_canonical(block_deg)
        if not canon.empty:
            return canon[canon["target"].eq(target)].copy()

    spec = ks.TARGETS[target]
    df, features = ks.read_target(spec, smoke=False)
    x, _encoded = ks.encode_features(df, features)
    spatial_splits = ks.make_spatial_splits(
        df, block_deg=block_deg, min_test=args.min_test, max_folds=None)
    if len(spatial_splits) < args.min_folds:
        return pd.DataFrame()
    random_splits = ks.make_random_splits(
        len(df), k=max(2, len(spatial_splits)), min_test=args.min_test, seed=args.seed)
    run_args = SimpleNamespace(
        idw_k=args.idw_k,
        device=args.device,
        nrounds=args.nrounds,
        nthread=args.nthread,
        seed=args.seed,
    )
    rows = []
    rows.extend(ks.run_splits(target, df, x, random_splits, "random_kfold", ["xgboost"], run_args))
    rows.extend(ks.run_splits(target, df, x, spatial_splits, "spatial_block", ["xgboost"], run_args))
    out = pd.DataFrame(rows)
    out.insert(0, "block_deg", block_deg)
    out["source"] = "fresh_blocksize_sensitivity"
    return out


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if fold_df.empty:
        return pd.DataFrame()
    for (target, block_deg, regime, model), g in fold_df.groupby(
            ["target", "block_deg", "regime", "model"], sort=True):
        rows.append({
            "target": target,
            "block_deg": float(block_deg),
            "regime": regime,
            "model": model,
            "n_folds": int(len(g)),
            "pooled_n_test": int(g["n_test"].sum()),
            "median_r2": float(g["r2"].median()),
            "weighted_mean_r2": float((g["r2"] * g["n_test"]).sum() / g["n_test"].sum()),
            "median_rmse": float(g["rmse"].median()),
        })
    summary = pd.DataFrame(rows)

    gap_rows = []
    for (target, block_deg), g in summary[summary["model"].eq("xgboost")].groupby(
            ["target", "block_deg"], sort=True):
        rand = g[g["regime"].eq("random_kfold")]
        spat = g[g["regime"].eq("spatial_block")]
        if rand.empty or spat.empty:
            continue
        gap_rows.append({
            "target": target,
            "block_deg": float(block_deg),
            "xgboost_random_median_r2": float(rand.iloc[0]["median_r2"]),
            "xgboost_spatial_median_r2": float(spat.iloc[0]["median_r2"]),
            "xgboost_gap_random_minus_spatial": float(
                rand.iloc[0]["median_r2"] - spat.iloc[0]["median_r2"]),
            "spatial_n_folds": int(spat.iloc[0]["n_folds"]),
        })
    gaps = pd.DataFrame(gap_rows)
    if not gaps.empty:
        summary = summary.merge(gaps, on=["target", "block_deg"], how="left")
    return summary


def build_screened_diagnostics(block_degs: list[float], min_test: int) -> list[dict]:
    import killshot_mississippi_salinity_spatial as ks

    out = []
    for block_deg in block_degs:
        for target in TARGET_ORDER:
            df, _features = ks.read_target(ks.TARGETS[target], smoke=False)
            splits = ks.make_spatial_splits(df, block_deg=block_deg, min_test=min_test, max_folds=None)
            out.append({
                "target": target,
                "block_deg": float(block_deg),
                "n_spatial_folds": int(len(splits)),
                "status": "screened_out" if len(splits) < 5 else "eligible_not_run",
                "reason": "fewer than five valid spatial folds for at least one headline target"
                if len(splits) < 5 else "diagnostic block size not included in default sweep",
            })
    return out


def write_results(fold_df: pd.DataFrame, summary_df: pd.DataFrame,
                  screened: list[dict], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = OUT_DIR / "fold_results.csv"
    summary_path = OUT_DIR / "blocksize_summary.csv"
    json_path = OUT_DIR / "mississippi_blocksize_sensitivity.json"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    headline = {}
    for target in TARGET_ORDER:
        target_rows = summary_df[
            (summary_df["target"].eq(target)) &
            (summary_df["model"].eq("xgboost")) &
            (summary_df["regime"].eq("spatial_block"))
        ].sort_values("block_deg")
        headline[target] = []
        for _, row in target_rows.iterrows():
            headline[target].append({
                "block_deg": float(row["block_deg"]),
                "spatial_n_folds": int(row["n_folds"]),
                "random_median_r2": float(row["xgboost_random_median_r2"]),
                "spatial_median_r2": float(row["xgboost_spatial_median_r2"]),
                "gap_random_minus_spatial": float(row["xgboost_gap_random_minus_spatial"]),
            })

    json.dump({
        "status": "Mississippi salinity block-size sensitivity; 0.5 degree is the canonical headline block size.",
        "dataset": "USGS Mississippi Embayment salinity, 10.5066/P9WBFR1T",
        "run_metadata": {
            "device_requested": args.device,
            "nrounds": args.nrounds,
            "nthread": args.nthread,
            "seed": args.seed,
            "idw_k": args.idw_k,
            "min_test": args.min_test,
            "included_block_degs": args.block_degs,
            "canonical_0p5_reused": not args.force_recompute_05,
        },
        "screened_out_diagnostics": screened,
        "headline": headline,
        "files": {
            "fold_results": str(fold_path.relative_to(ROOT)),
            "blocksize_summary": str(summary_path.relative_to(ROOT)),
        },
        "writing_summary": (
            "Coarsening the Mississippi spatial blocks is a sensitivity check; "
            "0.5 degree remains the canonical headline geometry."
        ),
    }, open(json_path, "w"), indent=2)
    print(f"Wrote {fold_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")


def check_outputs() -> None:
    fold_path = OUT_DIR / "fold_results.csv"
    summary_path = OUT_DIR / "blocksize_summary.csv"
    json_path = OUT_DIR / "mississippi_blocksize_sensitivity.json"
    for path in (fold_path, summary_path, json_path):
        if not path.exists():
            raise SystemExit(f"CHECK FAIL: missing {path}")
    summary = pd.read_csv(summary_path)
    results = json.load(open(json_path))
    included = [float(x) for x in results["run_metadata"]["included_block_degs"]]
    errors = []
    warnings_out = []
    for target in TARGET_ORDER:
        for block_deg in included:
            g = summary[
                (summary["target"].eq(target)) &
                (summary["block_deg"].round(6).eq(round(block_deg, 6))) &
                (summary["model"].eq("xgboost")) &
                (summary["regime"].eq("spatial_block"))
            ]
            if g.empty:
                errors.append(f"{target} block {block_deg}: missing spatial xgboost row")
                continue
            if int(g.iloc[0]["n_folds"]) < 5:
                errors.append(f"{target} block {block_deg}: fewer than five spatial folds")
            gap = float(g.iloc[0]["xgboost_gap_random_minus_spatial"])
            if pd.isna(gap):
                errors.append(f"{target} block {block_deg}: missing gap")
            elif gap <= 0:
                warnings_out.append(f"{target} block {block_deg}: nonpositive gap {gap:.3f}")

    canon = json.load(open(CANON_RESULTS))
    for target, prefix in [
        ("specific_conductance", "specific_conductance"),
        ("chloride", "chloride"),
    ]:
        row = summary[
            (summary["target"].eq(target)) &
            (summary["block_deg"].round(6).eq(0.5)) &
            (summary["model"].eq("xgboost")) &
            (summary["regime"].eq("spatial_block"))
        ]
        rand = summary[
            (summary["target"].eq(target)) &
            (summary["block_deg"].round(6).eq(0.5)) &
            (summary["model"].eq("xgboost")) &
            (summary["regime"].eq("random_kfold"))
        ]
        if row.empty or rand.empty:
            errors.append(f"{target}: missing 0.5 canonical rows")
            continue
        exp = canon["headline"][prefix]
        observed = {
            "random": float(rand.iloc[0]["median_r2"]),
            "spatial": float(row.iloc[0]["median_r2"]),
            "gap": float(row.iloc[0]["xgboost_gap_random_minus_spatial"]),
        }
        expected = {
            "random": float(exp["random_kfold_xgboost_median_r2"]),
            "spatial": float(exp["spatial_block_xgboost_median_r2"]),
            "gap": float(exp["gap_random_minus_spatial"]),
        }
        for key in expected:
            if abs(observed[key] - expected[key]) > 0.02:
                errors.append(f"{target} 0.5 {key}: observed {observed[key]:.3f}, expected {expected[key]:.3f}")

    if errors:
        raise SystemExit("CHECK FAIL:\n  " + "\n  ".join(errors))
    print("CHECK PASS")
    if warnings_out:
        print("Warnings:")
        for item in warnings_out:
            print(f"  {item}")
    print(f"  targets: {', '.join(TARGET_ORDER)}")
    print(f"  block sizes: {', '.join(str(x) for x in included)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--block-degs", default=",".join(str(x) for x in INCLUDED_DEFAULT))
    ap.add_argument("--min-test", type=int, default=80)
    ap.add_argument("--min-folds", type=int, default=5)
    ap.add_argument("--nrounds", type=int, default=500)
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--idw-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--force-recompute-05", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    args.block_degs = parse_block_degs(args.block_degs)

    if args.check:
        check_outputs()
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    all_rows = []
    screened = build_screened_diagnostics(list(SCREENED_DIAGNOSTIC), args.min_test)
    for block_deg in args.block_degs:
        for target in TARGET_ORDER:
            print(f"\n## {target}, block_deg={block_deg}", flush=True)
            rows = run_block_deg(target, block_deg, args)
            if rows.empty:
                screened.append({
                    "target": target,
                    "block_deg": float(block_deg),
                    "n_spatial_folds": 0,
                    "status": "screened_out",
                    "reason": "fewer than five valid spatial folds",
                })
                print("  screened out: fewer than five valid spatial folds", flush=True)
            else:
                all_rows.append(rows)
    if not all_rows:
        raise SystemExit("No included block-size results were produced.")
    fold_df = pd.concat(all_rows, ignore_index=True)
    summary_df = summarize(fold_df)
    write_results(fold_df, summary_df, screened, args)


if __name__ == "__main__":
    main()
