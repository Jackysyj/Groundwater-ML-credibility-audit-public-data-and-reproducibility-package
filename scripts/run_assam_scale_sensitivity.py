#!/usr/bin/env python3
"""Run the prespecified Assam spatial-block scale sensitivity analysis.

The released result tables compare fixed grid widths of 0.10, 0.15 and 0.20
degrees using the existing predictors, XGBoost configuration, ROC-AUC metric,
random seed and eight-neighbour IDW-frequency baseline.  The source input is
not redistributed.  Supply a separately obtained copy with ``--input`` and
write new results with ``--output``.
"""

from __future__ import annotations

import json
import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA = None
OUT_DIR = PACKAGE_ROOT / "data" / "processed" / "assam_arsenic_scale_sensitivity"

TARGET = "As_class"
LAT, LON = "latitude", "longitude"
FEATURES = [
    "ts_soc", "ts_silt", "ts_sand", "ts_ph", "ts_clay", "ts_cec", "ts_bulkdensity",
    "ss_soc", "ss_silt", "ss_sand", "ss_ph", "ss_clay", "ss_cec", "ss_bulkdensity",
    "fluvisols", "et0_yr", "ai_et0", "elevation", "slope", "precip", "temp",
    "distance_to_river", "lulc", "twi", "ts_cf", "ss_cf",
]

PRESET_SCALES_DEG = (0.10, 0.15, 0.20)
MIN_TEST = 40
MIN_POS = 5
IDW_K = 8
SEED = 42
NROUNDS = 500

# Same model hyperparameters as scripts/killshot_assam_arsenic_spatial.py.
# CPU is used only for this local Apple Silicon rerun.
XGB_PARAMS = dict(
    booster="gbtree", max_depth=5, eta=0.05, subsample=0.7,
    colsample_bytree=0.7, min_child_weight=5,
    objective="binary:logistic", eval_metric="auc",
    tree_method="hist", device="cpu",
)


def load() -> pd.DataFrame:
    if DATA is None:
        raise ValueError("An input table is required; provide --input assam_as_points.csv")
    df = pd.read_csv(DATA)
    df[TARGET] = df[TARGET].astype(int)
    return df[df[LAT].notna() & df[LON].notna()].copy()


def fit_auc(train: pd.DataFrame, test: pd.DataFrame) -> float:
    dtr = xgb.DMatrix(train[FEATURES].to_numpy(), label=train[TARGET].to_numpy(), missing=np.nan)
    dte = xgb.DMatrix(test[FEATURES].to_numpy(), label=test[TARGET].to_numpy(), missing=np.nan)
    model = xgb.train(XGB_PARAMS, dtr, num_boost_round=NROUNDS, verbose_eval=False)
    return float(roc_auc_score(test[TARGET].to_numpy(), model.predict(dte)))


def xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def idw_freq_predict(train: pd.DataFrame, test: pd.DataFrame, k: int = IDW_K) -> np.ndarray:
    tr_xyz = xyz(train[LAT].to_numpy(), train[LON].to_numpy())
    te_xyz = xyz(test[LAT].to_numpy(), test[LON].to_numpy())
    labels = train[TARGET].to_numpy(dtype=float)
    out = np.empty(len(test), dtype=float)
    kk = min(k, len(train) - 1)
    for start in range(0, len(test), 2000):
        chunk = te_xyz[start:start + 2000]
        d2 = ((chunk[:, None, :] - tr_xyz[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, kth=kk, axis=1)[:, :k]
        d = np.take_along_axis(d2, idx, axis=1) ** 0.5
        w = 1.0 / np.clip(d, 1e-6, None)
        out[start:start + len(chunk)] = (labels[idx] * w).sum(1) / w.sum(1)
    return out


def assign_blocks(df: pd.DataFrame, scale: float) -> np.ndarray:
    gx = np.floor(df[LON] / scale).astype(int)
    gy = np.floor(df[LAT] / scale).astype(int)
    return (gx.astype(str) + "_" + gy.astype(str)).to_numpy()


def spatial_splits(df: pd.DataFrame, scale: float):
    d = df.assign(_block=assign_blocks(df, scale))
    sizes = d["_block"].value_counts()
    valid = [b for b in sizes.index
             if sizes[b] >= MIN_TEST and int(d.loc[d["_block"] == b, TARGET].sum()) >= MIN_POS]
    return [(d[d["_block"] != b], d[d["_block"] == b], str(b)) for b in valid], d


def random_splits(df: pd.DataFrame, k: int):
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, k, size=len(df))
    out = []
    for f in range(k):
        te = df[fold == f]
        tr = df[fold != f]
        if len(te) >= MIN_TEST and int(te[TARGET].sum()) >= MIN_POS and te[TARGET].nunique() == 2:
            out.append((tr, te, str(f)))
    return out


def score(splits, scale: float, regime: str, fold_rows: list, split_rows: list) -> dict:
    ml, baseline, weights = [], [], []
    for train, test, tag in splits:
        if test[TARGET].nunique() < 2:
            continue
        ml_auc = fit_auc(train, test)
        base_auc = float(roc_auc_score(test[TARGET].to_numpy(), idw_freq_predict(train, test)))
        ml.append(ml_auc)
        baseline.append(base_auc)
        weights.append(len(test))
        fold_rows.append({
            "scale_deg": scale, "regime": regime, "fold": tag,
            "n_train": int(len(train)), "n_test": int(len(test)),
            "n_positive_test": int(test[TARGET].sum()),
            "ml_auc": ml_auc, "baseline_idw_auc": base_auc,
            "ml_skill": ml_auc - 0.5, "baseline_skill": base_auc - 0.5,
            "model_increment_auc": ml_auc - base_auc,
            "baseline_recovery_pct": (100.0 * max(base_auc - 0.5, 0.0) / (ml_auc - 0.5) if ml_auc > 0.5 else float("nan")),
        })
    if not ml:
        return {"n_folds": 0}
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    ml_med, base_med = float(np.median(ml)), float(np.median(baseline))
    ml_skill, base_skill = ml_med - 0.5, base_med - 0.5
    split_rows.append({
        "scale_deg": scale, "regime": regime, "n_folds": len(ml),
        "n_test_total": int(sum(weights)), "ml_median_auc": ml_med,
        "baseline_median_auc": base_med,
        "ml_mean_auc_weighted": float(np.average(ml, weights=w)),
        "baseline_mean_auc_weighted": float(np.average(baseline, weights=w)),
        "model_increment_median_auc": ml_med - base_med,
        "ml_skill_median": ml_skill, "baseline_skill_median": base_skill,
        "baseline_recovery_pct": 100.0 * max(base_skill, 0.0) / max(ml_skill, 1e-12),
        "baseline_meets_or_beats_model": bool(base_med >= ml_med),
    })
    return split_rows[-1]


def bootstrap_median_ci(values: list[float], seed: int = 20260922, n_boot: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap interval over held-out folds, retained as uncertainty."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True)
    med = np.median(samples, axis=1)
    return float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))


def main() -> None:
    global DATA, OUT_DIR
    parser=argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUT_DIR)
    args=parser.parse_args()
    DATA, OUT_DIR=args.input,args.output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    fold_rows, split_rows, scale_rows = [], [], []
    for scale in PRESET_SCALES_DEG:
        spatial, assigned = spatial_splits(df, scale)
        # Matched random k is kept per scale so the sensitivity changes only
        # the spatial partition, while the random reference remains comparable.
        k = max(len(spatial), 5)
        random = random_splits(df, k)
        spatial_result = score(spatial, scale, "spatial_block", fold_rows, split_rows)
        random_result = score(random, scale, "random_kfold", fold_rows, split_rows)
        scale_rows.append({
            "scale_deg": scale,
            "n_blocks_total": int(assigned["_block"].nunique()),
            "n_spatial_folds": int(spatial_result.get("n_folds", 0)),
            "n_random_folds": int(random_result.get("n_folds", 0)),
            "spatial_ml_median_auc": spatial_result.get("ml_median_auc"),
            "spatial_baseline_median_auc": spatial_result.get("baseline_median_auc"),
            "random_ml_median_auc": random_result.get("ml_median_auc"),
            "random_baseline_median_auc": random_result.get("baseline_median_auc"),
            "spatial_model_increment_median_auc": spatial_result.get("model_increment_median_auc"),
            "spatial_baseline_recovery_pct": spatial_result.get("baseline_recovery_pct"),
            "random_model_increment_median_auc": random_result.get("model_increment_median_auc"),
            "random_baseline_recovery_pct": random_result.get("baseline_recovery_pct"),
            "random_minus_spatial_ml_auc": (
                random_result.get("ml_median_auc") - spatial_result.get("ml_median_auc")
                if random_result.get("ml_median_auc") is not None and spatial_result.get("ml_median_auc") is not None else None
            ),
        })

    # Add fold-level percentile intervals to the split table and the compact
    # scale table. They describe fold-to-fold uncertainty, not a population
    # confidence interval and not uncertainty in the historical WSL run.
    split_lookup = {(r["scale_deg"], r["regime"]): r for r in split_rows}
    for key, row in split_lookup.items():
        vals = [r for r in fold_rows if (r["scale_deg"], r["regime"]) == key]
        for field, out_name in [
            ("ml_auc", "ml_median_auc_ci95"),
            ("baseline_idw_auc", "baseline_median_auc_ci95"),
        ]:
            row[out_name] = list(bootstrap_median_ci([v[field] for v in vals]))
        paired = np.asarray([[v['ml_auc'], v['baseline_idw_auc']] for v in vals])
        rng = np.random.default_rng(20260922)
        ids = rng.integers(0, len(paired), size=(2000, len(paired)))
        diff = np.median(paired[ids, 0], axis=1)-np.median(paired[ids, 1], axis=1)
        row['model_increment_median_auc_ci95'] = np.quantile(diff, [0.025, 0.975]).tolist()
        row['median_paired_fold_increment_auc'] = float(np.median(paired[:,0]-paired[:,1]))
    for row in scale_rows:
        for regime, prefix in [("spatial_block", "spatial"), ("random_kfold", "random")]:
            sr = split_lookup[(row["scale_deg"], regime)]
            row[f"{prefix}_ml_median_auc_ci95"] = sr["ml_median_auc_ci95"]
            row[f"{prefix}_baseline_median_auc_ci95"] = sr["baseline_median_auc_ci95"]
            row[f"{prefix}_model_increment_median_auc_ci95"] = sr["model_increment_median_auc_ci95"]

    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_detail.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT_DIR / "split_summary.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(OUT_DIR / "scale_summary.csv", index=False)
    (OUT_DIR / "preset_scales.json").write_text(json.dumps({
        "scales_deg": list(PRESET_SCALES_DEG),
        "main_scale_deg": 0.15,
        "scale_unit": "longitude/latitude grid degrees used by the original script",
        "selection_rule": "fixed before inspecting sensitivity results; all three scales reported",
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "run_metadata.json").write_text(json.dumps({
        "data": "input supplied with --input (not redistributed)",
        "data_sha256": __import__("hashlib").sha256(DATA.read_bytes()).hexdigest(),
        "script": "scripts/run_assam_scale_sensitivity.py",
        "python": sys.version,
        "platform": platform.platform(),
        "xgboost_version": xgb.__version__,
        "historical_model_device": "cuda in original WSL script",
        "rerun_model_device": "cpu",
        "xgb_params": XGB_PARAMS,
        "num_boost_round": NROUNDS,
        "seed": SEED,
        "baseline": "8-nearest-neighbour inverse-distance-weighted frequency; no fitted baseline model",
        "note": "The released scale sweep is a supporting sensitivity analysis. The historical WSL run used CUDA; a CPU rerun is not an exact cross-environment reproduction.",
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text("""# Assam spatial scale sensitivity\n\nThis analysis compares three fixed grid scales, 0.10°, 0.15° and 0.20°. The 0.15° grid is the main analysis scale. All scales and evaluable folds are reported using the existing 26 predictors, preprocessing, XGBoost settings, ROC-AUC metric, random seed and eight-neighbour inverse-distance-weighted frequency baseline.\n\nEach scale has a spatial-block result and a matched random-k-fold reference. `scale_summary.csv` reports model AUC, baseline AUC, model increment and baseline recovery; `fold_detail.csv` contains fold-level values.\n\nThe historical script used XGBoost histogram training with CUDA in WSL. The recorded rerun used the same settings with CPU execution on Apple Silicon. The environment record identifies this difference, so the results are treated as a sensitivity analysis rather than an exact cross-environment reproduction.\n\nGrid width is not the minimum training-test distance and does not estimate a spatial autocorrelation range. No variogram range or buffer distance was estimated. The results describe scale sensitivity within this task and do not establish that all spatial dependence has been removed.\n""", encoding="utf-8")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
