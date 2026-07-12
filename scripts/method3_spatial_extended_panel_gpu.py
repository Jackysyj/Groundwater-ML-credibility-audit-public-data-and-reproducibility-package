#!/usr/bin/env python3
"""
Extended spatial five-model panel for the external water-quality mapping axis.

The original Method-3 panel covers nitrate, US arsenic, and global fluoride in
scripts/method3_full_panel_gpu.py. This script keeps that output unchanged and
adds the two newer spatial reinforcements:

  - Assam arsenic exceedance classification
  - Mississippi Embayment specific conductance and chloride regression

It reuses the validated dataset modules for data loading, features, and fold
definitions. The combined output is a six-target panel across five external
spatial data sources, with five model families per target.

Output:
  data/processed/method3_spatial_extended_panel_gpu.json
  data/processed/method3_spatial_extended_panel_gpu.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
OUT_JSON = P / "method3_spatial_extended_panel_gpu.json"
OUT_CSV = P / "method3_spatial_extended_panel_gpu.csv"
OLD_PANEL_JSON = P / "method3_full_panel_gpu.json"

SEED = 42
MODEL_KEYS = ("linear", "random_forest", "grad_boosting", "hist_gbdt", "neural_net")
MODEL_LABELS = {
    "linear": "linear",
    "random_forest": "random_forest_xgbrf_gpu",
    "grad_boosting": "xgboost_gpu",
    "hist_gbdt": "hist_gbdt",
    "neural_net": "mlp_torch_gpu",
}
AUTHOR_ORIGINAL_FAMILY = {
    "nitrate": "grad_boosting",
    "arsenic": "random_forest",
    "fluoride": "random_forest",
    "assam_arsenic": "random_forest",
    "mississippi_specific_conductance": "grad_boosting",
    "mississippi_chloride": "grad_boosting",
}


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def _device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return requested


def _impute_std(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], standardize: bool):
    xtr = train[feats].astype(float).values
    xte = test[feats].astype(float).values
    med = np.nanmedian(xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    xtr = np.where(np.isnan(xtr), med, xtr)
    xte = np.where(np.isnan(xte), med, xte)
    if standardize:
        mu = xtr.mean(axis=0)
        sd = xtr.std(axis=0) + 1e-8
        xtr = (xtr - mu) / sd
        xte = (xte - mu) / sd
    return xtr, xte


def _mlp_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feats: list[str],
    target: str,
    task: str,
    device: str,
    epochs: int,
):
    torch.manual_seed(SEED)
    xtr, xte = _impute_std(train, test, feats, standardize=True)
    ytr = train[target].values.astype(float)
    y_mu = 0.0
    y_sd = 1.0
    y_train_loss = ytr
    if task == "reg":
        y_mu = float(np.mean(ytr))
        y_sd = float(np.std(ytr) + 1e-8)
        y_train_loss = (ytr - y_mu) / y_sd
    xtr_t = torch.tensor(xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_train_loss, dtype=torch.float32, device=device).view(-1, 1)
    xte_t = torch.tensor(xte, dtype=torch.float32, device=device)
    model = MLP(xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if task == "clf":
        pos = float(ytr.sum())
        neg = float(len(ytr) - pos)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device)
        )
    else:
        loss_fn = nn.MSELoss()
    n = len(xtr_t)
    bs = min(4096, max(64, n))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xte_t).cpu().numpy().ravel()
    if task == "clf":
        pred = 1.0 / (1.0 + np.exp(-pred))
    else:
        pred = pred * y_sd + y_mu
    return pred


def _predict(
    model_key: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feats: list[str],
    target: str,
    task: str,
    device: str,
    nrounds: int,
    epochs: int,
):
    if task == "clf" and (train[target].nunique() < 2 or test[target].nunique() < 2):
        return None
    if model_key == "linear":
        xtr, xte = _impute_std(train, test, feats, standardize=True)
        ytr = train[target].values
        if task == "clf":
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(xtr, ytr)
            return model.predict_proba(xte)[:, 1]
        model = Ridge(alpha=1.0)
        model.fit(xtr, ytr)
        return model.predict(xte)
    if model_key in ("random_forest", "grad_boosting"):
        xtr, xte = _impute_std(train, test, feats, standardize=False)
        ytr = train[target].values
        common = {"tree_method": "hist", "device": device, "n_jobs": -1, "random_state": SEED}
        if model_key == "random_forest":
            cls = xgb.XGBRFClassifier if task == "clf" else xgb.XGBRFRegressor
            model = cls(
                n_estimators=max(100, min(nrounds, 300)),
                subsample=0.7,
                colsample_bynode=0.7,
                **common,
            )
        else:
            cls = xgb.XGBClassifier if task == "clf" else xgb.XGBRegressor
            model = cls(
                n_estimators=max(100, nrounds),
                max_depth=5,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=5,
                **common,
            )
        if task == "clf":
            pos = float((ytr == 1).sum())
            neg = float((ytr == 0).sum())
            model.set_params(scale_pos_weight=neg / max(pos, 1.0), eval_metric="auc")
        model.fit(xtr, ytr)
        return model.predict_proba(xte)[:, 1] if task == "clf" else model.predict(xte)
    if model_key == "hist_gbdt":
        xtr, xte = _impute_std(train, test, feats, standardize=False)
        ytr = train[target].values
        if task == "clf":
            model = HistGradientBoostingClassifier(
                max_iter=max(100, min(nrounds, 300)),
                learning_rate=0.05,
                class_weight="balanced",
                random_state=SEED,
            )
            model.fit(xtr, ytr)
            return model.predict_proba(xte)[:, 1]
        model = HistGradientBoostingRegressor(
            max_iter=max(100, min(nrounds, 300)),
            learning_rate=0.05,
            random_state=SEED,
        )
        model.fit(xtr, ytr)
        return model.predict(xte)
    if model_key == "neural_net":
        return _mlp_predict(train, test, feats, target, task, device, epochs)
    raise ValueError(model_key)


def _score(y_true: np.ndarray, pred: np.ndarray | None, task: str):
    if pred is None:
        return None
    if task == "clf":
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, pred))
    return float(r2_score(y_true, pred))


def _median(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(vals)) if vals else None


def _to_split_frames(df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray, str]]):
    return [(df.iloc[tr].copy(), df.iloc[te].copy(), tag) for tr, te, tag in splits]


def folds_assam(smoke_folds: int | None):
    import killshot_assam_arsenic_spatial as ks

    df = ks.load().reset_index(drop=True)
    _, spatial = ks.spatial_block_splits(df, ks.BLOCK_DEG_MAIN)
    random = ks.random_kfold_splits(df, k=max(len(spatial), 5))
    if smoke_folds:
        spatial = spatial[:smoke_folds]
        random = random[:smoke_folds]
    return {
        "dataset": "assam_arsenic",
        "source": "HydroShare Assam Brahmaputra arsenic",
        "task": "clf",
        "metric": "AUC",
        "target": ks.TARGET,
        "features": ks.FEATURES,
        "random": [(tr.copy(), te.copy(), tag) for tr, te, tag in random],
        "spatial": [(tr.copy(), te.copy(), tag) for tr, te, tag in spatial],
    }


def folds_mississippi(target_key: str, smoke_folds: int | None, smoke_rows: int | None):
    import killshot_mississippi_salinity_spatial as ks

    spec = ks.TARGETS[target_key]
    smoke = smoke_rows is not None
    df, features = ks.read_target(spec, smoke=smoke, smoke_n=smoke_rows or 1200)
    x, encoded = ks.encode_features(df, features)
    panel = pd.concat(
        [df[[ks.LAT, ks.LON, "_target"]].reset_index(drop=True), x.reset_index(drop=True)],
        axis=1,
    )
    panel = panel.rename(columns={"_target": "__target"})
    spatial = ks.make_spatial_splits(panel, block_deg=0.5, min_test=80 if not smoke else 50,
                                     max_folds=smoke_folds)
    random = ks.make_random_splits(len(panel), k=max(len(spatial), 2), min_test=80 if not smoke else 50,
                                   seed=ks.SEED)
    if smoke_folds:
        random = random[:smoke_folds]
    return {
        "dataset": f"mississippi_{target_key}",
        "source": "USGS Mississippi Embayment salinity",
        "task": "reg",
        "metric": "R2",
        "target": "__target",
        "features": encoded,
        "random": _to_split_frames(panel, random),
        "spatial": _to_split_frames(panel, spatial),
    }


def run_dataset(spec: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(
        f"\n=== {spec['dataset']} ({spec['metric']}) "
        f"{len(spec['random'])} random / {len(spec['spatial'])} spatial folds ===",
        flush=True,
    )
    result = {
        "task": spec["task"],
        "metric": spec["metric"],
        "source": spec["source"],
        "target": spec["target"],
        "n_features": len(spec["features"]),
        "n_random_folds": len(spec["random"]),
        "n_spatial_folds": len(spec["spatial"]),
        "models": {},
    }
    rows = []
    for model_key in MODEL_KEYS:
        random_scores = []
        spatial_scores = []
        for train, test, _tag in spec["random"]:
            pred = _predict(
                model_key, train, test, spec["features"], spec["target"], spec["task"],
                args.device, args.nrounds, args.epochs,
            )
            random_scores.append(_score(test[spec["target"]].values, pred, spec["task"]))
        for train, test, _tag in spec["spatial"]:
            pred = _predict(
                model_key, train, test, spec["features"], spec["target"], spec["task"],
                args.device, args.nrounds, args.epochs,
            )
            spatial_scores.append(_score(test[spec["target"]].values, pred, spec["task"]))
        random_median = _median(random_scores)
        spatial_median = _median(spatial_scores)
        gap = (
            random_median - spatial_median
            if random_median is not None and spatial_median is not None
            else None
        )
        label = MODEL_LABELS[model_key]
        item = {
            "family": model_key,
            "random_median": random_median,
            "spatial_median": spatial_median,
            "gap": gap,
            "n_random": len([v for v in random_scores if v is not None]),
            "n_spatial": len([v for v in spatial_scores if v is not None]),
            "authors_original_family": AUTHOR_ORIGINAL_FAMILY.get(spec["dataset"]) == model_key,
            "warning_nonpositive_gap": gap is not None and gap <= 0,
        }
        result["models"][label] = item
        rows.append({
            "dataset": spec["dataset"],
            "source": spec["source"],
            "task": spec["task"],
            "metric": spec["metric"],
            "model": label,
            "family": model_key,
            "random_median": random_median,
            "spatial_median": spatial_median,
            "gap": gap,
            "n_random": item["n_random"],
            "n_spatial": item["n_spatial"],
            "authors_original_family": item["authors_original_family"],
            "warning_nonpositive_gap": item["warning_nonpositive_gap"],
        })
        print(
            f"  {model_key:14s} random={random_median if random_median is None else round(random_median, 3)} "
            f"spatial={spatial_median if spatial_median is None else round(spatial_median, 3)} "
            f"gap={gap if gap is None else round(gap, 3)}",
            flush=True,
        )
    return result, rows


def _old_panel_rows(old: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset, spec in old.items():
        for model, item in spec["models"].items():
            rows.append({
                "dataset": dataset,
                "source": "original_method3_full_panel",
                "task": spec["task"],
                "metric": spec["metric"],
                "model": model,
                "family": item["family"],
                "random_median": item["random_median"],
                "spatial_median": item["spatial_median"],
                "gap": item["gap"],
                "n_random": item["n_random"],
                "n_spatial": item["n_spatial"],
                "authors_original_family": AUTHOR_ORIGINAL_FAMILY.get(dataset) == item["family"],
                "warning_nonpositive_gap": item["gap"] <= 0 if item["gap"] is not None else True,
            })
    return rows


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [r["gap"] for r in rows if r["gap"] is not None and not math.isnan(float(r["gap"]))]
    datasets = sorted({r["dataset"] for r in rows})
    return {
        "n_target_tasks": len(datasets),
        "n_model_target_combinations": len(rows),
        "all_gaps_positive": bool(gaps) and all(g > 0 for g in gaps),
        "nonpositive_gap_rows": [
            {k: r[k] for k in ("dataset", "model", "gap")}
            for r in rows
            if r["gap"] is None or r["gap"] <= 0
        ],
        "gap_range": [float(min(gaps)), float(max(gaps))] if gaps else [None, None],
        "datasets": datasets,
    }


def write_outputs(new_results: dict[str, Any], new_rows: list[dict[str, Any]], args: argparse.Namespace):
    if not OLD_PANEL_JSON.exists():
        raise SystemExit(f"Missing original panel output: {OLD_PANEL_JSON}")
    old = json.load(open(OLD_PANEL_JSON))
    combined = dict(old)
    combined.update(new_results)
    rows = _old_panel_rows(old) + new_rows
    summary = _summary_from_rows(rows)
    payload = {
        "status": "extended five-model panel over five external spatial data sources and six target tasks",
        "generated_by": "scripts/method3_spatial_extended_panel_gpu.py",
        "device": args.device,
        "nrounds": args.nrounds,
        "epochs": args.epochs,
        "original_three_dataset_panel": str(OLD_PANEL_JSON.relative_to(ROOT)),
        "new_targets": sorted(new_results),
        "summary": summary,
        "results": combined,
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"All gaps positive: {summary['all_gaps_positive']}; gap range: {summary['gap_range']}")


def run(args: argparse.Namespace):
    specs = [
        folds_assam(args.smoke_folds),
        folds_mississippi("specific_conductance", args.smoke_folds, args.smoke_rows),
        folds_mississippi("chloride", args.smoke_folds, args.smoke_rows),
    ]
    results = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        result, ds_rows = run_dataset(spec, args)
        results[spec["dataset"]] = result
        rows.extend(ds_rows)
    write_outputs(results, rows, args)


def check() -> int:
    if not OUT_JSON.exists() or not OUT_CSV.exists():
        print(f"Missing output(s): {OUT_JSON.relative_to(ROOT)}, {OUT_CSV.relative_to(ROOT)}")
        return 1
    payload = json.load(open(OUT_JSON))
    rows = pd.read_csv(OUT_CSV)
    expected = {
        "assam_arsenic",
        "mississippi_specific_conductance",
        "mississippi_chloride",
    }
    missing = expected - set(payload.get("new_targets", []))
    errors = []
    if missing:
        errors.append(f"missing new targets: {sorted(missing)}")
    for target in expected:
        sub = rows[rows["dataset"] == target]
        if len(sub) != 5:
            errors.append(f"{target}: expected 5 model rows, found {len(sub)}")
        if sub["random_median"].isna().any() or sub["spatial_median"].isna().any():
            errors.append(f"{target}: null random/spatial median")
        if sub["gap"].isna().any():
            errors.append(f"{target}: null gap")
    summary = payload.get("summary", {})
    if summary.get("n_target_tasks") != 6:
        errors.append(f"combined panel expected 6 target tasks, found {summary.get('n_target_tasks')}")
    if summary.get("n_model_target_combinations") != 30:
        errors.append(
            "combined panel expected 30 model-target rows, "
            f"found {summary.get('n_model_target_combinations')}"
        )
    if errors:
        print("CHECK FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("CHECK PASS")
    print(f"  new targets: {', '.join(sorted(expected))}")
    print(f"  combined target tasks: {summary.get('n_target_tasks')}")
    print(f"  model-target combinations: {summary.get('n_model_target_combinations')}")
    print(f"  all gaps positive: {summary.get('all_gaps_positive')}")
    print(f"  gap range: {summary.get('gap_range')}")
    if summary.get("nonpositive_gap_rows"):
        print("  warning nonpositive gaps:")
        for row in summary["nonpositive_gap_rows"]:
            print(f"    {row}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--nrounds", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-folds", type=int, default=None)
    ap.add_argument("--smoke-rows", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check())
    args.device = _device(args.device)
    if args.smoke:
        args.nrounds = min(args.nrounds, 80)
        args.epochs = min(args.epochs, 20)
        args.smoke_folds = args.smoke_folds or 2
        args.smoke_rows = args.smoke_rows or 1200
    print(f"device: {args.device}", flush=True)
    if args.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    run(args)


if __name__ == "__main__":
    main()
