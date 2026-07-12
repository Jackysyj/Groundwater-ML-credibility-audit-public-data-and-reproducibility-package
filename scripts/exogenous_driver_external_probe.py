#!/usr/bin/env python3
"""External exogenous-driver groundwater-level probe.

This script extends the earlier CAMELS-GB exogenous-driver pilot to multiple
corpus-external datasets. It is intentionally separated from
`manuscript_stats.json`: the outputs are exploratory until the user decides
whether this branch belongs in the Free-Baseline Recovery family.

Inputs under data/raw/exogenous_driver_external/:
  - FrenchPiezo: daily GWL + precipitation/evaporation
  - Swiss Groundwater Database: daily heads + MeteoSwiss drivers in a zip
  - Tuscany: monthly per-well CSV files in a RAR archive
  - CAMELS-GB: previously computed per-well JSON, summarized for comparison

Outputs:
  data/processed/exogenous_driver_external_probe/dataset_inventory.json
  data/processed/exogenous_driver_external_probe/per_well_results.csv
  data/processed/exogenous_driver_external_probe/summary.csv
  data/processed/exogenous_driver_external_probe/exogenous_driver_external_probe_results.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import math
import subprocess
import sys
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "exogenous_driver_external"
DEFAULT_OUT = ROOT / "data" / "processed" / "exogenous_driver_external_probe"

SEED = 20260706
TEST_FRAC = 0.20
N_LAGS = 4
DAILY_HORIZONS = [7, 28, 84]
MONTHLY_HORIZONS = [1, 4, 12]
MIN_DAILY_ROWS = 400
MIN_MONTHLY_ROWS = 80
MIN_DAILY_TEST = 60
MIN_MONTHLY_TEST = 12


@dataclass
class SeriesSpec:
    dataset: str
    well_id: str
    frequency: str
    df: pd.DataFrame
    gwl_col: str
    driver_cols: list[str]
    source: str


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def floor0(x: float) -> float:
    if x is None or not np.isfinite(x):
        return float("nan")
    return max(0.0, float(x))


def safe_r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.nanstd(y) == 0:
        return float("nan")
    try:
        return float(r2_score(y, pred))
    except Exception:
        return float("nan")


def make_supervised(df: pd.DataFrame, gwl_col: str, driver_cols: list[str], horizon: int) -> pd.DataFrame:
    d = df[["date", gwl_col] + driver_cols].copy()
    d = d.sort_values("date").reset_index(drop=True)
    d[gwl_col] = numeric(d[gwl_col])
    for c in driver_cols:
        d[c] = numeric(d[c])

    out = pd.DataFrame({"date": d["date"], "y": d[gwl_col].shift(-horizon), "persistence": d[gwl_col]})
    for lag in range(N_LAGS + 1):
        out[f"gwl_lag{lag}"] = d[gwl_col].shift(lag)
        for c in driver_cols:
            out[f"{c}_lag{lag}"] = d[c].shift(lag)

    gwl_lag_cols = [f"gwl_lag{lag}" for lag in range(N_LAGS + 1)]
    keep_cols = ["date", "y", "persistence"] + gwl_lag_cols + [
        f"{c}_lag{lag}" for lag in range(N_LAGS + 1) for c in driver_cols
    ]
    out = out[keep_cols]
    out = out.dropna(subset=["y", "persistence"] + gwl_lag_cols)
    return out


def make_ml_model(seed: int, device: str) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def fit_score_one(spec: SeriesSpec, horizon: int, max_rows_per_well: int, ml_device: str) -> dict[str, Any] | None:
    min_rows = MIN_MONTHLY_ROWS if spec.frequency == "monthly" else MIN_DAILY_ROWS
    min_test = MIN_MONTHLY_TEST if spec.frequency == "monthly" else MIN_DAILY_TEST

    d = spec.df.sort_values("date").reset_index(drop=True)
    if max_rows_per_well and len(d) > max_rows_per_well:
        d = d.iloc[-max_rows_per_well:].reset_index(drop=True)
    sup = make_supervised(d, spec.gwl_col, spec.driver_cols, horizon)
    if len(sup) < min_rows:
        return None

    n_test = max(min_test, int(math.ceil(len(sup) * TEST_FRAC)))
    if len(sup) <= n_test + 30:
        return None
    train = sup.iloc[:-n_test].copy()
    test = sup.iloc[-n_test:].copy()
    feat_cols = [c for c in sup.columns if c not in {"date", "y", "persistence"}]
    y_train = train["y"].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)
    if np.nanstd(y_train) == 0 or np.nanstd(y_test) == 0:
        return None

    x_train = train[feat_cols]
    x_test = test[feat_cols]

    persist_pred = test["persistence"].to_numpy(dtype=float)
    persist_r2_raw = safe_r2(y_test, persist_pred)

    ridge = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    ridge.fit(x_train, y_train)
    ridge_pred = ridge.predict(x_test)
    ridge_r2_raw = safe_r2(y_test, ridge_pred)

    ml = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", make_ml_model(SEED, ml_device))])
    ml.fit(x_train, y_train)
    ml_pred = ml.predict(x_test)
    ml_r2_raw = safe_r2(y_test, ml_pred)

    persist_r2 = floor0(persist_r2_raw)
    ridge_r2 = floor0(ridge_r2_raw)
    ml_r2 = floor0(ml_r2_raw)
    return {
        "dataset": spec.dataset,
        "well_id": spec.well_id,
        "frequency": spec.frequency,
        "source": spec.source,
        "horizon_steps": horizon,
        "horizon_label": f"h{horizon}_{'months' if spec.frequency == 'monthly' else 'days'}",
        "n_supervised": int(len(sup)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_driver_cols": int(len(spec.driver_cols)),
        "driver_cols": "|".join(spec.driver_cols),
        "persistence_r2_raw": persist_r2_raw,
        "ridge_r2_raw": ridge_r2_raw,
        "ml_r2_raw": ml_r2_raw,
        "ml_model": f"xgboost_hist_{ml_device}",
        "persistence_r2": persist_r2,
        "ridge_r2": ridge_r2,
        "ml_r2": ml_r2,
        "ml_minus_persistence": ml_r2 - persist_r2,
        "ml_minus_ridge": ml_r2 - ridge_r2,
        "persistence_ge_ml": bool(persist_r2 >= ml_r2),
        "ridge_ge_ml": bool(ridge_r2 >= ml_r2),
    }


def fit_spec_all_horizons(args_tuple: tuple[SeriesSpec, list[int], int, str]) -> list[dict[str, Any]]:
    spec, horizons, max_rows_per_well, ml_device = args_tuple
    rows: list[dict[str, Any]] = []
    for h in horizons:
        r = fit_score_one(spec, h, max_rows_per_well, ml_device)
        if r is not None:
            rows.append(r)
    return rows


def select_ids(ids: list[str], max_wells: int) -> list[str]:
    ids = sorted(ids)
    if max_wells <= 0 or len(ids) <= max_wells:
        return ids
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(ids, size=max_wells, replace=False).tolist())


def load_french(max_wells: int) -> tuple[list[SeriesSpec], dict[str, Any]]:
    path = RAW / "french_piezoforecast" / "dataset_2015_2021_nomissing_linear.csv"
    usecols = ["bss", "time", "tp", "e", "p"]
    df = pd.read_csv(path, usecols=usecols)
    df["date"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["date", "p"])
    counts = df.groupby("bss")["p"].count()
    eligible = counts[counts >= MIN_DAILY_ROWS].index.astype(str).tolist()
    selected = select_ids(eligible, max_wells)
    specs = []
    for well_id, g in df[df["bss"].isin(selected)].groupby("bss"):
        specs.append(
            SeriesSpec(
                dataset="french_piezoforecast",
                well_id=str(well_id),
                frequency="daily",
                df=g[["date", "p", "tp", "e"]].copy(),
                gwl_col="p",
                driver_cols=["tp", "e"],
                source=str(path.relative_to(ROOT)),
            )
        )
    inv = {
        "source_file": str(path.relative_to(ROOT)),
        "raw_rows": int(len(df)),
        "eligible_wells": int(len(eligible)),
        "selected_wells": int(len(specs)),
        "selection": "seeded subset of eligible wells" if max_wells and len(eligible) > max_wells else "all eligible wells",
        "frequency": "daily",
        "drivers": ["tp", "e"],
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
    }
    return specs, inv


def read_zip_csv_member(zip_path: Path, member: str, usecols: list[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            return pd.read_csv(fh, usecols=usecols)


def load_swiss(max_wells: int) -> tuple[list[SeriesSpec], dict[str, Any]]:
    zip_path = RAW / "swiss_groundwater_database" / "0_data.zip"
    heads = read_zip_csv_member(zip_path, "0_data/final/heads.csv")
    heads = heads.rename(columns={"Unnamed: 0": "time"})
    heads["date"] = pd.to_datetime(heads["time"], errors="coerce")
    heads = heads[(heads["date"] >= "1971-01-01") & (heads["date"] <= "2023-08-31")].copy()
    station_cols = [c for c in heads.columns if c not in {"time", "date"}]
    counts = heads[station_cols].notna().sum()
    eligible = counts[counts >= MIN_DAILY_ROWS].sort_values(ascending=False).index.astype(str).tolist()
    selected = select_ids(eligible, max_wells)

    usecols = ["time"] + selected
    precip = read_zip_csv_member(zip_path, "0_data/final/precipitation.csv", usecols=usecols).rename(columns={"time": "date"})
    temp = read_zip_csv_member(zip_path, "0_data/final/temperature_mean.csv", usecols=usecols).rename(columns={"time": "date"})
    evap = read_zip_csv_member(zip_path, "0_data/final/evaporation.csv", usecols=usecols).rename(columns={"time": "date"})
    for d in (precip, temp, evap):
        d["date"] = pd.to_datetime(d["date"], errors="coerce")

    heads_sel = heads[["date"] + selected]
    specs = []
    for sid in selected:
        d = pd.DataFrame({"date": heads_sel["date"], "gwl": heads_sel[sid]})
        d = d.merge(precip[["date", sid]].rename(columns={sid: "precipitation"}), on="date", how="left")
        d = d.merge(temp[["date", sid]].rename(columns={sid: "temperature_mean"}), on="date", how="left")
        d = d.merge(evap[["date", sid]].rename(columns={sid: "evaporation"}), on="date", how="left")
        if d["gwl"].notna().sum() >= MIN_DAILY_ROWS:
            specs.append(
                SeriesSpec(
                    dataset="swiss_groundwater_database",
                    well_id=sid,
                    frequency="daily",
                    df=d,
                    gwl_col="gwl",
                    driver_cols=["precipitation", "temperature_mean", "evaporation"],
                    source=str(zip_path.relative_to(ROOT)) + "::0_data/final/*.csv",
                )
            )
    inv = {
        "source_file": str(zip_path.relative_to(ROOT)),
        "raw_station_count": int(len(station_cols)),
        "eligible_wells": int(len(eligible)),
        "selected_wells": int(len(specs)),
        "selection": "seeded subset of eligible wells" if max_wells and len(eligible) > max_wells else "all eligible wells",
        "frequency": "daily",
        "drivers": ["precipitation", "temperature_mean", "evaporation"],
        "date_min": str(heads["date"].min().date()),
        "date_max": str(heads["date"].max().date()),
    }
    return specs, inv


def bsdtar_list(archive: Path) -> list[str]:
    proc = subprocess.run(["bsdtar", "-tf", str(archive)], check=True, capture_output=True, text=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def bsdtar_read_csv(archive: Path, member: str) -> pd.DataFrame:
    proc = subprocess.run(["bsdtar", "-xOf", str(archive), member], check=True, capture_output=True)
    return pd.read_csv(io.BytesIO(proc.stdout))


def load_tuscany(max_wells: int) -> tuple[list[SeriesSpec], dict[str, Any]]:
    archive = RAW / "Tuscany.rar"
    members = [m for m in bsdtar_list(archive) if m.lower().endswith(".csv")]
    selected = select_ids(members, max_wells)
    specs = []
    n_read = 0
    n_eligible = 0
    for member in selected:
        try:
            d = bsdtar_read_csv(archive, member)
        except Exception:
            continue
        n_read += 1
        if "Livello [m]" not in d.columns:
            continue
        d["date"] = pd.to_datetime(
            d["Year"].astype(str) + "-" + d["Month"].astype(str).str.zfill(2) + "-15",
            errors="coerce",
        )
        driver_cols = [c for c in ["pr", "tp", "td", "ae", "rf", "ws"] if c in d.columns]
        use = d[["date", "Livello [m]"] + driver_cols].copy()
        if use["Livello [m]"].notna().sum() < MIN_MONTHLY_ROWS:
            continue
        n_eligible += 1
        specs.append(
            SeriesSpec(
                dataset="tuscany_groundwater_drought",
                well_id=str(d["Codice"].iloc[0]) if "Codice" in d.columns and len(d) else Path(member).stem,
                frequency="monthly",
                df=use,
                gwl_col="Livello [m]",
                driver_cols=driver_cols,
                source=str(archive.relative_to(ROOT)) + f"::{member}",
            )
        )
    inv = {
        "source_file": str(archive.relative_to(ROOT)),
        "raw_csv_members": int(len(members)),
        "read_members": int(n_read),
        "eligible_wells": int(n_eligible),
        "selected_wells": int(len(specs)),
        "selection": "seeded subset of CSV members" if max_wells and len(members) > max_wells else "all CSV members",
        "frequency": "monthly",
        "drivers": ["pr", "tp", "td", "ae", "rf", "ws"],
    }
    return specs, inv


def summarize_camels_existing() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = RAW / "exogenous_probe_camelsgb_results.json"
    if not path.exists():
        return [], {"status": "missing", "source_file": str(path.relative_to(ROOT))}
    data = json.loads(path.read_text())
    rows = []
    for freq, horizons in [("monthly", [1, 4, 12]), ("daily", [7, 28, 84])]:
        for rec in data.get(freq, []):
            well = str(rec.get("well", ""))
            for h in horizons:
                key = f"h{h}"
                r = rec.get(key)
                if not r:
                    continue
                persist = floor0(float(r.get("persist", float("nan"))))
                ridge = floor0(float(r.get("ridge", float("nan"))))
                ml = floor0(float(r.get("xgb", float("nan"))))
                rows.append(
                    {
                        "dataset": "camels_gb_external_existing",
                        "well_id": well,
                        "frequency": freq,
                        "source": str(path.relative_to(ROOT)),
                        "horizon_steps": h,
                        "horizon_label": f"h{h}_{'months' if freq == 'monthly' else 'days'}",
                        "n_supervised": np.nan,
                        "n_train": np.nan,
                        "n_test": int(r.get("n_test", 0) or 0),
                        "n_driver_cols": np.nan,
                        "driver_cols": "gwl_lags|borrowed_catchment_hydromet",
                        "persistence_r2_raw": float(r.get("persist", float("nan"))),
                        "ridge_r2_raw": float(r.get("ridge", float("nan"))),
                        "ml_r2_raw": float(r.get("xgb", float("nan"))),
                        "persistence_r2": persist,
                        "ridge_r2": ridge,
                        "ml_r2": ml,
                        "ml_minus_persistence": ml - persist,
                        "ml_minus_ridge": ml - ridge,
                        "persistence_ge_ml": bool(persist >= ml),
                        "ridge_ge_ml": bool(ridge >= ml),
                    }
                )
    inv = {
        "source_file": str(path.relative_to(ROOT)),
        "status": "precomputed_existing_result_summarized_not_refit",
        "n_rows_loaded": int(len(rows)),
        "frequencies": sorted({r["frequency"] for r in rows}),
    }
    return rows, inv


def summarize(per_well: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "frequency", "horizon_steps", "horizon_label"]
    for keys, g in per_well.groupby(group_cols, dropna=False):
        dataset, frequency, horizon, label = keys
        ml = g["ml_r2"].replace([np.inf, -np.inf], np.nan)
        pers = g["persistence_r2"].replace([np.inf, -np.inf], np.nan)
        ridge = g["ridge_r2"].replace([np.inf, -np.inf], np.nan)
        ml_med = float(np.nanmedian(ml)) if ml.notna().any() else float("nan")
        pers_med = float(np.nanmedian(pers)) if pers.notna().any() else float("nan")
        ridge_med = float(np.nanmedian(ridge)) if ridge.notna().any() else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "frequency": frequency,
                "horizon_steps": int(horizon),
                "horizon_label": label,
                "n_wells": int(len(g)),
                "median_persistence_r2": pers_med,
                "median_ridge_r2": ridge_med,
                "median_ml_r2": ml_med,
                "diff_median_ml_minus_persistence": ml_med - pers_med,
                "diff_median_ml_minus_ridge": ml_med - ridge_med,
                "median_pairwise_ml_minus_persistence": float(np.nanmedian(g["ml_minus_persistence"])),
                "median_pairwise_ml_minus_ridge": float(np.nanmedian(g["ml_minus_ridge"])),
                "persistence_recovery_pct": float(100.0 * pers_med / ml_med) if ml_med > 0 else float("nan"),
                "ridge_recovery_pct": float(100.0 * ridge_med / ml_med) if ml_med > 0 else float("nan"),
                "persistence_ge_ml_pct": float(g["persistence_ge_ml"].mean() * 100.0),
                "ridge_ge_ml_pct": float(g["ridge_ge_ml"].mean() * 100.0),
                "median_n_test": float(np.nanmedian(g["n_test"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "frequency", "horizon_steps"])


def write_outputs(out_dir: Path, inventory: dict[str, Any], all_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    per = pd.DataFrame(all_rows)
    if per.empty:
        return
    summ = summarize(per)
    per.to_csv(out_dir / "per_well_results.csv", index=False)
    summ.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False))
    res = result_json(inventory, summ, args)
    (out_dir / "exogenous_driver_external_probe_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


def result_json(inventory: dict[str, Any], summary_df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "computed_exploratory_not_truth_table",
        "analysis": "external exogenous-driver groundwater-level cheap/free-baseline recovery probe",
        "placement_status": "undecided",
        "modeling": {
            "test_frac": TEST_FRAC,
            "n_lags": N_LAGS,
            "daily_horizons_days": DAILY_HORIZONS,
            "monthly_horizons_months": MONTHLY_HORIZONS,
            "cheap_baselines": ["persistence: next = current groundwater level", "Ridge on same lagged GWL + drivers"],
            "flexible_ml": f"XGBoost hist on same features, device={args.ml_device} (GPU-first; CPU fallback available)",
            "skill": "test R2 floored at 0 before cross-well medians",
            "split": "per-well chronological holdout; no shuffled rows",
            "max_wells_per_large_dataset": args.max_wells_large,
            "max_rows_per_well": args.max_rows_per_well,
            "workers": args.workers,
            "checkpoint_every": args.checkpoint_every,
            "seed": SEED,
        },
        "inventory": inventory,
        "summary": summary_df.to_dict(orient="records"),
        "caveats": [
            "This run is not written to manuscript_stats.json and is not validator-protected yet.",
            "French and Swiss are deterministically capped by seeded well selection in the first run; uncap for final overnight run if the branch is promoted.",
            "CAMELS-GB rows summarize an existing precomputed run instead of refitting in this script.",
            "GWDBrazil is not included because the downloaded archive contains water-level series and static climate summaries, but no synchronized dynamic meteorological drivers.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="french,swiss,tuscany,camels", help="comma-separated dataset keys")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT), help="directory for output tables and JSON")
    ap.add_argument("--max-wells-large", type=int, default=200, help="cap French/Swiss wells for first-run speed; 0 means all")
    ap.add_argument("--max-wells-tuscany", type=int, default=0, help="cap Tuscany wells; 0 means all")
    ap.add_argument("--max-rows-per-well", type=int, default=5000, help="keep most recent N rows per well; 0 means all")
    ap.add_argument("--ml-device", choices=["cuda", "cpu"], default="cuda", help="XGBoost device for the flexible ML model")
    ap.add_argument("--workers", type=int, default=1, help="parallel well-level workers; use 4-6 for GPU saturation")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="write partial outputs after this many fitted wells")
    ap.add_argument("--validate-only", action="store_true", help="check outputs exist and are internally consistent")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        required = [
            out_dir / "dataset_inventory.json",
            out_dir / "per_well_results.csv",
            out_dir / "summary.csv",
            out_dir / "exogenous_driver_external_probe_results.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise SystemExit("missing outputs: " + ", ".join(missing))
        per = pd.read_csv(out_dir / "per_well_results.csv")
        summ = pd.read_csv(out_dir / "summary.csv")
        if per.empty or summ.empty:
            raise SystemExit("empty output table")
        if per[["dataset", "well_id", "horizon_label"]].duplicated().any():
            raise SystemExit("duplicate dataset/well/horizon rows")
        print("CHECK PASS: exogenous-driver external probe outputs exist and are internally consistent.")
        return

    requested = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
    inventory: dict[str, Any] = {}
    specs: list[SeriesSpec] = []
    precomputed_rows: list[dict[str, Any]] = []

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
    if "tuscany" in requested:
        s, inv = load_tuscany(args.max_wells_tuscany)
        specs.extend(s)
        inventory["tuscany_groundwater_drought"] = inv
        print(f"Tuscany: {len(s)} selected wells", flush=True)
    if "camels" in requested:
        rows, inv = summarize_camels_existing()
        precomputed_rows.extend(rows)
        inventory["camels_gb_external_existing"] = inv
        print(f"CAMELS-GB existing rows: {len(rows)}", flush=True)

    all_rows: list[dict[str, Any]] = list(precomputed_rows)
    total = len(specs)
    tasks = [
        (spec, MONTHLY_HORIZONS if spec.frequency == "monthly" else DAILY_HORIZONS, args.max_rows_per_well, args.ml_device)
        for spec in specs
    ]

    if args.workers <= 1:
        for i, task in enumerate(tasks, start=1):
            all_rows.extend(fit_spec_all_horizons(task))
            if i == 1 or i % 25 == 0 or i == total:
                print(f"processed {i}/{total} fitted wells", flush=True)
            if args.checkpoint_every and i % args.checkpoint_every == 0:
                write_outputs(out_dir, inventory, all_rows, args)
                print(f"checkpoint wrote {len(all_rows)} rows", flush=True)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(fit_spec_all_horizons, task) for task in tasks]
            for i, fut in enumerate(cf.as_completed(futures), start=1):
                all_rows.extend(fut.result())
                if i == 1 or i % 25 == 0 or i == total:
                    print(f"processed {i}/{total} fitted wells", flush=True)
                if args.checkpoint_every and i % args.checkpoint_every == 0:
                    write_outputs(out_dir, inventory, all_rows, args)
                    print(f"checkpoint wrote {len(all_rows)} rows", flush=True)

    if not all_rows:
        raise SystemExit("no valid per-well rows produced")
    write_outputs(out_dir, inventory, all_rows, args)
    summ = pd.read_csv(out_dir / "summary.csv")

    print(f"\nwrote {out_dir}")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
