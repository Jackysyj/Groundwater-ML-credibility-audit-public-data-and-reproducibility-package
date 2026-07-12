#!/usr/bin/env python3
"""
ROLLING-ORIGIN (chronological) temporal holdout for the water-level thesis,
with a MULTI-HORIZON forecast curve.

Motivation (reviewer #1 of the 4th review): the persistence result so far uses a
RANDOM test fold. A reviewer will ask whether REAL future forecasting behaves the
same. So here we do a deployment-style split: train on the EARLY period, test on
the LATER period, and on that same held-out future window we report THREE lines
at several forecast horizons h (steps ahead):
  1. ML                    - RandomForest on N lags (same model family as the
                             random-split water-level experiment, so numbers are
                             comparable)
  2. persistence           - forecast = last OBSERVED value (z[i]) for every h
  3. seasonal persistence  - forecast = same calendar month one year ago

CRITICAL FAIRNESS FIX (2026-07-05):
  The previous version mis-aligned features and target by one step: features
  ended at z[i-1] while persistence used z[i] to predict z[i+1]. That let the
  persistence baseline see one month that the ML model could not, so persistence
  won almost by construction. Fixed here: for a fixed information cutoff at index
  i (last OBSERVED month = z[i]):
    - features  = {z[i-N_LAGS+1], ..., z[i]}   (most recent observed = z[i])
    - target    = z[i+h]                       (h steps ahead)
    - persistence = z[i]                        (same cutoff as the ML features)
    - seasonal    = value at (target calendar month - 1 year)
  ML and persistence now share the SAME information cutoff at every horizon, so
  the comparison is fair. h=1 is the honest one-step forecast; the h curve shows
  at what lead time (if any) ML starts to beat naive persistence.

Honesty guards against leakage:
  - chronological cutoff is a GLOBAL year_month percentile (default 80th): every
    supervised row whose TARGET month is after the cutoff is test, else train.
  - per-well z-score uses TRAIN-period mean/std ONLY (fit on train, applied to
    test), so no future statistics leak into the features or the target scale.
  - one ML model is trained per horizon on the pooled train rows for that horizon.

Datasets: GEMS (panel_long.csv), NCP (panel_long.csv), GROW (per-country parquet,
pooled). Output: data/processed/rolling_origin_temporal.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
OUT = PROC / "rolling_origin_temporal.json"

N_LAGS = 3
CUTOFF_Q = 0.80          # train = first 80% of the timeline, test = last 20%
HORIZONS = [1, 3, 6, 12]  # steps ahead (index steps on the per-well series)
RF = dict(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1)
MIN_TRAIN_PER_WELL = N_LAGS + 2


def _load_gems():
    d = pd.read_csv(PROC / "gems_wells" / "panel_long.csv",
                    usecols=["well_id", "year_month", "depth_m"])
    return d.rename(columns={"depth_m": "value"})


def _load_ncp():
    d = pd.read_csv(PROC / "ncp_wells" / "panel_long.csv",
                    usecols=["well_id", "year_month", "depth_m"])
    return d.rename(columns={"depth_m": "value"})


def _load_grow():
    parts = []
    for f in sorted((PROC / "grow" / "panels").glob("*.parquet")):
        parts.append(pd.read_parquet(f, columns=["well_id", "year_month", "value"]))
    return pd.concat(parts, ignore_index=True)


def build_supervised(df, cutoff, horizon):
    """Per-well z-scored lag-N rows for a given forecast horizon.

    For information cutoff index i (last observed = z[i]):
      features    = {z[i-N_LAGS+1] .. z[i]}   (most recent observed value is z[i])
      y           = z[i+horizon]
      persistence = z[i]                       (same cutoff as ML features)
      seasonal    = value at target calendar month one year earlier
    z-score fit on TRAIN months only. Returns train/test supervised frames.
    """
    df = df.dropna(subset=["value"]).copy()
    df["year_month"] = df["year_month"].astype(int)
    tr_rows, te_rows = [], []
    for wid, g in df.groupby("well_id"):
        g = g.sort_values("year_month").reset_index(drop=True)
        ym = g["year_month"].values
        val = g["value"].values.astype(float)
        train_mask = ym <= cutoff
        if train_mask.sum() < MIN_TRAIN_PER_WELL:
            continue
        mu = np.nanmean(val[train_mask])
        sigma = np.nanstd(val[train_mask])
        if sigma == 0 or np.isnan(sigma):
            continue
        z = (val - mu) / sigma
        zmap = dict(zip(ym, z))          # for seasonal lookup (same month last year)
        # i = information cutoff index; features end AT i (include z[i]).
        for i in range(N_LAGS - 1, len(g) - horizon):
            lags = z[i - N_LAGS + 1:i + 1]           # N_LAGS values, most recent = z[i]
            target_ym = ym[i + horizon]
            if np.isnan(lags).any() or np.isnan(z[i + horizon]) or np.isnan(z[i]):
                continue
            seasonal = zmap.get(int(target_ym) - 100, np.nan)  # YYYYMM - 100 = prev year same month
            row = {"well_id": wid,
                   # lag_z_1 = most recent observed (z[i]), lag_z_2 = z[i-1], ...
                   **{f"lag_z_{j+1}": lags[-(j + 1)] for j in range(N_LAGS)},
                   "y": z[i + horizon], "y_persistence": z[i], "y_seasonal": seasonal}
            (te_rows if target_ym > cutoff else tr_rows).append(row)
    return pd.DataFrame(tr_rows), pd.DataFrame(te_rows)


def eval_horizon(df, cutoff, horizon):
    tr, te = build_supervised(df, cutoff, horizon)
    if len(tr) < 50 or len(te) < 50:
        return {"horizon": horizon, "error": f"too few rows tr={len(tr)} te={len(te)}"}
    feat = [c for c in tr.columns if c.startswith("lag_z_")]
    model = RandomForestRegressor(**RF)
    model.fit(tr[feat], tr["y"])
    pred = model.predict(te[feat])
    y = te["y"].values

    ml_r2 = float(r2_score(y, pred))
    pers_r2 = float(r2_score(y, te["y_persistence"].values))
    seas_mask = te["y_seasonal"].notna().values
    seas_r2 = (float(r2_score(y[seas_mask], te["y_seasonal"].values[seas_mask]))
               if seas_mask.sum() > 10 else None)

    # per-well medians (robustness): predict ONCE on the whole test frame, then
    # group by well (calling model.predict per well is the bottleneck at 1M+ rows).
    te = te.copy()
    te["_pred"] = pred
    per_well = []
    for wid, gg in te.groupby("well_id"):
        if len(gg) < 5:
            continue
        rec = {"ml": r2_score(gg["y"], gg["_pred"]),
               "pers": r2_score(gg["y"], gg["y_persistence"])}
        sm = gg["y_seasonal"].notna()
        if sm.sum() >= 3:
            rec["seas"] = r2_score(gg["y"][sm], gg["y_seasonal"][sm])
        per_well.append(rec)
    pw = pd.DataFrame(per_well)

    return {
        "horizon": horizon,
        "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
        "n_test_wells": int(te["well_id"].nunique()),
        "pooled": {"ml_r2": round(ml_r2, 4),
                   "persistence_r2": round(pers_r2, 4),
                   "seasonal_persistence_r2": round(seas_r2, 4) if seas_r2 is not None else None},
        "per_well_median": {
            "ml_r2": round(float(pw["ml"].median()), 4) if len(pw) else None,
            "persistence_r2": round(float(pw["pers"].median()), 4) if len(pw) else None,
            "seasonal_persistence_r2": round(float(pw["seas"].median()), 4) if "seas" in pw and pw["seas"].notna().any() else None,
            "n_wells": int(len(pw))},
        "persistence_beats_ml_pooled": bool(pers_r2 >= ml_r2),
        "persistence_beats_ml_well_pct": round(float((pw["pers"] >= pw["ml"]).mean() * 100), 1) if len(pw) else None,
        # raw count for the manuscript's n/N denominator rule (pct alone is not
        # citable as a count-based share); denominator = per_well_median.n_wells
        "persistence_beats_ml_well_n": int((pw["pers"] >= pw["ml"]).sum()) if len(pw) else None,
    }


def evaluate(name, df):
    df["year_month"] = df["year_month"].astype(int)
    cutoff = int(np.quantile(df["year_month"].unique(), CUTOFF_Q))
    horizons = {}
    for h in HORIZONS:
        r = eval_horizon(df, cutoff, h)
        horizons[f"h{h}"] = r
        p = r.get("pooled", {})
        pw = r.get("per_well_median", {})
        print(f"  [{name}] h={h} test_rows={r.get('n_test_rows')} "
              f"ML={p.get('ml_r2')} persist={p.get('persistence_r2')} "
              f"seasonal={p.get('seasonal_persistence_r2')} | "
              f"per-well ML={pw.get('ml_r2')} pers={pw.get('persistence_r2')} "
              f"pers_win%={r.get('persistence_beats_ml_well_pct')}", flush=True)
    return {"cutoff_year_month": cutoff, "horizons": horizons}


def main():
    res = {"design": ("Chronological rolling-origin with a multi-horizon forecast "
                      f"curve: train on the first {int(CUTOFF_Q*100)}% of the "
                      f"timeline, test on the last {int((1-CUTOFF_Q)*100)}%. z-score "
                      "fit on train months only (no future leakage). For information "
                      "cutoff index i (last observed = z[i]): features = the N most "
                      "recent observed values ending AT z[i]; target = z[i+h]; "
                      "persistence = z[i] (SAME cutoff as ML, fair at every h); "
                      "seasonal = same calendar month one year earlier. Reports ML "
                      "(RandomForest, 3 lags), persistence, and seasonal persistence "
                      f"at horizons h={HORIZONS} steps ahead."),
           "fairness_note": ("Fixes a one-step feature/target mis-alignment in the "
                             "prior version where persistence saw one month the ML "
                             "features did not; ML and persistence now share the same "
                             "information cutoff at every horizon."),
           "model": "RandomForestRegressor(n_estimators=200, max_depth=6)",
           "n_lags": N_LAGS, "cutoff_quantile": CUTOFF_Q, "horizons": HORIZONS}
    for name, loader in [("gems_de", _load_gems), ("ncp_cn", _load_ncp), ("grow_16", _load_grow)]:
        print(f"[{name}] loading + rolling-origin (multi-horizon) ...", flush=True)
        res[name] = evaluate(name, loader())
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
