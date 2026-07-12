#!/usr/bin/env python3
"""
SPATIAL MECHANISM MAP — why the water-quality (spatial) leakage happens.

This is a DESCRIPTIVE mechanism illustration, NOT a predictive model. It does not
try to predict the leakage gap (that "universal density->leakage curve" was
falsified on GROW, leave-one-country-out LOCO R2 negative; see
manuscript_stats.json:grow_killshot_universal_curve_FALSIFIED). Here we only SHOW,
within the standard random-split regime where the inflated skill lives, that ML
skill and the zero-learning neighbour baseline skill BOTH decay as a test point's
nearest-training-well distance grows. That is the mechanism: high random-split
skill is spatial-neighbour interpolation, so it evaporates exactly where local
neighbours are sparse.

Procedure (nitrate regression + fluoride classification):
  1. one random 80/20 split (fixed seed, same SEED as the shared fold-builder).
  2. fit the authors' model on train; also the zero-learning neighbour baseline
     (nitrate IDW-8; fluoride IDW-freq-16).
  3. for each TEST point, compute great-circle distance to its nearest TRAINING
     well (km) and local training density (# train wells within 50 km).
  4. bin test points by nearest-training-distance; per bin report ML skill and
     baseline skill (nitrate R2; fluoride AUC) and the ML-minus-baseline increment.
  5. report Spearman(distance, ML skill) and Spearman(distance, increment)
     DESCRIPTIVELY. No fitted curve, no extrapolation, no correction factor.

Output: data/processed/spatial_mechanism_map.json
        data/processed/spatial_mechanism_map_bins.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
OUT_JSON = PROC / "spatial_mechanism_map.json"
OUT_BINS = PROC / "spatial_mechanism_map_bins.csv"
SEED = 1182
EARTH_KM = 6371.0
DENS_RADIUS_KM = 50.0
TARGET_KEYS = (
    "nitrate",
    "fluoride",
    "assam_arsenic",
    "mississippi_specific_conductance",
    "mississippi_chloride",
)


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo),
                            np.cos(la) * np.sin(lo),
                            np.sin(la)])


def nearest_dist_and_density(tr_lat, tr_lon, te_lat, te_lon, radius_km=DENS_RADIUS_KM):
    """Great-circle nearest-training distance (km) and local training density
    (# train wells within radius_km) for each test point. Chord->arc conversion."""
    trx = _xyz(tr_lat, tr_lon)
    tex = _xyz(te_lat, te_lon)
    nd = np.empty(len(tex))
    dens = np.empty(len(tex), dtype=int)
    chord_thresh = 2 * np.sin(radius_km / (2 * EARTH_KM))    # chord length for radius_km arc
    step = 2000
    for i in range(0, len(tex), step):
        ch = tex[i:i + step]
        d2 = ((ch[:, None, :] - trx[None, :, :]) ** 2).sum(axis=2)   # squared chord
        chord = np.sqrt(np.clip(d2, 0, 4))
        nn_chord = chord.min(axis=1)
        arc = 2 * np.arcsin(np.clip(nn_chord / 2, 0, 1)) * EARTH_KM  # chord -> great-circle km
        nd[i:i + step] = arc
        dens[i:i + step] = (chord <= chord_thresh).sum(axis=1)
    return nd, dens


def _bin_and_report(dist, ml_skill_fn, base_skill_fn, y, ml_pred, base_pred, metric):
    """Bin test points by nearest-training-distance; per bin compute ML/baseline
    skill. skill funcs take (y_bin, pred_bin)."""
    # distance bins by quantile (robust to skew), >= this many points per bin
    qs = np.quantile(dist, [0, .2, .4, .6, .8, 1.0])
    qs = np.unique(qs)
    rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (dist >= lo) & (dist <= hi) if hi == qs[-1] else (dist >= lo) & (dist < hi)
        if m.sum() < 30:
            continue
        yb = y[m]
        if metric == "AUC" and len(np.unique(yb)) < 2:
            continue
        ms = ml_skill_fn(yb, ml_pred[m])
        bs = base_skill_fn(yb, base_pred[m])
        rows.append({"dist_lo_km": round(float(lo), 1), "dist_hi_km": round(float(hi), 1),
                     "median_dist_km": round(float(np.median(dist[m])), 1),
                     "n": int(m.sum()),
                     "ml_skill": round(float(ms), 4), "baseline_skill": round(float(bs), 4),
                     "increment": round(float(ms - bs), 4)})
    return rows


def _common_error_summary(dist, dens, y, ml_pred, base_pred):
    ml_err = np.abs(y - ml_pred)
    base_err = np.abs(y - base_pred)
    return {
        "spearman_dist_vs_ml_abserr": round(float(spearmanr(dist, ml_err).correlation), 3),
        "spearman_dist_vs_base_abserr": round(float(spearmanr(dist, base_err).correlation), 3),
        "spearman_density_vs_ml_abserr": round(float(spearmanr(dens, ml_err).correlation), 3),
        "spearman_density_vs_base_abserr": round(float(spearmanr(dens, base_err).correlation), 3),
    }


def _common_probability_error_summary(dist, dens, y, ml_prob, base_prob):
    ml_err = np.abs(ml_prob - y)
    base_err = np.abs(base_prob - y)
    return {
        "spearman_dist_vs_ml_proberr": round(float(spearmanr(dist, ml_err).correlation), 3),
        "spearman_dist_vs_base_proberr": round(float(spearmanr(dist, base_err).correlation), 3),
        "spearman_density_vs_ml_proberr": round(float(spearmanr(dens, ml_err).correlation), 3),
        "spearman_density_vs_base_proberr": round(float(spearmanr(dens, base_err).correlation), 3),
    }


def run_nitrate(device):
    import killshot_usgs_nitrate_spatial as ks
    import depth_nitrate_sensitivity_and_spatial_baseline as nb
    import xgboost as xgb
    ks.XGB_PARAMS = {**ks.XGB_PARAMS, "tree_method": "hist", "device": device}
    df, feats = ks.load_data()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    # ML per-point predictions — mirror fit_eval EXACTLY (.values, missing=nan,
    # full XGB_PARAMS + NROUNDS) so the skill matches the validated kill-shot.
    dtr = xgb.DMatrix(tr[feats].values, label=tr[ks.TARGET].values, missing=np.nan)
    dte = xgb.DMatrix(te[feats].values, label=te[ks.TARGET].values, missing=np.nan)
    model = xgb.train(ks.XGB_PARAMS, dtr, num_boost_round=ks.NROUNDS, verbose_eval=False)
    ml_pred = model.predict(dte)
    base_pred = nb.nn_predict(tr, te, 8)
    y = te[ks.TARGET].values
    dist, dens = nearest_dist_and_density(tr[ks.LAT].values, tr[ks.LON].values,
                                          te[ks.LAT].values, te[ks.LON].values)
    r2fn = lambda yy, pp: r2_score(yy, pp)
    bins = _bin_and_report(dist, r2fn, r2fn, y, ml_pred, base_pred, "R2")
    out = {
        "metric": "R2 (regression)", "baseline": "IDW-8",
        "n_test": int(len(te)),
        "bins": bins,
    }
    out.update(_common_error_summary(dist, dens, y, ml_pred, base_pred))
    return out


def run_fluoride(device):
    import killshot_global_fluoride_spatial as ks
    import depth_fluoride_sensitivity_and_spatial_baseline as fb
    import xgboost as xgb
    ks.XGB_PARAMS = {**ks.XGB_PARAMS, "tree_method": "hist", "device": device}
    df = ks.load()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    # ML per-point probability — mirror fit_auc EXACTLY (FEATURES, .values, missing=nan).
    FEAT = ks.FEATURES
    dtr = xgb.DMatrix(tr[FEAT].values, label=tr[ks.TARGET].values, missing=np.nan)
    dte = xgb.DMatrix(te[FEAT].values, label=te[ks.TARGET].values, missing=np.nan)
    model = xgb.train(ks.XGB_PARAMS, dtr, num_boost_round=ks.NROUNDS, verbose_eval=False)
    ml_prob = model.predict(dte)
    base_prob = fb.nn_freq(tr, te, 16)
    y = te[ks.TARGET].values
    dist, dens = nearest_dist_and_density(tr["lat"].values, tr["lon"].values,
                                          te["lat"].values, te["lon"].values)
    aucfn = lambda yy, pp: roc_auc_score(yy, pp)
    bins = _bin_and_report(dist, aucfn, aucfn, y, ml_prob, base_prob, "AUC")
    out = {
        "metric": "AUC (classification)", "baseline": "IDW-freq-16",
        "n_test": int(len(te)),
        "bins": bins,
    }
    out.update(_common_probability_error_summary(dist, dens, y, ml_prob, base_prob))
    return out


def run_assam_arsenic(device):
    import killshot_assam_arsenic_spatial as ks
    import xgboost as xgb

    ks.XGB_PARAMS = {**ks.XGB_PARAMS, "tree_method": "hist", "device": device}
    df = ks.load()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]

    dtr = xgb.DMatrix(tr[ks.FEATURES].values, label=tr[ks.TARGET].values, missing=np.nan)
    dte = xgb.DMatrix(te[ks.FEATURES].values, label=te[ks.TARGET].values, missing=np.nan)
    model = xgb.train(ks.XGB_PARAMS, dtr, num_boost_round=ks.NROUNDS, verbose_eval=False)
    ml_prob = model.predict(dte)
    base_prob = ks.idw_freq_predict(tr, te, k=ks.IDW_K)
    y = te[ks.TARGET].values
    dist, dens = nearest_dist_and_density(
        tr[ks.LAT].values, tr[ks.LON].values, te[ks.LAT].values, te[ks.LON].values)
    aucfn = lambda yy, pp: roc_auc_score(yy, pp)
    bins = _bin_and_report(dist, aucfn, aucfn, y, ml_prob, base_prob, "AUC")
    out = {
        "metric": "AUC (classification)", "baseline": f"IDW-freq-{ks.IDW_K}",
        "n_test": int(len(te)),
        "bins": bins,
    }
    out.update(_common_probability_error_summary(dist, dens, y, ml_prob, base_prob))
    return out


def run_mississippi_target(target_key, device, nrounds=500):
    import killshot_mississippi_salinity_spatial as ms

    spec = ms.TARGETS[target_key]
    df, features = ms.read_target(spec, smoke=False)
    x, encoded_features = ms.encode_features(df, features)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    train_idx, test_idx = idx[:cut], idx[cut:]
    y = df["_target"].to_numpy(dtype=float)
    ml_pred = ms.fit_predict_model(
        "xgboost", x, y, train_idx, test_idx, device=device,
        nrounds=nrounds, nthread=8, seed=SEED)
    base_pred = ms.idw_predict(df, train_idx, test_idx, k=8)
    dist, dens = nearest_dist_and_density(
        df.iloc[train_idx][ms.LAT].values, df.iloc[train_idx][ms.LON].values,
        df.iloc[test_idx][ms.LAT].values, df.iloc[test_idx][ms.LON].values)
    r2fn = lambda yy, pp: r2_score(yy, pp)
    bins = _bin_and_report(dist, r2fn, r2fn, y[test_idx], ml_pred, base_pred, "R2")
    out = {
        "metric": "R2 (regression)",
        "baseline": "IDW-8",
        "n_test": int(len(test_idx)),
        "n_encoded_features": int(len(encoded_features)),
        "bins": bins,
    }
    out.update(_common_error_summary(dist, dens, y[test_idx], ml_pred, base_pred))
    return out


def _check_outputs():
    if not OUT_JSON.exists() or not OUT_BINS.exists():
        raise SystemExit("CHECK FAIL: mechanism output files are missing.")
    res = json.load(open(OUT_JSON))
    missing = [k for k in TARGET_KEYS if k not in res]
    if missing:
        raise SystemExit(f"CHECK FAIL: missing target keys: {missing}")
    for key in TARGET_KEYS:
        item = res[key]
        if not item.get("bins"):
            raise SystemExit(f"CHECK FAIL: {key} has no bins")
        if "metric" not in item or "baseline" not in item:
            raise SystemExit(f"CHECK FAIL: {key} missing metric/baseline metadata")
        corr_keys = [k for k in item if k.startswith("spearman_")]
        if len(corr_keys) < 3:
            raise SystemExit(f"CHECK FAIL: {key} missing correlation fields")
    bins = pd.read_csv(OUT_BINS)
    expected_rows = sum(len(res[k]["bins"]) for k in TARGET_KEYS)
    if len(bins) != expected_rows:
        raise SystemExit(f"CHECK FAIL: CSV rows {len(bins)} != JSON bins {expected_rows}")
    if set(bins["dataset"]) != set(TARGET_KEYS):
        raise SystemExit("CHECK FAIL: CSV dataset keys do not match expected targets")
    print("CHECK PASS")
    print(f"  targets: {', '.join(TARGET_KEYS)}")
    print(f"  bin rows: {len(bins)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--mississippi-nrounds", type=int, default=500)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        _check_outputs()
        return

    res = {
        "status": "DESCRIPTIVE mechanism illustration (NOT a predictive/universal curve; the predictive version was falsified on GROW).",
        "design": ("Standard random 80/20 split (seed 1182). Bin test points by nearest-training-well distance; "
                   "per bin report ML skill and zero-learning neighbour-baseline skill. Report Spearman(distance, error) "
                   "descriptively. No fitted correction curve, no extrapolation."),
        "density_radius_km": DENS_RADIUS_KM,
        "run_metadata": {
            "seed": SEED,
            "device_requested": args.device,
            "mississippi_nrounds": args.mississippi_nrounds,
        },
    }
    print("[nitrate] spatial mechanism map ...", flush=True)
    res["nitrate"] = run_nitrate(args.device)
    print("  bins:", [(b["median_dist_km"], b["ml_skill"], b["baseline_skill"]) for b in res["nitrate"]["bins"]], flush=True)
    print("[fluoride] spatial mechanism map ...", flush=True)
    res["fluoride"] = run_fluoride(args.device)
    print("  bins:", [(b["median_dist_km"], b["ml_skill"], b["baseline_skill"]) for b in res["fluoride"]["bins"]], flush=True)
    print("[assam arsenic] spatial mechanism map ...", flush=True)
    res["assam_arsenic"] = run_assam_arsenic(args.device)
    print("  bins:", [(b["median_dist_km"], b["ml_skill"], b["baseline_skill"]) for b in res["assam_arsenic"]["bins"]], flush=True)
    print("[mississippi specific conductance] spatial mechanism map ...", flush=True)
    res["mississippi_specific_conductance"] = run_mississippi_target(
        "specific_conductance", args.device, args.mississippi_nrounds)
    print("  bins:", [(b["median_dist_km"], b["ml_skill"], b["baseline_skill"]) for b in res["mississippi_specific_conductance"]["bins"]], flush=True)
    print("[mississippi chloride] spatial mechanism map ...", flush=True)
    res["mississippi_chloride"] = run_mississippi_target(
        "chloride", args.device, args.mississippi_nrounds)
    print("  bins:", [(b["median_dist_km"], b["ml_skill"], b["baseline_skill"]) for b in res["mississippi_chloride"]["bins"]], flush=True)
    density_vals = {
        k: res[k].get("spearman_density_vs_ml_abserr", res[k].get("spearman_density_vs_ml_proberr"))
        for k in TARGET_KEYS
    }
    res["writing_summary"] = {
        "target_keys": list(TARGET_KEYS),
        "density_radius_km": DENS_RADIUS_KM,
        "density_vs_ml_error_spearman": density_vals,
        "reading": (
            "Mechanism evidence is dataset-dependent. The coordinate-based "
            "datasets do not support a single universal near-neighbour leakage "
            "mechanism; use the map as bounded mechanism evidence for random "
            "validation optimism and spatial non-transferability."
        ),
    }
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    # flatten bins to CSV for plotting
    rows = []
    for ds in TARGET_KEYS:
        for b in res[ds]["bins"]:
            rows.append({"dataset": ds, **b})
    pd.DataFrame(rows).to_csv(OUT_BINS, index=False)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_BINS}")


if __name__ == "__main__":
    main()
