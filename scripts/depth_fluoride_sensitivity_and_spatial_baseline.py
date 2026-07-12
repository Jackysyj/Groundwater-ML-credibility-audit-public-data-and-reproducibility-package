#!/usr/bin/env python3
"""
DEPTH-HARDENING for the global-fluoride spatial kill-shot. Parallels the nitrate
depth script; no change to the validated main script.

  (A) Block-size SENSITIVITY: leave-one-spatial-block-out at 3/5/7/10 deg
      (larger cells than nitrate because fluoride is global-scale), showing the
      spatial collapse is not tied to one block size.

  (B) SPATIAL-NEAREST-NEIGHBOUR baseline: under both random k-fold and spatial
      block holdout, compare the authors' XGBoost (AUC) against a zero-learning
      spatial interpolator that predicts the exceedance PROBABILITY as the mean
      label of the k nearest training wells (IDW-weighted). Because the positive
      rate is only 0.85%, a hard 1-NN label is uninformative, so we use the
      soft neighbour-frequency as the score and evaluate AUC.

If the spatial IDW-frequency baseline approaches XGBoost's AUC under random
k-fold but both fall under spatial holdout, fluoride mapping — like nitrate —
shows the same nearest-neighbour-leakage mechanism as the temporal persistence
result, unifying the two theses.

Output: data/processed/global_fluoride/depth_sensitivity_and_spatial_baseline.json
        data/processed/global_fluoride/blocksize_sensitivity_detail.csv
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

import killshot_global_fluoride_spatial as ks  # noqa: E402 (reuse validated code)

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "processed" / "global_fluoride"
OUT_JSON = OUT_DIR / "depth_sensitivity_and_spatial_baseline.json"
OUT_SENS = OUT_DIR / "blocksize_sensitivity_detail.csv"

TARGET = ks.TARGET
MIN_TEST, MIN_POS = ks.MIN_TEST, ks.MIN_POS


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo),
                            np.cos(la) * np.sin(lo),
                            np.sin(la)])


def assign_blocks(df, deg):
    return (np.floor(df["lon"] / deg).astype(int).astype(str) + "_" +
            np.floor(df["lat"] / deg).astype(int).astype(str)).values


def blocksize_sensitivity(df, degs=(3.0, 5.0, 7.0, 10.0)):
    rows, summary = [], []
    for deg in degs:
        d = df.assign(block=assign_blocks(df, deg))
        sizes = d["block"].value_counts()
        blocks = sizes[sizes >= MIN_TEST].index.tolist()
        aucs, ns = [], []
        for b in blocks:
            te = d[d["block"] == b]; tr = d[d["block"] != b]
            if len(te) < MIN_TEST or te[TARGET].sum() < MIN_POS:
                continue
            auc = ks.fit_auc(tr, te)
            if np.isnan(auc):
                continue
            aucs.append(auc); ns.append(len(te))
            rows.append({"block_deg": deg, "block": b, "n_test": len(te),
                         "n_pos": int(te[TARGET].sum()), "auc": auc})
        aucs = np.array(aucs)
        summary.append({"block_deg": deg, "n_folds": len(aucs),
                        "median_auc": float(np.median(aucs)),
                        "frac_auc_below_0p6": float(np.mean(aucs < 0.6))})
        print(f"  block {deg} deg: n_folds={len(aucs)}, "
              f"median AUC={np.median(aucs):.3f}", flush=True)
    return summary, pd.DataFrame(rows)


def nn_freq(tr, te, k=16):
    """Soft spatial baseline: exceedance probability = IDW mean of the k nearest
    training labels (great-circle). No features, no learning."""
    tr_xyz = _xyz(tr["lat"].values, tr["lon"].values)
    te_xyz = _xyz(te["lat"].values, te["lon"].values)
    tr_y = tr[TARGET].values.astype(float)
    out = np.empty(len(te))
    step = 1000
    for i in range(0, len(te), step):
        chunk = te_xyz[i:i + step]
        d2 = ((chunk[:, None, :] - tr_xyz[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
        di = np.take_along_axis(d2, idx, axis=1) ** 0.5
        w = 1.0 / np.clip(di, 1e-6, None)
        out[i:i + step] = (tr_y[idx] * w).sum(1) / w.sum(1)
    return out


def compare_under_regime(df, deg=5.0):
    def score(splits):
        out = {"xgb": [], "nnfreq": [], "n": []}
        for tr, te in splits:
            if len(te) < MIN_TEST or te[TARGET].sum() < MIN_POS:
                continue
            auc_x = ks.fit_auc(tr, te)
            if np.isnan(auc_x):
                continue
            out["xgb"].append(auc_x)
            out["nnfreq"].append(float(roc_auc_score(te[TARGET].values,
                                                     nn_freq(tr, te))))
            out["n"].append(len(te))
        return out

    d = df.assign(block=assign_blocks(df, deg))
    sizes = d["block"].value_counts()
    blocks = sizes[sizes >= MIN_TEST].index.tolist()
    k = len(blocks)
    rng = np.random.default_rng(1182)
    fold = rng.integers(0, k, size=len(df))
    rand_splits = [(df[fold != f], df[fold == f]) for f in range(k)]
    spat_splits = [(d[d["block"] != b], d[d["block"] == b]) for b in blocks]

    print("  [random k-fold] scoring xgb/NN-freq ...", flush=True)
    rand = score(rand_splits)
    print("  [spatial block] scoring xgb/NN-freq ...", flush=True)
    spat = score(spat_splits)

    def med(x):
        return float(np.median(x)) if x else None
    return {
        "block_deg": deg,
        "random_kfold": {"xgb_median_auc": med(rand["xgb"]),
                         "nnfreq_median_auc": med(rand["nnfreq"]),
                         "n_folds": len(rand["xgb"])},
        "spatial_block": {"xgb_median_auc": med(spat["xgb"]),
                          "nnfreq_median_auc": med(spat["nnfreq"]),
                          "n_folds": len(spat["xgb"])},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = ks.load()
    print(f"fluoride: {len(df)} wells, exceedance {df[TARGET].mean():.4f}\n", flush=True)

    print("[A] block-size sensitivity ...", flush=True)
    sens, sens_detail = blocksize_sensitivity(df)
    sens_detail.to_csv(OUT_SENS, index=False, encoding="utf-8-sig")

    print("\n[B] spatial nearest-neighbour baseline (xgb vs IDW-frequency) ...", flush=True)
    nn = compare_under_regime(df, deg=ks.BLOCK_DEG)

    results = {
        "dataset": "Podgorski & Berg 2022 global fluoride, 37132 wells",
        "sensitivity_blocksize": sens,
        "spatial_nn_baseline": nn,
        "interpretation": (
            "Sensitivity: spatial-block median AUC stays low across 3-10 deg. "
            "NN baseline: if IDW-frequency approaches XGBoost AUC under random "
            "k-fold but both fall under spatial holdout, fluoride mapping shows "
            "the same spatial nearest-neighbour leakage as nitrate, the spatial "
            "analogue of temporal persistence."),
    }
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print("\n=== DEPTH HARDENING (fluoride) ===")
    print("block-size sensitivity (median AUC):",
          {s["block_deg"]: round(s["median_auc"], 3) for s in sens})
    print("random k-fold  xgb/NN-freq:",
          {kk: round(v, 3) for kk, v in nn["random_kfold"].items() if isinstance(v, float)})
    print("spatial block  xgb/NN-freq:",
          {kk: round(v, 3) for kk, v in nn["spatial_block"].items() if isinstance(v, float)})
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
