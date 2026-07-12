#!/usr/bin/env python3
"""Recompute the seven-row ACR table from released fold-level inputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXPECTED = PROC / "acr_unified_table.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def result_row(axis: str, dataset: str, metric: str, ml_value: float,
               baseline_value: float, baseline_form: str) -> dict[str, object]:
    ml_skill = ml_value if metric == "R2" else ml_value - 0.5
    baseline_skill = baseline_value if metric == "R2" else baseline_value - 0.5
    recovery = baseline_skill / ml_skill * 100 if ml_skill > 0 else math.nan
    return {
        "axis": axis,
        "dataset": dataset,
        "metric": metric,
        "ml_value": round(ml_value, 4),
        "baseline_value": round(baseline_value, 4),
        "baseline_form": baseline_form,
        "ml_skill": round(ml_skill, 4),
        "baseline_skill": round(baseline_skill, 4),
        "increment_ml_minus_baseline": round(ml_skill - baseline_skill, 4),
        "acr_pct": round(recovery, 1),
    }


def spatial(dataset: str, subdir: str, metric: str, baseline_form: str) -> dict[str, object]:
    rows = read_rows(PROC / subdir / "acr_folds.csv")
    return result_row(
        "spatial", dataset, metric,
        median(float(row["ml_value"]) for row in rows),
        median(float(row["base_value"]) for row in rows),
        baseline_form,
    )


def temporal(dataset: str, path: str, ml_column: str, baseline_column: str) -> dict[str, object]:
    rows = read_rows(PROC / path)
    return result_row(
        "temporal", dataset, "R2",
        median(float(row[ml_column]) for row in rows if row[ml_column]),
        median(float(row[baseline_column]) for row in rows if row[baseline_column]),
        "persistence",
    )


def build() -> list[dict[str, object]]:
    return [
        spatial("nitrate", "usgs_nitrate", "R2", "IDW-8 (distance)"),
        spatial("fluoride", "global_fluoride", "AUC", "IDW-freq-16 (distance)"),
        spatial("arsenic", "usgs_arsenic", "AUC", "ecoregion-mean (categorical)"),
        spatial("assam_arsenic", "assam_arsenic", "AUC", "IDW-freq-8 (distance)"),
        temporal("gems_de", "gems_wells/spatial_cv_results.csv", "r2_random_split", "r2_persistence"),
        temporal("ncp_cn", "ncp_wells/spatial_cv_results.csv", "r2_random_split", "r2_persistence"),
        temporal("grow_16", "grow/persistence_vs_ml_by_country.csv", "ml_random_r2", "persistence_r2"),
    ]


def check(rows: list[dict[str, object]]) -> None:
    expected = {row["dataset"]: row for row in read_rows(EXPECTED)}
    if len(rows) != 7 or len(expected) != 7:
        raise SystemExit("ACR table must contain exactly seven rows")
    drift = []
    for row in rows:
        old = expected.get(str(row["dataset"]))
        if old is None or abs(float(old["acr_pct"]) - float(row["acr_pct"])) > 0.1:
            drift.append((row["dataset"], None if old is None else old["acr_pct"], row["acr_pct"]))
    if drift:
        raise SystemExit(f"ACR drift detected: {drift}")
    print("PASS: seven-row ACR table reproduces the released reference values.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with released acr_unified_table.csv")
    args = parser.parse_args()
    rows = build()
    for row in rows:
        print(
            f"{row['axis']:8} {row['dataset']:15} "
            f"ML={row['ml_value']:.4f} baseline={row['baseline_value']:.4f} "
            f"increment={row['increment_ml_minus_baseline']:.4f} ACR={row['acr_pct']:.1f}%"
        )
    if args.check:
        check(rows)


if __name__ == "__main__":
    main()
