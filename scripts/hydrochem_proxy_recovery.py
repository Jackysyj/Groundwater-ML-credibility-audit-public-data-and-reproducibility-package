#!/usr/bin/env python3
"""Hydrochemical proxy-recovery probe for the Free-Baseline Recovery family.

This is an exploratory, external-data probe. It does NOT update
manuscript_stats.json and it is NOT part of the narrow autocorrelation ACR
table. It asks one question: when the target is predicted from co-measured
hydrochemical variables, how much of flexible ML skill is recovered by a cheap
hydrochemical proxy baseline on the same folds?

Outputs:
  data/processed/hydrochem_proxy_recovery/dataset_inventory.json
  data/processed/hydrochem_proxy_recovery/fold_results.csv
  data/processed/hydrochem_proxy_recovery/target_summary.csv
  data/processed/hydrochem_proxy_recovery/hydrochem_proxy_recovery_results.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hydrochem_proxy_recovery"
OUT_DIR = ROOT / "data" / "processed" / "hydrochem_proxy_recovery"

SEED = 42
N_FOLDS = 5
SEEDS = [42, 1182, 20260706, 7, 99]
MAX_N = 20_000
MIN_REG_N = 80
MIN_CLASS_N = 100
MIN_POS = 20
MIN_NEG = 20

NO_SKILL_FLOOR = {"regression": 0.0, "classification": 0.5}


@dataclass
class TargetSpec:
    dataset: str
    target: str
    target_col: str
    task: str
    threshold: float | None = None
    target_units: str | None = None


@dataclass
class DatasetSpec:
    name: str
    df: pd.DataFrame
    predictor_cols: list[str]
    targets: list[TargetSpec]
    group_cols: list[str]
    provenance: dict[str, Any]


def clean_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def norm_token(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def to_numeric_series(s: pd.Series) -> pd.Series:
    def conv(v: Any) -> float:
        if pd.isna(v):
            return np.nan
        txt = str(v).strip()
        if txt in {"", "-", "--", "NA", "NaN", "nan", "NC"}:
            return np.nan
        txt = txt.replace(",", "")
        if txt.startswith("<"):
            # Conservative half-reporting-limit treatment.
            try:
                return float(txt[1:].strip()) / 2.0
            except Exception:
                return np.nan
        if txt.startswith(">"):
            txt = txt[1:].strip()
        try:
            return float(txt)
        except Exception:
            return np.nan

    return s.map(conv).astype(float)


def numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: to_numeric_series(df[c]) for c in cols if c in df.columns})


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def skill_from_metric(task: str, metric_value: float | None) -> float | None:
    if metric_value is None or not np.isfinite(metric_value):
        return None
    if task == "regression":
        return max(0.0, float(metric_value))
    return max(0.0, float(metric_value) - NO_SKILL_FLOOR["classification"])


def recovery_pct(baseline_skill: float | None, ml_skill: float | None) -> float | None:
    if baseline_skill is None or ml_skill is None or ml_skill <= 0:
        return None
    return float(100.0 * baseline_skill / ml_skill)


def read_csv_from_zip(zip_path: Path, member: str, **kwargs: Any) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            return pd.read_csv(fh, **kwargs)


def load_usgs() -> DatasetSpec:
    zip_path = RAW / "usgs_nawqa_2017_2019_DR_2017-19_Tables.zip"
    key = ["Network_type", "Network_name", "NAWQA_well_identification_number", "State", "Sample_date", "Sample_time"]
    table_members = [
        "DSR_2017-19_Table3_QWIndicators.txt",
        "DSR_2017-19_Table4_Nutrients.txt",
        "DSR_2017-19_Table5_MajorIons.txt",
        "DSR_2017-19_Table6_TraceElements.txt",
    ]
    merged: pd.DataFrame | None = None
    for member in table_members:
        d = read_csv_from_zip(zip_path, member, sep="\t", dtype=str)
        d.columns = [clean_name(c) for c in d.columns]
        d = d[[c for c in d.columns if c]]
        if merged is None:
            merged = d
        else:
            use = [c for c in d.columns if c in key or c not in merged.columns]
            merged = merged.merge(d[use], on=key, how="outer")
    assert merged is not None
    df = merged.copy()

    rename = {
        "V_P00400_pH": "pH",
        "V_P00300_Dissolved_oxygen": "dissolved_oxygen",
        "V_P00095_Specific_cond_at_25C": "specific_conductance",
        "V_P00010_Temperature_water": "temperature",
        "V_P29802_Alkalinity_wf_Gran_field": "alkalinity",
        "V_P63786_Bicarbonate_wf_Gran_field": "bicarbonate",
        "V_P00631_NO3+NO2_wf": "nitrate_nitrite",
        "V_P00608_Ammonia_wf": "ammonia",
        "V_P00681_Organic_carbon_wf": "organic_carbon",
        "V_P00915_Calcium_wf": "calcium",
        "V_P00925_Magnesium_wf": "magnesium",
        "V_P00935_Potassium_wf": "potassium",
        "V_P00930_Sodium_wf": "sodium",
        "V_P00940_Chloride_wf": "chloride",
        "V_P00950_Fluoride_wf": "fluoride",
        "V_P00955_Silica_wf": "silica",
        "V_P00945_Sulfate_wf": "sulfate",
        "V_P70300_Total_diss_solids_dry_at_180C": "tds",
        "V_P01046_Iron_wf": "iron",
        "V_P01000_Arsenic_wf": "arsenic",
        "V_P01020_Boron_wf": "boron",
        "V_P01056_Manganese_wf": "manganese",
        "V_P01130_Lithium_wf": "lithium",
        "V_P01060_Molybdenum_wf": "molybdenum",
        "V_P01145_Selenium_wf": "selenium",
        "V_P22703_Uranium_wf": "uranium",
    }
    # Keep only columns that actually exist; USGS table names can include odd
    # header glitches, so missing optional targets are handled downstream.
    inv = {v: k for k, v in rename.items()}
    for raw, new in rename.items():
        if raw in df.columns:
            df[new] = to_numeric_series(df[raw])

    predictors = [
        "pH", "dissolved_oxygen", "specific_conductance", "temperature",
        "alkalinity", "bicarbonate", "ammonia", "organic_carbon",
        "calcium", "magnesium", "potassium", "sodium", "chloride",
        "silica", "sulfate", "tds", "iron", "boron", "lithium",
        "molybdenum", "selenium",
    ]
    predictors = [c for c in predictors if c in df.columns]
    targets = [
        TargetSpec("usgs_nawqa_2017_2019", "arsenic_class_10ugL", "arsenic", "classification", 10.0, "ug/L"),
        TargetSpec("usgs_nawqa_2017_2019", "nitrate_regression", "nitrate_nitrite", "regression", None, "mg/L as N"),
        TargetSpec("usgs_nawqa_2017_2019", "fluoride_class_1p5mgL", "fluoride", "classification", 1.5, "mg/L"),
        TargetSpec("usgs_nawqa_2017_2019", "manganese_class_50ugL", "manganese", "classification", 50.0, "ug/L"),
        TargetSpec("usgs_nawqa_2017_2019", "iron_class_300ugL", "iron", "classification", 300.0, "ug/L"),
    ]
    targets = [t for t in targets if t.target_col in df.columns]
    return DatasetSpec(
        name="usgs_nawqa_2017_2019",
        df=df,
        predictor_cols=predictors,
        targets=targets,
        group_cols=["State", "Network_name"],
        provenance={
            "source_file": str(zip_path.relative_to(ROOT)),
            "external_to_corpus_check": "government_dataset; not DOI-table paper",
            "renamed_columns_present": {k: v for k, v in rename.items() if k in df.columns},
            "optional_missing_raw_columns": sorted(set(rename) - set(df.columns)),
        },
    )


def load_cgwb() -> DatasetSpec:
    folder = RAW / "india_cgwb_groundwater_quality_csvs"
    files = sorted(p for p in folder.glob("*_1961_2025.csv") if " (1)" not in p.name and ":Zone.Identifier" not in p.name)
    frames = []
    for p in files:
        d = pd.read_csv(p, dtype=str)
        d["source_file"] = p.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [clean_name(c) for c in df.columns]
    rename = {
        "Potential of Hydrogen (pH)": "pH",
        "Electric Conductivity (μS/cm)": "ec",
        "Total Dissolved Solids (mg/L)": "tds",
        "Carbonate (mg/L)": "carbonate",
        "Bicarbonate (mg/L)": "bicarbonate",
        "Total Alkalinity (mg/L as CaCO3)": "alkalinity",
        "Chloride (mg/L)": "chloride",
        "Nitrate N (mgN/L)": "nitrate_n",
        "Sulphate (mg/L)": "sulfate",
        "Phosphate(mg/L)": "phosphate",
        "Silica(mg/L)": "silica",
        "Fluoride (mg/L)": "fluoride",
        "Total Hardness (mgCaCO3/L)": "hardness",
        "Calcium (mg/L)": "calcium",
        "Magnesium (mg/L)": "magnesium",
        "Sodium (mg/L)": "sodium",
        "Potassium (mg/L)": "potassium",
        "Iron(mg/L)": "iron",
        "Arsenic (mg/L)": "arsenic",
        "Uranium(mg/L)": "uranium",
        "Manganese (mg/L)": "manganese",
        "Copper (mg/L)": "copper",
        "Lead (mg/L)": "lead",
        "Zinc (mg/L)": "zinc",
        "Nickel (mg/L)": "nickel",
        "Cadmium (mg/L)": "cadmium",
        "Chromium (mg/L)": "chromium",
    }
    for raw, new in rename.items():
        if raw in df.columns:
            df[new] = to_numeric_series(df[raw])
    # Well identity for the same-well persistence probe. A CGWB well is a
    # Station within a State (Station labels can repeat across states), and the
    # panel is longitudinal (median 6, up to 51 distinct sampling times per
    # well). well_id lets us (a) hold out whole wells and (b) build a
    # feature-free same-well baseline that only reads a well's other rows.
    df["well_id"] = (
        df["State"].astype(str).str.strip() + "||" + df["Station"].astype(str).str.strip()
    )
    predictors = [
        "pH", "ec", "tds", "carbonate", "bicarbonate", "alkalinity",
        "chloride", "sulfate", "phosphate", "silica", "hardness",
        "calcium", "magnesium", "sodium", "potassium", "iron",
        "copper", "lead", "zinc", "nickel", "cadmium", "chromium",
    ]
    predictors = [c for c in predictors if c in df.columns]
    # Fluoride (mg/L) and Nitrate N (mgN/L) are present as column headers but
    # entirely empty across all 30 downloaded CGWB 1961_2025 state exports
    # (0 non-null values corpus-wide, verified 2026-07-06), so they can never
    # score from this source and are not declared as targets. Arsenic, uranium
    # and manganese are the trainable exceedance targets here.
    targets = [
        TargetSpec("india_cgwb", "arsenic_class_0p01mgL", "arsenic", "classification", 0.01, "mg/L"),
        TargetSpec("india_cgwb", "uranium_class_0p03mgL", "uranium", "classification", 0.03, "mg/L"),
        TargetSpec("india_cgwb", "manganese_class_0p1mgL", "manganese", "classification", 0.1, "mg/L"),
    ]
    return DatasetSpec(
        name="india_cgwb",
        df=df,
        predictor_cols=predictors,
        targets=targets,
        # well_id first so the same-well holdout is the primary group diagnostic;
        # State retained as the coarser cross-region holdout.
        group_cols=["well_id", "State"],
        provenance={
            "source_folder": str(folder.relative_to(ROOT)),
            "n_csv_1961_2025_loaded": len(files),
            "ignored": "2026_2030 empty shells, Zone.Identifier files, duplicate '(1)' CSVs",
            "external_to_corpus_check": "government_dataset; not DOI-table paper",
            "well_id_definition": "State||Station; longitudinal panel with repeat sampling",
            "empty_source_columns": "Fluoride (mg/L) and Nitrate N (mgN/L) headers exist but are entirely empty across all state exports; dropped as targets",
        },
    )


def read_east_anglia_csv(zip_path: Path) -> pd.DataFrame:
    member = "Groundwater chemistry for East Anglia Chalk/anglian regional data 250424.csv"
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(member).decode("utf-8-sig", errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(raw):
        cells = next(csv.reader([line]))
        if cells and clean_name(cells[0]) == "Serial":
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("East Anglia header row not found")
    return pd.read_csv(io.StringIO("\n".join(raw)), skiprows=header_idx)


def load_east_anglia() -> DatasetSpec:
    zip_path = RAW / "east_anglia_chalk_groundwater_chemistry_mendeley_gf5c94fk4y.zip"
    df = read_east_anglia_csv(zip_path)
    df.columns = [clean_name(c) for c in df.columns]
    rename = {
        "East": "easting",
        "North": "northing",
        "Ca": "calcium",
        "Mg": "magnesium",
        "Na": "sodium",
        "K": "potassium",
        "SO4": "sulfate",
        "Cl": "chloride",
        "NO3": "nitrate",
        "F": "fluoride",
        "Sr": "strontium",
        "Fe": "iron",
        "Alk": "alkalinity",
        "pH": "pH",
        "pH_adj": "pH_adj",
        "PCO2_adj": "pco2_adj",
    }
    for raw, new in rename.items():
        if raw in df.columns:
            df[new] = to_numeric_series(df[raw])
    predictors = [
        "calcium", "magnesium", "sodium", "potassium", "sulfate",
        "chloride", "iron", "alkalinity", "pH", "pH_adj", "pco2_adj",
    ]
    predictors = [c for c in predictors if c in df.columns]
    targets = [
        TargetSpec("east_anglia_chalk", "nitrate_regression", "nitrate", "regression", None, "mg/L"),
        TargetSpec("east_anglia_chalk", "fluoride_regression", "fluoride", "regression", None, "mg/L"),
        TargetSpec("east_anglia_chalk", "strontium_regression", "strontium", "regression", None, "mg/L"),
    ]
    return DatasetSpec(
        name="east_anglia_chalk",
        df=df,
        predictor_cols=predictors,
        targets=targets,
        group_cols=[],
        provenance={
            "source_file": str(zip_path.relative_to(ROOT)),
            "member": "anglian regional data 250424.csv",
            "header_detection": "row whose first cell is Serial (line 32 in downloaded file)",
            "external_to_corpus_check": "DOI 10.1016/j.ejrh.2025.102383 and data DOI gf5c94fk4y not found in DOI.xlsx",
        },
    )


def normalize_argentina_block(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    cols = [c for c in df.columns if (suffix == "" and not str(c).endswith(".1")) or (suffix == ".1" and str(c).endswith(".1"))]
    if suffix == "":
        cols = [c for c in cols if c != "Unnamed: 23"]
    out = df[cols].copy()
    if suffix:
        out.columns = [str(c).replace(suffix, "") for c in out.columns]
    return out


def load_argentina() -> DatasetSpec:
    zip_path = RAW / "Hydrological dataset from a large sub-humid plain.zip"
    with zipfile.ZipFile(zip_path) as zf:
        member = [n for n in zf.namelist() if n.endswith("Supplementary data.xlsx")][0]
        data = zf.read(member)
    with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
        tmp.write(data)
        tmp.flush()
        raw = pd.read_excel(tmp.name, sheet_name="Groundwater")
    left = normalize_argentina_block(raw, "")
    right = normalize_argentina_block(raw, ".1")
    df = pd.concat([left, right], ignore_index=True)
    df = df[df["ID"].notna()].copy()
    df.columns = [clean_name(c) for c in df.columns]
    rename = {
        "Water table": "water_table",
        "EC": "ec",
        "EC ": "ec",
        "pH": "pH",
        "TA": "alkalinity",
        "TA ": "alkalinity",
        "HCO3-": "bicarbonate",
        "CO32-": "carbonate",
        "CO32- ": "carbonate",
        "Cl-": "chloride",
        "SO42-": "sulfate",
        "SO42- ": "sulfate",
        "NO3-": "nitrate",
        "NO3- ": "nitrate",
        "F-": "fluoride",
        "As": "arsenic",
        "Na+": "sodium",
        "Na+ ": "sodium",
        "K+": "potassium",
        "Ca2+": "calcium",
        "Mg2+": "magnesium",
        "*SiO2": "silica",
        "δ18O": "delta18o",
        "δ18O ": "delta18o",
        "δ2H": "delta2h",
        "δ2H ": "delta2h",
    }
    # Collapse possible duplicate source names by first valid value.
    for raw_col, new in rename.items():
        if raw_col in df.columns:
            val = to_numeric_series(df[raw_col])
            if new in df.columns:
                df[new] = to_numeric_series(df[new]).combine_first(val)
            else:
                df[new] = val
    predictors = [
        "water_table", "ec", "pH", "alkalinity", "bicarbonate", "carbonate",
        "chloride", "sulfate", "sodium", "potassium", "calcium", "magnesium",
        "silica", "delta18o", "delta2h",
    ]
    predictors = [c for c in predictors if c in df.columns]
    targets = [
        TargetSpec("argentina_subhumid_plain", "arsenic_regression", "arsenic", "regression", None, "ug/L"),
        TargetSpec("argentina_subhumid_plain", "fluoride_regression", "fluoride", "regression", None, "mg/L"),
        TargetSpec("argentina_subhumid_plain", "nitrate_regression", "nitrate", "regression", None, "mg/L"),
    ]
    return DatasetSpec(
        name="argentina_subhumid_plain",
        df=df,
        predictor_cols=predictors,
        targets=targets,
        group_cols=[],
        provenance={
            "source_file": str(zip_path.relative_to(ROOT)),
            "workbook_member": member,
            "sheet": "Groundwater",
            "reshape": "left and right groundwater blocks converted to one long table",
            "role": "small-sample external sanity check, not headline evidence",
            "external_to_corpus_check": "data DOI b34kg4jx7r not found in DOI.xlsx",
        },
    )


def direct_derivative_tokens(target_col: str, target_name: str) -> set[str]:
    toks = {norm_token(target_col), norm_token(target_name)}
    t = norm_token(target_col + " " + target_name)
    if "arsenic" in t or t == "as":
        toks |= {"arsenic", "arsenate", "arsenite", "as", "totalas", "asiii", "asv", "dimethylarsinate", "monomethylarsonate"}
    if "nitrate" in t or "no3" in t:
        toks |= {"nitrate", "nitrite", "no3", "no2", "no3no2", "nitrogen", "totalnitrogen"}
    if "fluoride" in t or t in {"f", "fmg"}:
        toks |= {"fluoride", "f"}
    if "manganese" in t or t == "mn":
        toks |= {"manganese", "mn"}
    if "iron" in t or t == "fe":
        toks |= {"iron", "fe"}
    if "uranium" in t or t == "u":
        toks |= {"uranium", "u"}
    if "strontium" in t or t == "sr":
        toks |= {"strontium", "sr", "strontiumratio"}
    return toks


def predictor_set(spec: DatasetSpec, target: TargetSpec) -> tuple[list[str], list[str]]:
    banned = direct_derivative_tokens(target.target_col, target.target)
    kept, removed = [], []
    for c in spec.predictor_cols:
        if norm_token(c) in banned:
            removed.append(c)
        else:
            kept.append(c)
    # Explicit structural exclusions.
    structural = {"class", "group", "serial", "id", "station", "latitude", "longitude", "easting", "northing"}
    kept2 = []
    for c in kept:
        if norm_token(c) in structural:
            removed.append(c)
        else:
            kept2.append(c)
    return kept2, sorted(set(removed))


def make_target_vector(d: pd.DataFrame, target: TargetSpec) -> pd.Series:
    y = to_numeric_series(d[target.target_col])
    if target.task == "classification":
        return (y >= float(target.threshold)).astype(float).where(y.notna(), np.nan)
    return y


def prepare_analysis_data(spec: DatasetSpec, target: TargetSpec) -> dict[str, Any]:
    predictors, removed = predictor_set(spec, target)
    raw = spec.df.copy()
    if target.target_col not in raw.columns:
        return {"screened_out": True, "reason": f"target column missing: {target.target_col}"}
    X = numeric_frame(raw, predictors)
    y = make_target_vector(raw, target)
    keep = y.notna()
    if int(keep.sum()) == 0:
        return {
            "screened_out": True,
            "reason": f"target column has no non-missing numeric values: {target.target_col}",
            "n_trainable": 0,
            "usable_predictors": [],
            "removed_direct_predictors": removed,
        }
    X = X.loc[keep].copy()
    y = y.loc[keep].astype(float)
    # Drop predictors that are mostly missing or constant.
    missing_rate = X.isna().mean()
    X = X.loc[:, missing_rate <= 0.8]
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique >= 2]
    row_has_any = X.notna().any(axis=1)
    X = X.loc[row_has_any]
    y = y.loc[row_has_any]
    groups = {}
    for g in spec.group_cols:
        if g in raw.columns:
            groups[g] = raw.loc[keep, g].loc[row_has_any].astype(str).fillna("missing").values

    n = int(len(y))
    if len(X.columns) < 2:
        return {"screened_out": True, "reason": "fewer than 2 non-target usable predictors", "n_trainable": n}
    if target.task == "classification":
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if n < MIN_CLASS_N or pos < MIN_POS or neg < MIN_NEG:
            return {
                "screened_out": True,
                "reason": f"classification sample too small/imbalanced: n={n}, pos={pos}, neg={neg}",
                "n_trainable": n,
                "n_positive": pos,
                "n_negative": neg,
                "usable_predictors": list(X.columns),
                "removed_direct_predictors": removed,
            }
    else:
        if n < MIN_REG_N:
            return {
                "screened_out": True,
                "reason": f"regression sample too small: n={n}",
                "n_trainable": n,
                "usable_predictors": list(X.columns),
                "removed_direct_predictors": removed,
            }

    return {
        "screened_out": False,
        "X": X.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "groups": groups,
        "n_trainable": n,
        "usable_predictors": list(X.columns),
        "removed_direct_predictors": removed,
        "target_missing_rate": round(float(1.0 - keep.mean()), 4),
        "positive_count": int(y.sum()) if target.task == "classification" else None,
        "negative_count": int(len(y) - y.sum()) if target.task == "classification" else None,
    }


def sample_for_runtime(X: pd.DataFrame, y: pd.Series, groups: dict[str, np.ndarray], task: str, seed: int) -> tuple[pd.DataFrame, pd.Series, dict[str, np.ndarray], int]:
    n = len(y)
    if n <= MAX_N:
        return X, y, groups, n
    rng = np.random.default_rng(seed)
    if task == "classification":
        idx_pos = np.flatnonzero(y.values == 1)
        idx_neg = np.flatnonzero(y.values == 0)
        n_pos = max(MIN_POS, int(MAX_N * len(idx_pos) / n))
        n_pos = min(len(idx_pos), n_pos)
        n_neg = MAX_N - n_pos
        n_neg = min(len(idx_neg), n_neg)
        pick = np.concatenate([
            rng.choice(idx_pos, size=n_pos, replace=False),
            rng.choice(idx_neg, size=n_neg, replace=False),
        ])
    else:
        pick = rng.choice(np.arange(n), size=MAX_N, replace=False)
    pick = np.sort(pick)
    g2 = {k: np.asarray(v)[pick] for k, v in groups.items()}
    return X.iloc[pick].reset_index(drop=True), y.iloc[pick].reset_index(drop=True), g2, int(len(pick))


def split_indices(y: pd.Series, task: str, seed: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    n_splits = min(N_FOLDS, len(y))
    if task == "classification":
        counts = y.value_counts()
        min_class = int(counts.min()) if not counts.empty else 0
        n_splits = min(N_FOLDS, min_class)
        if n_splits < 2:
            return []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te, f"random_seed{seed}_fold{i}") for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y.astype(int)))]
    if n_splits < 2:
        return []
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te, f"random_seed{seed}_fold{i}") for i, (tr, te) in enumerate(cv.split(np.zeros(len(y))))]


def group_split_indices(y: pd.Series, groups: np.ndarray, task: str) -> list[tuple[np.ndarray, np.ndarray, str]]:
    g = pd.Series(groups).astype(str).fillna("missing").values
    unique = np.unique(g)
    if len(unique) < 3:
        return []
    n_splits = min(N_FOLDS, len(unique))
    rows = []
    cv = GroupKFold(n_splits=n_splits)
    for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups=g)):
        if task == "classification" and len(np.unique(y.iloc[te])) < 2:
            continue
        rows.append((tr, te, f"group_fold{i}"))
    return rows


def linear_model(task: str) -> Pipeline:
    if task == "classification":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    else:
        model = Ridge(alpha=1.0)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def rf_model(task: str) -> Pipeline:
    if task == "classification":
        model = RandomForestClassifier(n_estimators=250, min_samples_leaf=5, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample")
    else:
        model = RandomForestRegressor(n_estimators=250, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def hgb_model(task: str) -> Pipeline:
    if task == "classification":
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.05, random_state=SEED)
    else:
        model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.05, random_state=SEED)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def score_predictions(task: str, y_true: np.ndarray, pred: np.ndarray) -> tuple[str, float | None, float | None]:
    if task == "classification":
        auc = safe_auc(y_true.astype(int), pred)
        return "AUC", auc, skill_from_metric(task, auc)
    r2 = float(r2_score(y_true, pred))
    return "R2", r2, skill_from_metric(task, r2)


def evaluate_single_proxy(X: pd.DataFrame, y: pd.Series, tr: np.ndarray, te: np.ndarray, task: str) -> dict[str, Any]:
    best_col, best_train_skill = None, -np.inf
    for col in X.columns:
        xtr = X.iloc[tr][[col]]
        xte = X.iloc[te][[col]]
        if xtr[col].notna().sum() < 10 or xtr[col].nunique(dropna=True) < 2:
            continue
        pipe = linear_model(task)
        try:
            pipe.fit(xtr, y.iloc[tr])
            if task == "classification":
                score = pipe.predict_proba(xtr)[:, 1]
            else:
                score = pipe.predict(xtr)
            _, _, sk = score_predictions(task, y.iloc[tr].values, np.asarray(score))
            if sk is not None and sk > best_train_skill:
                best_train_skill = sk
                best_col = col
        except Exception:
            continue
    if best_col is None:
        return {"model": "single_best_proxy", "status": "failed", "reason": "no usable single proxy"}
    pipe = linear_model(task)
    pipe.fit(X.iloc[tr][[best_col]], y.iloc[tr])
    pred = pipe.predict_proba(X.iloc[te][[best_col]])[:, 1] if task == "classification" else pipe.predict(X.iloc[te][[best_col]])
    metric, value, skill = score_predictions(task, y.iloc[te].values, np.asarray(pred))
    return {
        "model": "single_best_proxy",
        "status": "ok",
        "metric": metric,
        "metric_value": value,
        "skill": skill,
        "selected_proxy": best_col,
        "train_selected_proxy_skill": best_train_skill,
    }


def evaluate_well_persistence(y: pd.Series, tr: np.ndarray, te: np.ndarray, task: str, well_id: np.ndarray | None) -> dict[str, Any]:
    """Feature-free same-well baseline.

    For each test row, predict the mean target of the *same well's other rows*
    seen in training; fall back to the global training mean when the well is
    unseen. This uses zero hydrochemical features. Under random CV the same
    well straddles train and test, so this baseline can "copy" a well's own
    history; under well holdout every test well is unseen, so it collapses to
    the global mean and its skill should fall to the no-skill floor. The gap
    between random-CV persistence skill and ML skill answers the question
    "is the reported score same-well memory or genuine feature mapping?".
    """
    if well_id is None:
        return {"model": "well_identity_persistence", "status": "failed", "reason": "no well identifier for this dataset"}
    wtr = np.asarray(well_id)[tr]
    wte = np.asarray(well_id)[te]
    ytr = y.iloc[tr].to_numpy(dtype=float)
    well_mean: dict[str, float] = {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for w, val in zip(wtr, ytr):
        sums[w] = sums.get(w, 0.0) + float(val)
        counts[w] = counts.get(w, 0) + 1
    for w in sums:
        well_mean[w] = sums[w] / counts[w]
    global_mean = float(ytr.mean())
    seen = 0
    pred = np.empty(len(te), dtype=float)
    for i, w in enumerate(wte):
        if w in well_mean:
            pred[i] = well_mean[w]
            seen += 1
        else:
            pred[i] = global_mean
    same_well_exposure = float(seen / len(te)) if len(te) else 0.0
    metric, value, skill = score_predictions(task, y.iloc[te].to_numpy(), pred)
    return {
        "model": "well_identity_persistence",
        "status": "ok",
        "metric": metric,
        "metric_value": value,
        "skill": skill,
        "selected_proxy": "same_well_mean",
        "train_selected_proxy_skill": same_well_exposure,
    }


def evaluate_model(name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series, tr: np.ndarray, te: np.ndarray, task: str) -> dict[str, Any]:
    try:
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict_proba(X.iloc[te])[:, 1] if task == "classification" else model.predict(X.iloc[te])
        metric, value, skill = score_predictions(task, y.iloc[te].values, np.asarray(pred))
        return {"model": name, "status": "ok", "metric": metric, "metric_value": value, "skill": skill}
    except Exception as exc:
        return {"model": name, "status": "failed", "reason": str(exc)}


def run_target(spec: DatasetSpec, target: TargetSpec, seed_mode: str = "multi") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prep = prepare_analysis_data(spec, target)
    inventory = {
        "dataset": spec.name,
        "target": target.target,
        "target_col": target.target_col,
        "task": target.task,
        "threshold": target.threshold,
        "target_units": target.target_units,
    }
    inventory.update({k: v for k, v in prep.items() if k not in {"X", "y", "groups"}})
    if prep.get("screened_out"):
        return [], inventory

    X, y, groups = prep["X"], prep["y"], prep["groups"]
    X, y, groups, analysis_n = sample_for_runtime(X, y, groups, target.task, SEED)
    inventory["analysis_n"] = analysis_n
    if target.task == "classification":
        inventory["analysis_positive_count"] = int(y.sum())
        inventory["analysis_negative_count"] = int(len(y) - y.sum())
    seeds = SEEDS if seed_mode == "multi" and analysis_n <= MAX_N else [SEED]
    all_rows = []

    well_id = groups.get("well_id")

    for seed in seeds:
        for tr, te, fold_label in split_indices(y, target.task, seed):
            rows = evaluate_fold(spec, target, X, y, tr, te, "random_cv", fold_label, seed, well_id)
            all_rows.extend(rows)

    # Group holdout is a robustness diagnostic and uses SEED-independent groups.
    # We iterate every declared group column (e.g. CGWB gets both same-well and
    # cross-state holdout) so same-well leakage and cross-region transfer are
    # separated rather than conflated.
    for group_col in spec.group_cols:
        if group_col in groups:
            for tr, te, fold_label in group_split_indices(y, groups[group_col], target.task):
                rows = evaluate_fold(spec, target, X, y, tr, te, f"group_holdout_{group_col}", fold_label, None, well_id)
                all_rows.extend(rows)
    return all_rows, inventory


def evaluate_fold(spec: DatasetSpec, target: TargetSpec, X: pd.DataFrame, y: pd.Series, tr: np.ndarray, te: np.ndarray, regime: str, fold_label: str, seed: int | None, well_id: np.ndarray | None = None) -> list[dict[str, Any]]:
    base = {
        "dataset": spec.name,
        "target": target.target,
        "task": target.task,
        "regime": regime,
        "fold": fold_label,
        "seed": seed,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }
    if target.task == "classification":
        base["test_positive_count"] = int(y.iloc[te].sum())
        base["test_prevalence"] = float(y.iloc[te].mean())
    else:
        base["test_positive_count"] = None
        base["test_prevalence"] = None
    evaluations = [
        evaluate_well_persistence(y, tr, te, target.task, well_id),
        evaluate_single_proxy(X, y, tr, te, target.task),
        evaluate_model("ridge_or_logistic", linear_model(target.task), X, y, tr, te, target.task),
        evaluate_model("random_forest", rf_model(target.task), X, y, tr, te, target.task),
        evaluate_model("hist_gradient_boosting", hgb_model(target.task), X, y, tr, te, target.task),
    ]
    rows = []
    for ev in evaluations:
        row = dict(base)
        row.update(ev)
        rows.append(row)
    return rows


def summarize_target(rows: list[dict[str, Any]], inventory_rows: list[dict[str, Any]]) -> pd.DataFrame:
    d = pd.DataFrame([r for r in rows if r.get("status") == "ok"])
    if d.empty:
        return pd.DataFrame()
    summaries = []
    for (dataset, target, task, regime), sub in d.groupby(["dataset", "target", "task", "regime"], dropna=False):
        sub = sub.copy()
        sub["seed_key"] = sub["seed"].astype("string").fillna("group")
        pivot = sub.pivot_table(index=["fold", "seed_key"], columns="model", values="skill", aggfunc="first")
        metric_pivot = sub.pivot_table(index=["fold", "seed_key"], columns="model", values="metric_value", aggfunc="first")
        for ml_model in ["hist_gradient_boosting", "random_forest"]:
            if ml_model not in pivot.columns:
                continue
            for baseline in ["well_identity_persistence", "single_best_proxy", "ridge_or_logistic"]:
                if baseline not in pivot.columns:
                    continue
                paired = pivot[[baseline, ml_model]].dropna()
                if paired.empty:
                    continue
                metric_pair = metric_pivot.reindex(paired.index)
                baseline_skill = float(paired[baseline].median())
                ml_skill = float(paired[ml_model].median())
                rec = {
                    "dataset": dataset,
                    "target": target,
                    "task": task,
                    "regime": regime,
                    "baseline_model": baseline,
                    "ml_model": ml_model,
                    "n_folds": int(len(paired)),
                    "baseline_median_skill": baseline_skill,
                    "ml_median_skill": ml_skill,
                    "baseline_median_metric": float(metric_pair[baseline].median()),
                    "ml_median_metric": float(metric_pair[ml_model].median()),
                    "ml_increment_over_baseline": float(ml_skill - baseline_skill),
                    "proxy_recovery_pct": recovery_pct(baseline_skill, ml_skill),
                }
                summaries.append(rec)
    return pd.DataFrame(summaries)


def compute() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [load_usgs(), load_cgwb(), load_east_anglia(), load_argentina()]
    all_fold_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    dataset_provenance = []

    for spec in specs:
        dataset_provenance.append({
            "dataset": spec.name,
            "n_raw_rows": int(len(spec.df)),
            "n_predictor_columns_declared": int(len(spec.predictor_cols)),
            "predictor_columns_declared": spec.predictor_cols,
            "targets_requested": [t.target for t in spec.targets],
            "group_cols": spec.group_cols,
            "provenance": spec.provenance,
        })
        print(f"[{spec.name}] rows={len(spec.df)} targets={len(spec.targets)}", flush=True)
        for target in spec.targets:
            rows, inv = run_target(spec, target)
            inventory.append(inv)
            all_fold_rows.extend(rows)
            status = "screened_out" if inv.get("screened_out") else "scored"
            print(f"  {target.target}: {status}; folds rows={len(rows)}", flush=True)

    fold_df = pd.DataFrame(all_fold_rows)
    summary_df = summarize_target(all_fold_rows, inventory)

    inventory_json = {
        "status": "computed",
        "analysis": "hydrochemical proxy recovery",
        "not_acr_narrow_table": True,
        "not_truth_table": True,
        "seed": SEED,
        "seeds_for_random_cv": SEEDS,
        "n_folds": N_FOLDS,
        "max_analysis_n_per_dataset_target": MAX_N,
        "datasets": dataset_provenance,
        "targets": inventory,
        "rules": {
            "direct_derivative_exclusion": "target self/direct derivative columns removed before fitting; removed columns listed per target",
            "structural_exclusion": "coordinates, IDs, Class/Group, station labels not used as hydrochemical predictors",
            "metric": "R2 for regression with floor 0 skill; AUC for classification with AUC-0.5 skill",
            "interpretation": "co-measured hydrochemical proxies recover reported ML skill; not spatial mapping failure; not ACR",
        },
    }
    summary_records = summary_df.to_dict("records") if not summary_df.empty else []
    random_headline = []
    same_well_contrast = []
    if not summary_df.empty:
        cand = summary_df[(summary_df["regime"] == "random_cv") & (summary_df["baseline_model"] == "ridge_or_logistic") & (summary_df["ml_model"] == "hist_gradient_boosting")].copy()
        cand = cand.sort_values(["dataset", "target"])
        random_headline = cand.to_dict("records")

        # Same-well persistence contrast: for each CGWB target, line up ML skill
        # under random CV, the feature-free same-well baseline under random CV,
        # and ML skill under whole-well holdout. This is the direct test of
        # "is the score same-well memory or genuine feature mapping?".
        persist = summary_df[
            (summary_df["ml_model"] == "hist_gradient_boosting")
            & (summary_df["baseline_model"] == "well_identity_persistence")
        ].copy()
        for _, r in persist[persist["regime"] == "random_cv"].iterrows():
            wl = summary_df[
                (summary_df["dataset"] == r["dataset"])
                & (summary_df["target"] == r["target"])
                & (summary_df["regime"] == "group_holdout_well_id")
                & (summary_df["ml_model"] == "hist_gradient_boosting")
                & (summary_df["baseline_model"] == "ridge_or_logistic")
            ]
            same_well_contrast.append({
                "dataset": r["dataset"],
                "target": r["target"],
                "task": r["task"],
                "ml_skill_random_cv": round(float(r["ml_median_skill"]), 4),
                "same_well_persistence_skill_random_cv": round(float(r["baseline_median_skill"]), 4),
                "same_well_recovery_pct_random_cv": None if r["proxy_recovery_pct"] is None or not np.isfinite(r["proxy_recovery_pct"]) else round(float(r["proxy_recovery_pct"]), 1),
                "ml_skill_well_holdout": None if wl.empty else round(float(wl["ml_median_skill"].iloc[0]), 4),
                "ml_skill_drop_random_to_well_holdout": None if wl.empty else round(float(r["ml_median_skill"] - wl["ml_median_skill"].iloc[0]), 4),
            })
    results = {
        "status": "computed",
        "analysis": "hydrochemical proxy recovery kill-shot-first probe",
        "placement_status": "exploratory_not_registered_in_truth_table",
        "reading": "co-measured hydrochemical proxies recover reported ML skill; this is a candidate Free-Baseline Recovery branch, not the narrow autocorrelation ACR table.",
        "outputs": {
            "dataset_inventory": "data/processed/hydrochem_proxy_recovery/dataset_inventory.json",
            "fold_results": "data/processed/hydrochem_proxy_recovery/fold_results.csv",
            "target_summary": "data/processed/hydrochem_proxy_recovery/target_summary.csv",
            "results_json": "data/processed/hydrochem_proxy_recovery/hydrochem_proxy_recovery_results.json",
        },
        "headline_random_cv_ridge_vs_hgb": random_headline,
        "same_well_persistence_contrast": same_well_contrast,
        "same_well_contrast_reading": "For each target: ml_skill_random_cv is the reported-style skill; same_well_persistence_skill_random_cv is a feature-free baseline that only reads a well's own other rows; ml_skill_well_holdout is ML skill when whole wells are held out. If same-well persistence recovers most of ML skill under random CV and ML skill collapses under well holdout, the reported score is same-well memory, not feature mapping.",
        "all_summaries": summary_records,
        "caveats": [
            "Argentina is a small-sample sanity check and should not carry headline claims.",
            "CGWB is large and nationally heterogeneous; random-CV recovery should be paired with state-holdout sensitivity.",
            "Hydrochemical proxy recovery is not spatial validation and should not be described as spatial leakage.",
            "This run does not update manuscript_stats.json by design.",
        ],
    }

    (OUT_DIR / "dataset_inventory.json").write_text(json.dumps(inventory_json, indent=2, ensure_ascii=False), encoding="utf-8")
    fold_df.to_csv(OUT_DIR / "fold_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "target_summary.csv", index=False)
    (OUT_DIR / "hydrochem_proxy_recovery_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def check_outputs() -> None:
    required = [
        OUT_DIR / "dataset_inventory.json",
        OUT_DIR / "fold_results.csv",
        OUT_DIR / "target_summary.csv",
        OUT_DIR / "hydrochem_proxy_recovery_results.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"FAIL: missing outputs: {missing}")
    inv = json.loads((OUT_DIR / "dataset_inventory.json").read_text(encoding="utf-8"))
    fold = pd.read_csv(OUT_DIR / "fold_results.csv")
    summ = pd.read_csv(OUT_DIR / "target_summary.csv")
    datasets = {d["dataset"] for d in inv["datasets"]}
    if len(datasets) != 4:
        raise SystemExit(f"FAIL: expected 4 datasets, got {sorted(datasets)}")
    if fold.empty:
        raise SystemExit("FAIL: fold_results.csv is empty")
    if summ.empty:
        raise SystemExit("FAIL: target_summary.csv is empty")
    for target in inv["targets"]:
        if target.get("screened_out"):
            if not target.get("reason"):
                raise SystemExit(f"FAIL: screened target lacks reason: {target}")
            continue
        if not target.get("usable_predictors"):
            raise SystemExit(f"FAIL: scored target lacks usable predictors: {target}")
        target_token = norm_token(target["target_col"])
        for pred in target.get("usable_predictors", []):
            if norm_token(pred) == target_token:
                raise SystemExit(f"FAIL: target leaked into predictors: {target['dataset']} {target['target']} {pred}")
    if not np.isfinite(summ["ml_median_skill"]).all():
        raise SystemExit("FAIL: non-finite ML skill in summary")
    if (summ["ml_median_skill"] < 0).any() or (summ["baseline_median_skill"] < 0).any():
        raise SystemExit("FAIL: negative floored skill in summary")
    print("CHECK PASS: hydrochemical proxy recovery outputs are present and internally consistent.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="recompute outputs, then validate them")
    ap.add_argument("--validate-only", action="store_true", help="validate existing outputs without recomputing")
    args = ap.parse_args()

    if args.validate_only:
        check_outputs()
        return
    compute()
    if args.check:
        check_outputs()


if __name__ == "__main__":
    main()
