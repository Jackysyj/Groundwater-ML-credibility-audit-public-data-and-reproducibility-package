#!/usr/bin/env python3
"""Validate the standalone public data package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def verify_hashes() -> None:
    manifest = ROOT / "metadata" / "SHA256SUMS.txt"
    require(manifest.exists(), "metadata/SHA256SUMS.txt is missing")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        require(path.is_file(), f"manifest file missing: {rel}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"checksum mismatch: {rel}")


def structural_checks() -> None:
    corpus = csv_rows("data/corpus/canonical_paper_audit.csv")
    dois = [row["doi"] for row in corpus]
    require(len(corpus) == 532, "canonical corpus must contain 532 rows")
    require(len(set(dois)) == 532, "canonical corpus DOI values must be unique")

    wq = csv_rows("data/corpus/water_quality_double_blind_audit.csv")
    wl = csv_rows("data/corpus/water_level_researcher_verified_audit.csv")
    full = csv_rows("data/corpus/full_schema_manual_audit_99.csv")
    require(len(wq) == 154, "water-quality double-blind audit must contain 154 rows")
    require(len(wl) == 179, "water-level researcher-verified audit must contain 179 rows")
    require(len(full) == 99, "full-schema manual audit must contain 99 DOI-unique rows")
    kunz = next(row for row in full if row["doi"].lower() == "10.5194/hess-29-3405-2025")
    require(
        kunz["title"] == (
            "Towards a global time series model for seasonal groundwater level predictions in Germany"
        ),
        "Kunz bibliographic title correction is missing",
    )
    require(sum(row["final_core_denominator"] == "yes" for row in wl) == 91,
            "water-level core denominator must be 91")
    require(sum(row["final_persistence_reported"] == "yes" for row in wl) == 2,
            "water-level persistence numerator must be 2")

    acr = csv_rows("data/processed/acr_unified_table.csv")
    panel = csv_rows("data/processed/method3_spatial_extended_panel_gpu.csv")
    require(len(acr) == 7, "ACR table must contain seven rows")
    require(len(panel) == 30, "spatial extended panel must contain 30 model-target rows")
    require(all(float(row["gap"]) > 0 for row in panel), "all spatial panel gaps must be positive")

    truth = json.loads((ROOT / "data" / "processed" / "truth_table.json").read_text())
    require(truth.get("corpus_n") == 532, "truth mirror corpus denominator drift")
    require(truth.get("wl_taskmode_core_n") == 91, "truth mirror water-level denominator drift")
    require(truth.get("wl_taskmode_core_persistence_n") == 2,
            "truth mirror water-level numerator drift")
    require(truth.get("double_blind_adjudicated_genuine_n") == 68,
            "truth mirror water-quality denominator drift")
    require(truth.get("double_blind_adjudicated_genuine_spatial_holdout_n") == 6,
            "truth mirror water-quality numerator drift")

    forbidden = ["full_text_evidence", "evidence_1_quote", "markdown_record", "if_2024", "jcr"]
    corpus_headers = set(corpus[0])
    wl_headers = set(wl[0])
    require(not corpus_headers.intersection(forbidden), "restricted columns leaked into corpus release")
    require(not wl_headers.intersection(forbidden), "full-text evidence leaked into water-level release")

    provenance = json.loads((ROOT / "metadata/dataset_provenance_registry.json").read_text())
    expected_provenance = {
        "french_piezoforecast": "10.5281/zenodo.7193812",
        "swiss_groundwater_database": "10.5281/zenodo.14260400",
        "tuscany_groundwater_drought": "10.5281/zenodo.17491496",
    }
    require(set(provenance["datasets"]) == set(expected_provenance),
            "provenance registry dataset keys drifted")
    for key, doi in expected_provenance.items():
        record = provenance["datasets"][key]
        require(record["status"] == "verified", f"provenance is not verified: {key}")
        require(record["version_doi"].lower() == doi, f"version DOI drift: {key}")

    sources = {row["dataset"]: row for row in csv_rows("metadata/external_sources.csv")}
    for dataset in ("FrenchPiezo", "Swiss Groundwater Database", "Tuscany groundwater drought"):
        require(sources[dataset]["provenance_status"] == "verified",
                f"external-source ledger is not verified: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-only", action="store_true",
                        help="skip checksums after intentionally regenerating figures")
    args = parser.parse_args()
    if not args.structural_only:
        verify_hashes()
    structural_checks()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "recompute_acr.py"), "--check"],
        cwd=ROOT,
    )
    require(result.returncode == 0, "standalone ACR check failed")
    print("PASS: public package structure, denominators, result tables and integrity checks are valid.")


if __name__ == "__main__":
    main()
