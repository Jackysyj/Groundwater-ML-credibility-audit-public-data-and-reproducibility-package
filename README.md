# Groundwater ML credibility audit: public data and reproducibility package

This release candidate supports the manuscript **Groundwater machine learning studies rarely report the validation and baselines their deployment claims require**. It contains the DOI-level literature-audit labels, manual-audit decisions without copyrighted full-text excerpts, fold/unit-level model and baseline scores, sensitivity summaries, figure inputs, and scripts needed to verify the principal reported quantities and redraw Figures 1-4.

## Release status

This is a deposition-ready **release candidate**, not the final archival record. Replace the citation placeholder after Zenodo or another repository assigns a DOI. Version-specific landing pages, cite-as records, licenses and retained-file fingerprints are verified for the French, Swiss and Tuscany exogenous-driver inputs.

## What is included

- `data/corpus/`: 532-paper DOI-level audit table; 99-paper full-schema audit; 154-paper double-blind water-quality audit; 179-paper researcher-verified water-level audit; manual-audit coverage ledger.
- `data/processed/`: released truth mirror, seven-row autocorrelation-recovery (ACR) inputs/results, random-versus-spatial fold summaries, water-level same-fold and rolling-origin summaries, robustness tables, and index-recovery outputs.
- `scripts/`: standalone verification, ACR recomputation, Figure 1-4 plotting, and the spatial, temporal, index-recovery, exogenous-driver and hydrochemical-boundary workflows. Model-training workflows require source data obtained separately.
- `figures/`: reference PNGs generated for the manuscript.
- `metadata/`: data dictionary, external-source/provenance ledger, verbatim LLM extraction prompt (`llm_extraction_prompt_v2.md`), build metadata and SHA-256 checksums.

The package does **not** contain publisher full texts, OCR/Markdown derivatives, Web of Science exports, JCR/impact-factor data, API credentials, raw third-party well observations, or reviewer evidence quotations.

## Quick start

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_package.py
```

Install `requirements-full.txt` only when rerunning model-training workflows from separately obtained source data. GPU workflows additionally require a CUDA-compatible PyTorch/XGBoost installation appropriate for the host system.

The integrity check validates checksums, the 532/154/179/99 row contracts, the 91-paper water-level denominator, the 2-paper persistence numerator, the seven ACR rows and all 30 positive model-target spatial gaps.

## Recompute ACR

```bash
python scripts/recompute_acr.py --check
```

ACR is calculated on matched evaluation units as baseline skill divided by fitted-model skill. Regression uses R2; classification uses ROC-AUC minus 0.5. The package reports raw model skill, raw baseline skill and model-minus-baseline increment beside the ratio.

## Reproduce the figures

```bash
python scripts/plot_fig1_corpus_audit.py --check
python scripts/plot_fig4_recovery_protocol.py --check
python scripts/plot_fig1_corpus_audit.py
python scripts/plot_fig2_spatial_collapse.py
python scripts/plot_fig3_horizon_decay.py
python scripts/plot_fig4_recovery_protocol.py
```

The plotting scripts write PNG files to `figures/`. Because this changes files covered by the original archive checksum, use `python scripts/verify_package.py --structural-only` after redrawing figures, or unpack a fresh archive before rerunning the full checksum test.

## Reproducibility levels

1. **Exact from this package:** corpus denominators, reporting rates, manual-audit agreement summaries, seven-row ACR, sensitivity summaries, figure panels and manuscript-facing derived values.
2. **Requires original third-party data:** retraining the spatial and groundwater-level models. Raw datasets are intentionally not duplicated; obtain them from `metadata/external_sources.csv` and use the methods and model settings reported in the manuscript/SI.
3. **Source data verified but not redistributed:** French, Swiss and Tuscany raw inputs must be obtained from the version-specific repositories listed in metadata/external_sources.csv.

## Corpus-table notes

- `canonical_paper_audit.csv` contains one row per normalized DOI. It reports automated extraction fields; final water-quality and water-level task decisions are supplied in their dedicated manual-audit tables.
- `water_quality_double_blind_audit.csv` preserves reviewer A, reviewer B and adjudicated labels, but excludes notes and copied article text.
- `water_level_researcher_verified_audit.csv` contains the final researcher-confirmed task and persistence labels. The Codex pre-review is not represented as human auditing.
- JCR quartiles and impact factors are excluded because they are licensed metadata rather than original project data.

## External data and licensing

No raw third-party dataset is redistributed. Source identifiers and known license constraints are listed in `metadata/external_sources.csv`. Users must obtain source data under the terms of the originating repositories. Project-authored code in this package is released under the MIT License; to the extent that the authors hold the relevant rights, project-authored tabular data and documentation are released under CC BY 4.0. These licenses do not override third-party rights.

## Citation

Until an archival DOI is assigned, cite the associated manuscript and this release as:

> Authors. Groundwater ML credibility audit: public data and reproducibility package. Version 0.1.0-rc1 (2026). DOI pending.

## Contact and issues

Author and contact metadata should be added to the archival repository record before deposit. Report data or code issues through the repository associated with the final DOI.
