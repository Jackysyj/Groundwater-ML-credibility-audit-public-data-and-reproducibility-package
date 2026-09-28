# Data dictionary

## Corpus tables

### `canonical_paper_audit.csv`

One row per DOI-unique paper. Boolean reporting fields record whether the paper reports the named design or comparator. `model_families_json` is a JSON list and is multi-label. This table contains automated extraction outputs; use the dedicated manual-audit tables for headline task denominators. The verbatim extraction prompt that produced these fields, including all category boundary rules, is released as `metadata/llm_extraction_prompt_v2.md`.

### `water_quality_double_blind_audit.csv`

Reviewer A and B independently coded task mappability and spatial-holdout reporting. `consensus_*` fields are the adjudicated values used for the 6/68 headline. No full-text quotation or adjudication narrative is released.

### `water_level_researcher_verified_audit.csv`

Final full-text-verified task mode and persistence fields for all 179 water-level papers. `final_core_denominator=yes` identifies local-autoregressive or explicit multi-horizon forecasts used in the 2/91 headline.

### `full_schema_manual_audit_99.csv`

Paired automated (`llm_*`) and manual (`manual_*`) values plus field-level agreement flags (`match_*`) for 99 DOI-unique papers and 15 audited fields.

## Result tables

- `acr_folds.csv`: paired per-fold fitted-model and baseline values/skills.
- `acr_unified_table.csv`: seven manuscript ACR rows with model skill, baseline skill, increment and recovery percentage.
- `method3_spatial_extended_panel_gpu.csv`: 6 targets x 5 learner families; `gap = random_median - spatial_median`.
- `same_fold_panel.csv`: per-unit persistence and learner scores under identical folds and information cutoffs.
- `rolling_panel.csv`: learner results by dataset and forecast horizon under chronological evaluation.
- `bootstrap_ci.csv`: percentile-bootstrap intervals for same-fold and rolling-origin summaries.
- `truth_table.json`: flat manuscript-facing mirror used by Figures 1 and 4 and by plot annotations. It is generated from analysis outputs, not edited by hand.
- `assam_arsenic_scale_sensitivity/`: fold-level and summary results for the prespecified 0.10°, 0.15° and 0.20° spatial grid widths. The input table is identified by checksum but is not redistributed.

## Missing values and score conventions

CSV empty fields are missing values. Regression skill uses R2 and is floored at zero only when it enters a recovery ratio. Classification skill uses ROC-AUC minus 0.5. Negative deployment-holdout R2 values are retained in fold tables. Recovery above 100% means that the baseline met or exceeded the fitted model on the specified evaluation units.
