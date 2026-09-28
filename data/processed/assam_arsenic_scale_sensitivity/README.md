# Assam spatial scale sensitivity

This analysis evaluates fixed grid scales of 0.10, 0.15 and 0.20 degrees. The 0.15-degree grid is the main scale used for the Assam task. All scales and evaluable folds are reported using the existing 26 predictors, preprocessing, 500-round XGBoost configuration, ROC-AUC and 8-neighbour IDW baseline. The same dataset and model settings were used across widths.

The historical WSL script used CUDA. This saved run used Apple Silicon CPU; it is a supporting analysis tied to the recorded environment, rather than an exact reproduction of historical WSL output. Model estimates and baseline scores are reported from this run.

`input_manifest.json` gives the input path, checksum, variables and row count. `preset_scales.json` gives the fixed design. `split_summary.csv` and `fold_detail.csv` give training/test counts and every evaluated fold. `exceptions.json` records the single-class spatial-block exclusion at 0.10 degrees, so 15 spatial and 16 random folds are reported for that scale. Training and test index sets are disjoint under each partition.

`model_increment_median_auc` is the difference between the model and baseline medians. Its paired bootstrap interval resamples paired fold indices and takes that same difference of medians. `median_paired_fold_increment_auc` is a separately named statistic. Model and baseline median intervals also use 2,000 fold resamples. Fold training sets overlap, so these intervals are descriptive fold-resampling uncertainty, not independent-population inference. Ratios at or below the model no-skill floor are undefined. Spatial ratios close to that floor are not used as attribution claims.

## Run

Use the environment recorded in `run_metadata.json`. After opening the assam_bundle directory, supply the processed input table and a new output directory:

    python run_assam_scale_sensitivity.py --input /path/to/assam_as_points.csv --output /path/to/new_results

The input is the retained Assam processed table identified by checksum, derived from the public source documented in manuscript reference 22 and SI Table S16. The input table is not included in this archive and must be obtained or prepared separately from the documented source; it is identified by its SHA-256 checksum. Grid degrees are reported without a distance conversion. No variogram, minimum training-test distance or buffer estimate is claimed. The observed random-to-spatial direction is stable, while spatial estimates depend on scale.
