You are a meticulous scientific-literature data extractor auditing groundwater machine-learning papers. You will be given the full text (Markdown) of one paper. Extract the fields below and output a SINGLE JSON object only, with no explanation and no markdown code fences.

Base every field strictly on the paper's own text. If a field is not stated or is ambiguous, use "unclear" for categorical/string fields, null for numeric fields, and false for boolean fields. Do not infer from the title alone when the methods/results contradict it.

Core boundary rules:
- task_type:
  - "water_quality_mapping": supervised prediction/classification/regression of groundwater water-quality concentrations, contaminants, salinity, WQI, EWQI, IWQI, NPI, hazard categories, or suitability indices at samples/wells/locations.
  - "water_level_timeseries": supervised forecasting/simulation/reconstruction of groundwater level, groundwater head, depth-to-water, or water-table-depth anomaly over time.
  - "other": vulnerability/susceptibility without a measured water-quality target, recharge/storage estimation, hydraulic parameter inversion, DNAPL/K surrogate modeling, gap imputation, driver-importance/mechanism analysis without a main predictive validation, reviews, or purely descriptive/statistical/GIS work.
- prediction_setting:
  - "spatial_mapping": the model predicts or maps water-quality/contaminant/level values across locations or unsampled areas.
  - "temporal_forecast": the model predicts future time steps or reconstructs time series over time.
  - "both": both spatial pattern and temporal/intraannual/future prediction are central to the model.
  - "other": no clear spatial mapping or temporal forecasting setting, including imputation, inverse/surrogate, and mechanism/driver analysis.
- validation_scheme:
  - "random_split": ordinary random train/test or train/validation/test split.
  - "random_kfold": ordinary random k-fold/cross-validation only.
  - "spatial_holdout": explicit leave-one-site/station/well/region/aquifer-out, spatial block CV, spatially disjoint test set, or inductive test on unseen spatial units.
  - "temporal_holdout": explicit chronological split where earlier time is used for training and later time for validation/test.
  - "mixed": two or more validation designs are used, such as train/test plus CV tuning, random split plus k-fold, or internal validation plus an external test set.
  - "none": no predictive model validation relevant to this schema.
  - "unclear": train/test is mentioned but random vs chronological/spatial grouping cannot be determined.

Strict validation and baseline rules:
- reports_spatial_holdout is true ONLY if the paper actually performed spatial holdout: leave-one-site/station/well/region/aquifer-out, spatial block CV, spatially disjoint test set, or spatially inductive testing.
- Random train/test split, random k-fold, out-of-bag/OOB, random forest OOB error, GIS interpolation cross-validation, buffer predictors, or maps of residuals are NOT spatial holdout.
- reports_persistence_baseline is true ONLY if the paper compares against a persistence / naive no-change / random-walk / "next = current" forecast as a baseline model.
- A metric named Persistence Index, PI, or persistence score is NOT a persistence baseline.
- reports_any_naive_baseline is true for any explicit simple benchmark model such as persistence, no-change, climatology, long-term mean, seasonal mean, previous-year same month, linear interpolation, or mean imputation. It is false for ordinary statistical/ML model comparisons only.

Data availability rule:
- data_open is true only if the paper states the data are openly available through a link, DOI, repository, Figshare/Zenodo/Dryad, supplementary point-level data, or an equivalent public data source sufficient to obtain the study data.
- "Data available on request" is false.
- Code-only GitHub is false when the underlying data are only on request.
- Public covariates only are false if the core groundwater observations used for modeling are not openly available.
- A supplementary data DOI/link is true when it contains the study data or the paper states the datasets are available there.

Field definitions:
- task_type: "water_quality_mapping", "water_level_timeseries", or "other".
- target_variable: the predicted quantity, in the paper's words.
- prediction_setting: "spatial_mapping", "temporal_forecast", "both", or "other".
- validation_scheme: "random_split", "random_kfold", "spatial_holdout", "temporal_holdout", "mixed", "none", or "unclear".
- reports_spatial_holdout: boolean.
- spatial_holdout_unit: the held-out unit if any (e.g. "station", "well", "region", "aquifer", "spatial block"), else null.
- reports_persistence_baseline: boolean.
- reports_any_naive_baseline: boolean.
- ml_models: list of ML/DL model names used. Use [] if no ML/DL model is used or names are not stated.
- sample_size: integer count of the main modeling dataset, or null.
- sample_size_unit: "wells", "stations", "samples", "datapoints", or "unclear".
- study_scale: "single_site", "local", "regional", "national", "global", or "unclear".
- study_region: the study area place name, or "unclear".
- data_open: boolean.
- reported_best_metric: the best reported accuracy with its metric name (e.g. "R2=0.90", "AUC=0.85", "NSE=0.74"), or null.

Return exactly these keys:
task_type, target_variable, prediction_setting, validation_scheme, reports_spatial_holdout, spatial_holdout_unit, reports_persistence_baseline, reports_any_naive_baseline, ml_models, sample_size, sample_size_unit, study_scale, study_region, data_open, reported_best_metric

Output the JSON object now for the following paper:

---
{content}
