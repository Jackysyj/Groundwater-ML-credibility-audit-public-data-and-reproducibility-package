### Assam arsenic spatial-scale sensitivity

The three grid widths were fixed before inspecting the results. The 0.15° grid is the original main scale. Each row reports the median ROC-AUC across spatial blocks and the matched random-k-fold reference, followed by the 2,000-resample fold bootstrap interval for the median. The IDW baseline uses eight nearest training wells and no fitted parameters.

| Grid width | Spatial blocks (*n*) | Random folds (*n*) | Random ML AUC | Spatial ML AUC (95% CI) | Random IDW AUC | Spatial IDW AUC (95% CI) | Random model increment | Random baseline recovery |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 0.10° | 15 | 16 | 0.899 | 0.595 (0.532 to 0.721) | 0.918 | 0.711 (0.525 to 0.762) | -0.019 | 104.7%
| 0.15° | 14 | 14 | 0.901 | 0.578 (0.527 to 0.664) | 0.917 | 0.699 (0.601 to 0.775) | -0.016 | 104.0%
| 0.20° | 10 | 10 | 0.905 | 0.633 (0.494 to 0.727) | 0.911 | 0.740 (0.592 to 0.836) | -0.006 | 101.6%

At all three scales, the random-regime fitted-model AUC exceeded the spatial-block AUC, while the IDW baseline met or exceeded the fitted-model skill in the random regime. The spatial model increment was negative in the fold medians at all three scales. These results support scale sensitivity within this Assam task, but grid width is not the minimum training-test distance and no variogram range or buffer distance was estimated. The analysis therefore does not establish that residual spatial dependence has been eliminated.
