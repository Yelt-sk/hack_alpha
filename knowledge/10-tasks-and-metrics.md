# Tasks and metrics

## Choosing the metric
The server picks a sane default (`metric: "auto"`), but override it when the problem demands:

- **binary, balanced classes, you need ranking** -> `roc_auc` (default). Threshold-free, robust.
- **binary, heavy imbalance (<10% positives), positives are what matter** -> `pr_auc`. ROC AUC looks flattering when negatives dominate; PR AUC does not.
- **binary, you must commit to a hard yes/no** -> `f1`, or `balanced_accuracy` if both classes matter equally.
- **you need calibrated probabilities** (pricing, expected value, risk) -> `logloss`. AUC is indifferent to calibration; logloss is not.
- **multiclass, balanced** -> `accuracy`. **multiclass, imbalanced** -> `macro_f1` (default) weights every class equally.
- **regression, outliers matter** -> `rmse` (default, punishes large errors). **outliers are noise** -> `mae`. **relative error matters** -> `mape` (never with values near zero). **you want "fraction of variance explained"** -> `r2`.

## Never judge imbalanced classification by accuracy
At 2% positives, predicting "no" for everything scores 98% accuracy and is worthless. The dummy baseline run exposes this immediately - that is what it is for.

## Lower-is-better metrics
`rmse`, `mae`, `mape`, `logloss` improve as they go **down**. The leaderboard already accounts for direction; do not re-invert it yourself.

## Validation schemes
- `stratified_kfold` - default for classification; keeps class ratios stable across folds.
- `kfold` - default for regression.
- `group_kfold` - **mandatory** when rows share an entity (several visits per patient, several sessions per user). Without it the same entity lands in train and validation and the score is inflated. Set `validation.group_column`.
- `time_series` - **mandatory** when predicting the future from the past. Random folds let the model train on tomorrow to predict today, which cannot happen in production.

Picking the wrong scheme produces an optimistic score that collapses on the holdout. This is a correctness decision, not a tuning knob.

## Folds
5 is standard. Use 10 when the dataset is under ~2000 rows or when fold variance is high and you need to resolve small differences. More folds cost proportionally more time.
