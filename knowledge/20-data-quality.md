# Data quality

## Target leakage is the first thing to check
A leak is a column that would not be known at prediction time. `load_dataset` flags columns with high out-of-sample single-feature power as `critical / target_leakage`.

Symptoms: a near-perfect score (roc_auc > 0.99) on a problem that should be hard, especially from one column.

The fix is always the same: add the column to `drop_columns`. Before you do, ask the question that settles it: **at the moment the model has to make this prediction, does this value exist yet?** `support_ticket_after_cancel` only exists because the customer already cancelled, so it cannot help predict cancellation.

A model with a leak is worse than no model - it will score 1.0 in validation and fail completely in production.

## Identifier columns
Columns unique on nearly every row (`customer_id`, `uuid`, row index) carry no generalisable signal. Tree models will happily split on them and memorise the training set. Always put them in `drop_columns`.

Note the difference from a leak: an ID *inflates training fit* but does not usually inflate honest cross-validation, since the IDs in the validation fold were never seen. It wastes capacity rather than faking a score.

## Missing values
Missingness is often informative - a blank income field may itself predict default. Two complementary moves:
- Impute so the model can run: `numeric.impute="median"` (robust) and `categorical.impute="most_frequent"`, or `"constant"` which creates an explicit `__MISSING__` level.
- Preserve the signal: add `{"op": "missing_count", "name": "n_missing"}`, which counts missing fields per row.

A column that is >50% missing is usually better dropped unless the missingness pattern itself is the signal.

## Outliers
Tree models (lightgbm, catboost, random_forest) are rank-based and essentially immune to outliers - do not bother winsorising for them. Linear models and k-NN are very sensitive. For those, set `numeric.winsorize: 0.01` to clip at the 1st/99th percentile, and always scale.

## Duplicate rows
Duplicates that land in different folds leak information across the split and inflate cross-validation. If the quality report flags many duplicates, treat the CV score with suspicion.

## Class imbalance
Below ~10% minority:
1. Switch the metric to `roc_auc` or `pr_auc`.
2. Try `class_weight: "balanced"`, which re-weights the loss rather than resampling.
3. Expect high fold variance - use 10 folds.

`class_weight: "balanced"` usually improves recall and `f1` while slightly lowering raw accuracy. If your metric is `roc_auc` it often changes very little, because AUC already ignores the threshold.
