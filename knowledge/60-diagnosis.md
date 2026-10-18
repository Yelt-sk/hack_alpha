# Reading the diagnosis

Every `run_experiment` returns a `diagnosis` list. Each line maps to a specific next action.

## "OVERFITTING: train X vs validation Y"
The model memorises the training folds. In order of what to try:
1. Reduce capacity: `num_leaves` 31 -> 15, or `max_depth` down.
2. Raise regularisation: `min_child_samples` up (20 -> 50), `reg_lambda` up.
3. Add stochasticity: `subsample: 0.8`, `colsample_bytree: 0.8`.
4. Lower `learning_rate` and `n_estimators` together.
5. Drop noisy or high-cardinality features, or raise `rare_threshold`.

A gap is not automatically bad. Boosting on tabular data often runs a visible train/validation gap at its optimum. Only act if validation actually improves when you close it.

## "UNDERFITTING: barely beats the dummy baseline"
The model fits neither the training nor the validation data. Either it lacks capacity, or the features carry no signal.
1. First check that the features are real: did `drop_columns` remove too much?
2. Raise capacity: more `n_estimators`, more `num_leaves`.
3. Add features - `ratio` terms especially.
4. If nothing moves the metric, conclude honestly that the data does not support the target. That is a legitimate finding, not a failure.

## "HIGH VARIANCE across folds"
Fold scores disagree. Small differences between experiments are meaningless here.
1. Raise `folds` to 10 for a more stable estimate.
2. Expect this on small or heavily imbalanced datasets.
3. Prefer a configuration with lower `std` over one with a marginally better mean.

## "inside fold noise"
The improvement is smaller than `2 * std`. Do not keep the change on this evidence, and do not build the next experiment on top of it. Try a different axis.

## "NEW BEST"
A real improvement, larger than fold noise. Keep it and build the next experiment from this recipe.

## A perfect score, too good to be true (roc_auc 1.0, accuracy 100%)
A perfect or near-perfect score is a red flag, not a success. `roc_auc` above 0.99 on a problem that should be hard is a leak until proven otherwise. The server marks such runs `tainted` and excludes them from the leaderboard's best. Go back to the quality report, find the column that is doing all the work with `inspect_run`, and ask whether it exists at prediction time.
