# Workflow

## The loop you must follow
Work in this order. Each step exists because skipping it is the most common way agents waste their budget.

1. `load_dataset` - gets you a dataset_id, the task type and a quality report.
2. `profile_dataset` - read the columns and the issues. Do not model blind.
3. Fix the critical issues first (drop leaks, drop IDs). A leak makes every later number meaningless.
4. `run_experiment` with `model.type="dummy"` - the floor. Any model that does not clearly beat it is broken.
5. `run_experiment` with `model.type="lightgbm"` and defaults - the reference point.
6. Iterate: change **one thing at a time**, run, read the diagnosis, keep what helps.
7. `tune_hyperparams` only after the preprocessing is settled. Tuning a bad pipeline buys almost nothing.
8. `final_evaluate` once, at the end, on your single best run.
9. `export_solution` to produce the deliverable.

## One change per experiment
If you change the encoder and the model and three hyperparameters at once, a better score tells you nothing about which change caused it, and a worse score tells you nothing about which to undo. Change one axis, keep the `name` descriptive and put the reason in `notes`.

## What counts as a real improvement
Every run reports `std` across folds. A gain smaller than `2 * std` is noise. The server will tell you this explicitly in the diagnosis. Chasing noise is the single biggest waste of an iteration budget - when you see "inside fold noise", do not build on that change.

## When to stop
Stop and call `final_evaluate` when any of these is true:
- Three consecutive experiments failed to beat the best by more than `2 * std`.
- You have run out of iteration budget.
- The score is already at the practical ceiling for the data (e.g. the remaining error is irreducible noise).

Stopping early with an honest number beats grinding out an overfitted one.

## The holdout is not yours to explore
`final_evaluate` is limited to 3 calls per dataset and every call is logged. The holdout exists to give one unbiased estimate of the model you already chose. If you pick your model *by* holdout score, the holdout stops being unbiased and you have simply overfitted more slowly. Choose with cross-validation; confirm with the holdout.
