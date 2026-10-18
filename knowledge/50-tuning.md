# Hyperparameter tuning

## Tune last, not first
Preprocessing choices (dropping a leak, the right encoder, the right CV scheme) move the metric far more than hyperparameters. Tuning a pipeline you are still going to change is wasted budget. Settle the recipe, then tune.

Realistic expectation on tabular data: tuning a sensible LightGBM buys perhaps 1-3% over defaults. Finding and fixing a leak or a wrong CV scheme can buy 30%.

## What actually matters
For gradient boosting, in descending order of effect:
1. `learning_rate` x `n_estimators` - the fundamental trade-off. Lower rate with more trees is usually better but slower.
2. `num_leaves` (lightgbm) / `depth` (catboost) - capacity. The main overfitting knob.
3. `min_child_samples` - minimum rows per leaf. Raise it to regularise on small or noisy data.
4. `subsample` and `colsample_bytree` - stochastic regularisation. 0.8 is a reasonable starting point.
5. `reg_lambda` - L2 penalty. Matters most when leaves are many.

For linear models there is exactly one knob worth tuning: `C` (logreg) or `alpha` (ridge), on a log scale.

## Using tune_hyperparams
Pass a base recipe and a time budget. The server searches only the parameters registered for that model, with Optuna's TPE sampler, using the same cross-validation as `run_experiment` - so tuned scores are directly comparable to untuned ones.

Start with a budget of 60-120 seconds. If the best trial barely beats the base recipe, more budget will not rescue it; change the pipeline instead.

Narrow the space when you have a reason to: `search_space_overrides` lets you pin `n_estimators` and search only `learning_rate`, which spends the budget where it matters.

## Do not tune on the holdout
Optuna will happily optimise against any number you give it. The server only ever lets it see cross-validation scores. Tuning against the holdout would produce a model selected on the very data meant to measure it.

## Overfitting the validation set
Running 200 tuning trials and keeping the single best selects partly for luck - with enough trials, some configuration scores well by chance. Guard against it:
- Prefer a configuration whose fold `std` is low, not just whose mean is high.
- Treat a tuned gain smaller than `2 * std` as noise, exactly as with any other experiment.
- Keep the trial count modest (30-60) rather than exhaustive.
