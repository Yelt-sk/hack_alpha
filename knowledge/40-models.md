# Choosing a model

## Start here
`lightgbm` with defaults is the right first real model for almost any tabular problem. It handles mixed types, missing values and categoricals natively, trains fast, and is rarely far from the best achievable score. Beat it before you reach for anything else.

## When to use what
- **lightgbm** - the default. Fast, strong, `encode="native"` handles categoricals without widening the matrix.
- **catboost** - usually wins when categoricals are many and high-cardinality. Slower to train. Its ordered target statistics handle category/target relationships better than LightGBM's splits.
- **hist_gbm** - sklearn's gradient boosting. Very close to LightGBM in quality; useful as an independent check that a result is not a LightGBM quirk.
- **random_forest / extra_trees** - hard to overfit, no tuning needed to get something reasonable. Rarely the winner on tabular data, but a good sanity check: if a tuned booster cannot beat an untuned forest, something is wrong with the boosting setup.
- **logreg / ridge** - fast, interpretable, well-calibrated. Always worth one run: if a linear model matches the booster, the relationship is essentially linear and you should prefer the simpler model.
- **knn** - a diagnostic. If it is competitive, the signal is local and geometric, which suggests distance-based features may help.
- **dummy** - not a model. The floor. Run it first, always.

## Reading the comparison
- **Linear ≈ boosting** -> the problem is linear. Stop adding capacity; add features instead.
- **Boosting >> linear** -> interactions and non-linearities matter. Tuning the booster is worth the budget.
- **Everything ≈ dummy** -> the features do not carry signal. More modelling will not help; the problem is the data. Say so rather than grinding.
- **Everything ≈ perfect** -> you have a leak. Go back to the quality report.

## Model-specific notes
**LightGBM** overfits small datasets with default settings. Under ~2000 rows, lower `num_leaves` to 15-31 and raise `min_child_samples`.

**CatBoost** is slow with high `iterations`. Explore at `iterations: 300`, then raise it for the final model with a correspondingly lower `learning_rate`.

**Logistic regression** needs `numeric.scale: "standard"` and `categorical.encode: "onehot"`. Without scaling the penalty is applied unevenly across features and the result is often much worse - the server warns you when you get this combination wrong.
