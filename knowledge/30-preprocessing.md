# Preprocessing decisions

## Encoding categoricals - the decision rule
This is the choice with the largest effect on tabular results. Pick with this table:

| Situation | `categorical.encode` |
|---|---|
| Model is lightgbm or catboost | `native` - best default, no information loss, no matrix blow-up |
| Linear model (logreg, ridge) or knn | `onehot` - these models need orthogonal columns |
| Very high cardinality (>50 levels) and a non-native model | `target` |
| The *frequency* of a level is the signal (rare = suspicious) | `frequency` |
| Tree model, ordering is genuinely meaningful (small/medium/large) | `ordinal` |

Never use `ordinal` with a linear model: it invents a fake numeric ordering (`red`=1, `blue`=2, `green`=3) and the model will interpret the arithmetic literally.

## Target encoding is safe here, but know why
Target encoding replaces a category with the mean target for that category. Done naively it leaks the target and produces spectacular validation scores that evaporate on new data.

The server uses sklearn's `TargetEncoder`, which fits with an internal cross-fitting scheme, *inside* the pipeline, so it is refitted on each training fold. You cannot leak with it through this interface. It is still worth preferring `native` on lightgbm/catboost, which handles categoricals without the extra variance.

## Rare categories
`categorical.rare_threshold` (default 0.01) folds levels appearing in under 1% of rows into `__RARE__`. This reduces overfitting on noisy tails and keeps one-hot matrices narrow. Raise it to 0.05 when a column has a long tail of near-unique values.

## Scaling
Only matters for distance- and penalty-based models:
- **logreg, ridge, knn** -> `numeric.scale: "standard"` always. Unscaled, these are often dramatically worse, and the server warns you about it.
- **all tree models** -> scaling is a no-op. Leave `"none"`; it saves time and changes nothing.

Use `"robust"` instead of `"standard"` when the column has heavy outliers (it scales by the IQR).

## Feature engineering
Only stateless row-wise ops are available, because stateless transforms cannot leak across folds. That constraint is a feature, not a limitation.

What actually tends to help on tabular data:
- `ratio` - normalise a quantity by an exposure: debt/income, clicks/impressions, charges/tenure. Usually the single highest-value engineered feature.
- `datetime_parts` - a raw timestamp is useless to a model; day-of-week and month capture seasonality. **Raw datetime columns are dropped**, so you must expand any date you want to use.
- `missing_count` - cheap, and often surprisingly predictive.
- `log1p` - compresses a heavily right-skewed non-negative column. Helps linear models; does nothing for trees.
- `product`, `diff`, `sum` - explicit interactions. Gradient boosting finds many of these itself, so the payoff is smaller than people expect.

Add features in small batches and keep the ones that move the metric beyond noise.
