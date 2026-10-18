# Recipe templates

Copy, adapt, run. Each is a complete valid recipe.

## Baseline floor (always run first)
```json
{"name": "baseline_dummy", "notes": "Floor: what a constant prediction scores.",
 "model": {"type": "dummy"}}
```

## Reference model (run second)
```json
{"name": "lgbm_default", "notes": "Untuned LightGBM reference point.",
 "drop_columns": ["<id columns>", "<leak columns>"],
 "model": {"type": "lightgbm"}}
```

## Mixed types with dates and missing values
```json
{"name": "lgbm_engineered",
 "notes": "Expanded signup_date, added a usage ratio and a missingness counter.",
 "drop_columns": ["customer_id", "support_ticket_after_cancel"],
 "features": [
   {"op": "datetime_parts", "name": "signup", "a": "signup_date"},
   {"op": "ratio", "name": "charge_per_month", "a": "total_charges", "b": "tenure_months"},
   {"op": "missing_count", "name": "n_missing"}],
 "numeric": {"impute": "median", "scale": "none"},
 "categorical": {"impute": "most_frequent", "encode": "native", "rare_threshold": 0.01},
 "model": {"type": "lightgbm", "params": {"learning_rate": 0.05, "num_leaves": 31}}}
```

## Heavy class imbalance
```json
{"name": "lgbm_balanced", "notes": "Minority class under 5%; weighting the loss and scoring on PR AUC.",
 "class_weight": "balanced",
 "metric": "pr_auc",
 "validation": {"scheme": "stratified_kfold", "folds": 10, "seed": 42},
 "model": {"type": "lightgbm"}}
```

## Linear reference (is the problem actually linear?)
```json
{"name": "logreg_reference", "notes": "If this matches the booster, prefer the simpler model.",
 "numeric": {"impute": "median", "scale": "standard"},
 "categorical": {"impute": "most_frequent", "encode": "onehot", "rare_threshold": 0.02},
 "model": {"type": "logreg", "params": {"C": 1.0}}}
```

## Many high-cardinality categoricals
```json
{"name": "catboost_highcard", "notes": "Several columns with 50+ levels; CatBoost handles these best.",
 "categorical": {"impute": "constant", "encode": "native", "rare_threshold": 0.005},
 "model": {"type": "catboost", "params": {"iterations": 500, "depth": 6}}}
```

## Overfitting - regularised variant
```json
{"name": "lgbm_regularised", "notes": "Previous run showed a 30% train/validation gap.",
 "model": {"type": "lightgbm",
   "params": {"num_leaves": 15, "min_child_samples": 50, "subsample": 0.8,
              "colsample_bytree": 0.8, "reg_lambda": 5.0, "learning_rate": 0.03, "n_estimators": 600}}}
```

## Grouped data (repeated measurements per entity)
```json
{"name": "lgbm_grouped", "notes": "Multiple rows per patient; random folds would leak.",
 "validation": {"scheme": "group_kfold", "folds": 5, "group_column": "patient_id"},
 "model": {"type": "lightgbm"}}
```

## Regression with skewed, outlier-heavy targets
```json
{"name": "lgbm_regression", "notes": "Outliers present; comparing RMSE against MAE.",
 "numeric": {"impute": "median", "scale": "none", "winsorize": 0.01},
 "metric": "mae",
 "validation": {"scheme": "kfold", "folds": 5, "seed": 42},
 "model": {"type": "lightgbm"}}
```
