"""Compiling a recipe into a leak-safe pipeline, and scoring it on shared folds.

Everything that learns from data -- imputation statistics, rare-category lists,
winsorisation bounds, target encodings -- lives *inside* the ``Pipeline``, so it
is refitted on each training fold. There is no recipe field that would let the
agent fit a transform on the full target and then validate against it.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)

from agent_recipe_schema import MODELS, N_JOBS

EPS = 1e-9


class CompileError(Exception):
    def __init__(self, problem: str, hint: str = ""):
        super().__init__(problem)
        self.problem, self.hint = problem, hint


class _Named:
    """sklearn enables pandas output only for transformers that can name columns."""

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(list(input_features), dtype=object)
        return np.asarray(list(getattr(self, "columns_", [])), dtype=object)


class ColumnDropper(_Named, BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str] | None = None):
        self.columns = columns  # stored verbatim; sklearn's clone() requires it

    def fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if c not in (self.columns or [])]
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in (self.columns or []) if c in X.columns])


class FeatureEngineer(_Named, BaseEstimator, TransformerMixin):
    """Row-wise engineered columns.

    Every operation is stateless, so none of them can leak across a fold boundary
    by construction. That is the reason the catalogue is short.
    """

    def __init__(self, features: list[dict] | None = None):
        self.features = features

    def fit(self, X, y=None):
        self.columns_ = list(self.transform(X.head(5)).columns)
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features or []:
            op, name = feature["op"], feature["name"]
            try:
                if op == "missing_count":
                    X[name] = X.isna().sum(axis=1).astype("float64")
                elif op == "row_stat":
                    block = X[feature["columns"]].astype("float64")
                    X[name] = getattr(block, feature["stat"])(axis=1)
                elif op == "log1p":
                    X[name] = np.log1p(X[feature["a"]].astype("float64").clip(lower=0))
                else:
                    a = X[feature["a"]].astype("float64")
                    b = X[feature["b"]].astype("float64")
                    X[name] = {
                        "ratio": lambda: a / (b + EPS),
                        "product": lambda: a * b,
                        "diff": lambda: a - b,
                        "sum": lambda: a + b,
                    }[op]()
            except KeyError as err:
                raise CompileError(f"feature '{name}' ({op}) references a missing column {err}",
                                   "Check profile_data for the exact column names.") from None
            except Exception as err:
                raise CompileError(f"feature '{name}' ({op}) failed: {err}",
                                   f"'{op}' needs numeric inputs.") from None
        return X


class Winsorizer(_Named, BaseEstimator, TransformerMixin):
    """Clip numeric tails at bounds learned on the training fold only."""

    def __init__(self, q: float = 0.01):
        self.q = q

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.lo_, self.hi_ = frame.quantile(self.q), frame.quantile(1 - self.q)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(lower=self.lo_, upper=self.hi_, axis=1)


class RareGrouper(_Named, BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.keep_ = {
            c: set(frame[c].astype(str).value_counts(normalize=True).loc[lambda s: s >= self.threshold].index)
            for c in frame.columns
        }
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for column in frame.columns:
            as_str = frame[column].astype(str)
            frame[column] = as_str.where(as_str.isin(self.keep_.get(column, set())), "__RARE__")
        return frame


class FrequencyEncoder(_Named, BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.freq_ = {c: frame[c].astype(str).value_counts(normalize=True).to_dict() for c in frame.columns}
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        return pd.DataFrame(
            {c: frame[c].astype(str).map(self.freq_.get(c, {})).fillna(0.0).astype("float64")
             for c in frame.columns},
            index=frame.index,
        )


class ToCategory(_Named, BaseEstimator, TransformerMixin):
    """Cast ordinal codes to pandas ``category`` so boosters treat them natively."""

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for column in frame.columns:
            frame[column] = frame[column].fillna(-1).astype("int64").astype("category")
        return frame


class SanitizeNames(_Named, BaseEstimator, TransformerMixin):
    """LightGBM rejects feature names containing JSON-special characters."""

    def fit(self, X, y=None):
        seen: dict[str, int] = {}
        names: list[str] = []
        for column in pd.DataFrame(X).columns:
            clean = re.sub(r"[^0-9a-zA-Z_]+", "_", str(column)).strip("_") or "f"
            if clean in seen:
                seen[clean] += 1
                clean = f"{clean}_{seen[clean]}"
            else:
                seen[clean] = 0
            names.append(clean)
        self.names_ = self.columns_ = names
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        frame.columns = self.names_
        return frame


# ---------------------------------------------------------------------------
# Estimators that need an adapter
# ---------------------------------------------------------------------------


class _CatBoostBase(BaseEstimator):
    """CatBoost wants ``cat_features`` by name and refuses pandas ``category`` dtype."""

    _regression = False

    def __init__(self, params: dict | None = None):
        self.params = params

    def _prep(self, X):
        frame = pd.DataFrame(X).copy()
        for column in self.cat_features_:
            frame[column] = frame[column].astype(str)
        return frame

    def fit(self, X, y=None, **kwargs):
        from catboost import CatBoostClassifier, CatBoostRegressor

        frame = pd.DataFrame(X)
        self.cat_features_ = [c for c in frame.columns if str(frame[c].dtype) == "category"]
        self.n_features_in_ = frame.shape[1]
        cls = CatBoostRegressor if self._regression else CatBoostClassifier
        self.model_ = cls(cat_features=self.cat_features_, **(self.params or {}))
        self.model_.fit(self._prep(frame), y)
        if not self._regression:
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(self._prep(X))).ravel()

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


class CatBoostClf(ClassifierMixin, _CatBoostBase):
    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))


class CatBoostReg(RegressorMixin, _CatBoostBase):
    _regression = True


class _FlamlBase(BaseEstimator):
    """FLAML AutoML behind a plain sklearn surface.

    Exposed as a "model" in the recipe vocabulary on purpose: the agent decides
    *that* a search is worth the time and how much of it to spend, and a proven
    AutoML does the searching. Deciding is the part a language model is good at.
    """

    _task = "regression"

    def __init__(self, params: dict | None = None, time_budget_s: int = 300):
        self.params = params
        self.time_budget_s = time_budget_s

    def fit(self, X, y=None, **kwargs):
        from flaml import AutoML

        settings = {
            "task": self._task,
            "time_budget": int(self.time_budget_s),
            "verbose": 0,
            "n_jobs": N_JOBS,
            "early_stop": True,
            **(self.params or {}),
        }
        self.model_ = AutoML()
        self.model_.fit(X_train=pd.DataFrame(X), y_train=np.asarray(y), **settings)
        if self._task != "regression":
            self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(pd.DataFrame(X))).ravel()


class FlamlClf(ClassifierMixin, _FlamlBase):
    _task = "classification"

    def predict_proba(self, X):
        return self.model_.predict_proba(pd.DataFrame(X))


class FlamlReg(RegressorMixin, _FlamlBase):
    _task = "regression"


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_estimator(recipe: dict, kind: str, time_budget_s: float | None = None):
    model_type = recipe["model"]["type"]
    params = {**MODELS[model_type]["defaults"], **(recipe["model"].get("params") or {})}
    classification = kind != "regression"
    balanced = recipe["class_weight"] == "balanced" and classification

    if model_type == "dummy":
        from sklearn.dummy import DummyClassifier, DummyRegressor
        return DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="median")

    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)

    if model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(class_weight="balanced" if balanced else None, **params)

    if model_type in ("random_forest", "extra_trees"):
        from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                                      RandomForestClassifier, RandomForestRegressor)
        table = {
            ("random_forest", True): RandomForestClassifier, ("random_forest", False): RandomForestRegressor,
            ("extra_trees", True): ExtraTreesClassifier, ("extra_trees", False): ExtraTreesRegressor,
        }
        if balanced:
            params["class_weight"] = "balanced"
        return table[(model_type, classification)](random_state=0, **params)

    if model_type == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        if balanced:
            params["class_weight"] = "balanced"
        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        return cls(random_state=0, **params)

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        if balanced:
            params["class_weight"] = "balanced"
        cls = LGBMClassifier if classification else LGBMRegressor
        return cls(random_state=0, **params)

    if model_type == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if classification else XGBRegressor
        return cls(random_state=0, **params)

    if model_type == "catboost":
        params.setdefault("random_seed", 0)
        if balanced:
            params["auto_class_weights"] = "Balanced"
        return (CatBoostClf if classification else CatBoostReg)(params=params)

    if model_type == "flaml":
        # time_budget_s is already per fold: the caller divided the experiment's
        # total by the fold count. Reading the recipe field directly here would
        # multiply it back up -- an agent asking for ten minutes would spend
        # fifty, which is most of an hour out of a three-hour run.
        budget = time_budget_s or 300
        cls = FlamlClf if classification else FlamlReg
        return cls(params=params, time_budget_s=int(budget))

    raise CompileError(f"no builder for model '{model_type}'", f"Available: {sorted(MODELS)}")


def compile_pipeline(
    recipe: dict, X: pd.DataFrame, kind: str, time_budget_s: float | None = None
) -> tuple[Pipeline, list[str]]:
    """Turn a validated recipe into a pipeline that is refitted per fold."""
    warnings: list[str] = []
    numeric, categorical = recipe["numeric"], recipe["categorical"]
    model_type = recipe["model"]["type"]
    encode = categorical["encode"]

    if encode == "native" and not MODELS[model_type]["native_cat"]:
        encode = "ordinal"

    probe = FeatureEngineer(recipe["features"]).fit_transform(
        ColumnDropper(recipe["drop_columns"]).fit_transform(X.head(200))
    )
    datetimes = [c for c in probe.columns if pd.api.types.is_datetime64_any_dtype(probe[c])]
    if datetimes:
        warnings.append(f"dropping datetime column(s) {datetimes}: a model cannot consume them directly")

    numeric_steps: list[tuple[str, Any]] = []
    if numeric["impute"] != "none":
        numeric_steps.append(("impute", SimpleImputer(
            strategy=numeric["impute"], fill_value=0 if numeric["impute"] == "constant" else None)))
    if numeric.get("winsorize"):
        numeric_steps.append(("winsorize", Winsorizer(numeric["winsorize"])))
    if numeric["scale"] != "none":
        numeric_steps.append(("scale", {"standard": StandardScaler, "robust": RobustScaler,
                                        "minmax": MinMaxScaler}[numeric["scale"]]()))

    categorical_steps: list[tuple[str, Any]] = []
    if categorical["impute"] != "none":
        categorical_steps.append(("impute", SimpleImputer(
            strategy=categorical["impute"], fill_value="__MISSING__")))
    if categorical.get("rare_threshold"):
        categorical_steps.append(("rare", RareGrouper(categorical["rare_threshold"])))
    if encode == "onehot":
        categorical_steps.append(("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist", sparse_output=False, min_frequency=5)))
    elif encode == "target":
        target_type = {"regression": "continuous", "binary": "binary", "multiclass": "multiclass"}[kind]
        categorical_steps.append(("encode", TargetEncoder(target_type=target_type, random_state=0)))
    elif encode == "frequency":
        categorical_steps.append(("encode", FrequencyEncoder()))
    else:
        categorical_steps.append(("encode", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)))
        if encode == "native":
            categorical_steps.append(("to_category", ToCategory()))

    preprocess = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps) if numeric_steps else "passthrough",
             make_column_selector(dtype_include=["number", "bool"])),
            ("cat", Pipeline(categorical_steps),
             make_column_selector(dtype_exclude=["number", "bool", "datetime"])),
        ],
        remainder="drop",
    ).set_output(transform="pandas")

    pipeline = Pipeline([
        ("drop", ColumnDropper(recipe["drop_columns"])),
        ("features", FeatureEngineer(recipe["features"])),
        ("prep", preprocess),
        ("names", SanitizeNames()),
        ("model", build_estimator(recipe, kind, time_budget_s)),
    ])
    return pipeline, warnings


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _forward(y: pd.Series, transform: str) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype="float64")) if transform == "log1p" else np.asarray(y)


def _inverse(values: np.ndarray, transform: str) -> np.ndarray:
    return np.expm1(values) if transform == "log1p" else values


def evaluate(
    recipe: dict,
    store,
    time_budget_s: float | None = None,
) -> dict[str, Any]:
    """Fit the recipe on every fold, returning out-of-fold predictions and timings.

    Returns OOF predictions aligned to the training frame, plus test predictions
    averaged over folds -- fold-averaged predictions are both a free ensemble and
    the reason a saved submission matches the score it was judged on.
    """
    from agent_recipe_schema import MODELS as _MODELS  # local import keeps the module import-light

    spec = store.spec
    kind, transform = spec.kind, recipe["target_transform"]
    X, y = store.X, store.y
    classification = spec.is_classification

    oof = np.empty(len(X), dtype=object if classification else "float64")
    oof_proba: np.ndarray | None = None
    test_predictions: list[np.ndarray] = []
    test_proba: list[np.ndarray] = []
    fold_seconds: list[float] = []
    warnings: list[str] = []

    # A recipe's own budget is the cost of the whole experiment, so the folds
    # share it. An explicit budget wins over the caller's share of the clock.
    declared = recipe.get("time_budget_s")
    total_budget = declared if declared else time_budget_s
    budget_per_fold = (total_budget / store.n_splits) if total_budget else None

    for fold, (train_idx, val_idx) in enumerate(store.splits()):
        started = time.perf_counter()
        pipeline, compile_warnings = compile_pipeline(recipe, X, kind, budget_per_fold)
        if fold == 0:
            warnings += compile_warnings

        pipeline.fit(X.iloc[train_idx], _forward(y.iloc[train_idx], transform))
        predictions = _inverse(np.asarray(pipeline.predict(X.iloc[val_idx])), transform)
        oof[val_idx] = predictions

        if classification and hasattr(pipeline[-1], "predict_proba"):
            proba = pipeline.predict_proba(X.iloc[val_idx])
            if oof_proba is None:
                oof_proba = np.zeros((len(X), proba.shape[1]), dtype="float64")
            oof_proba[val_idx] = proba
            test_proba.append(pipeline.predict_proba(store.test))

        test_predictions.append(_inverse(np.asarray(pipeline.predict(store.test)), transform))
        fold_seconds.append(time.perf_counter() - started)

    classes = list(getattr(pipeline[-1], "classes_", [])) or None
    if classification:
        stacked = np.vstack([np.asarray(p, dtype=object) for p in test_predictions])
        test_final = pd.DataFrame(stacked).mode(axis=0).iloc[0].to_numpy()
        if test_proba:
            # Averaging probabilities beats voting on labels when classes are rare.
            test_final = np.asarray(classes)[np.mean(test_proba, axis=0).argmax(axis=1)]
    else:
        test_final = np.mean(test_predictions, axis=0)

    return {
        "oof": oof,
        "oof_proba": oof_proba,
        "test_pred": test_final,
        "classes": classes,
        "fold_seconds": fold_seconds,
        "seconds": float(sum(fold_seconds)),
        "warnings": warnings,
    }
