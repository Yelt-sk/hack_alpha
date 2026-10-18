"""
Recipe -> sklearn Pipeline compiler.

Every transform that learns anything from data (imputation statistics, rare
category lists, winsorisation bounds, target encodings) lives *inside* the
Pipeline, so cross_validate refits it on each training fold. The agent cannot
leak statistics across folds even if it tries -- there is no recipe field that
would let it.
"""
from __future__ import annotations

import re
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

from .schema import MODEL_REGISTRY

EPS = 1e-9


class CompileError(Exception):
    """Raised with an agent-readable hint, never a raw stack trace."""

    def __init__(self, problem: str, hint: str):
        super().__init__(problem)
        self.problem, self.hint = problem, hint


# --------------------------------------------------------------------------
# Stateless, fold-safe transformers
# --------------------------------------------------------------------------



class _KeepsNames:
    """sklearn only enables pandas output for transformers that can name their columns."""

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(list(input_features), dtype=object)
        return np.asarray(list(getattr(self, "columns_", [])), dtype=object)


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str] | None = None):
        # sklearn's clone() requires constructor args to be stored verbatim.
        self.columns = columns

    @property
    def _cols(self) -> list[str]:
        return self.columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self._cols if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.array([c for c in (input_features or []) if c not in self._cols])


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-wise engineered columns. Stateless by design -- cannot leak."""

    def __init__(self, features: list[dict] | None = None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for f in (self.features or []):
            op, name = f["op"], f["name"]
            try:
                if op == "missing_count":
                    X[name] = X.isna().sum(axis=1).astype("float64")
                    continue
                a = X[f["a"]]
                if op == "log1p":
                    X[name] = np.log1p(a.astype("float64").clip(lower=0))
                elif op == "datetime_parts":
                    d = pd.to_datetime(a, errors="coerce")
                    X[f"{name}_year"] = d.dt.year.astype("float64")
                    X[f"{name}_month"] = d.dt.month.astype("float64")
                    X[f"{name}_day"] = d.dt.day.astype("float64")
                    X[f"{name}_dayofweek"] = d.dt.dayofweek.astype("float64")
                    X[f"{name}_hour"] = d.dt.hour.astype("float64")
                else:
                    b = X[f["b"]].astype("float64")
                    a = a.astype("float64")
                    X[name] = {"ratio": lambda: a / (b + EPS), "product": lambda: a * b,
                               "diff": lambda: a - b, "sum": lambda: a + b}[op]()
            except KeyError as e:
                raise CompileError(f"feature '{name}' ({op}) references missing column {e}",
                                   "Check profile_dataset for the exact column names.") from None
            except Exception as e:
                raise CompileError(f"feature '{name}' ({op}) failed: {e}",
                                   f"'{op}' needs numeric inputs (datetime_parts needs a date column).") from None
        return X


class DropDatetime(_KeepsNames, BaseEstimator, TransformerMixin):
    """Raw datetimes cannot go into a model; expand them with datetime_parts first."""

    def fit(self, X, y=None):
        self.cols_ = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
        self.columns_ = [c for c in X.columns if c not in self.cols_]
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols_ if c in X.columns])


class Winsorizer(_KeepsNames, BaseEstimator, TransformerMixin):
    """Clip numeric tails at bounds learned on the training fold only."""

    def __init__(self, q: float = 0.01):
        self.q = q

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        self.columns_ = list(Xd.columns)
        self.lo_ = Xd.quantile(self.q)
        self.hi_ = Xd.quantile(1 - self.q)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(lower=self.lo_, upper=self.hi_, axis=1)


class RareGrouper(_KeepsNames, BaseEstimator, TransformerMixin):
    """Fold categories below `threshold` frequency into __RARE__, learned per fold."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        self.columns_ = list(Xd.columns)
        self.keep_ = {c: set(Xd[c].astype(str).value_counts(normalize=True)
                             .loc[lambda s: s >= self.threshold].index) for c in Xd.columns}
        return self

    def transform(self, X):
        Xd = pd.DataFrame(X).copy()
        for c in Xd.columns:
            keep = self.keep_.get(c, set())
            Xd[c] = Xd[c].astype(str).where(Xd[c].astype(str).isin(keep), "__RARE__")
        return Xd


class FrequencyEncoder(_KeepsNames, BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        self.columns_ = list(Xd.columns)
        self.freq_ = {c: Xd[c].astype(str).value_counts(normalize=True).to_dict() for c in Xd.columns}
        return self

    def transform(self, X):
        Xd = pd.DataFrame(X)
        return pd.DataFrame(
            {c: Xd[c].astype(str).map(self.freq_.get(c, {})).fillna(0.0).astype("float64") for c in Xd.columns},
            index=Xd.index)


class ToCategory(_KeepsNames, BaseEstimator, TransformerMixin):
    """Cast ordinal codes to pandas `category` so LightGBM/CatBoost treat them natively."""

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        Xd = pd.DataFrame(X).copy()
        for c in Xd.columns:
            Xd[c] = Xd[c].fillna(-1).astype("int64").astype("category")
        return Xd


class SanitizeNames(_KeepsNames, BaseEstimator, TransformerMixin):
    """LightGBM rejects feature names containing JSON-special characters."""

    def fit(self, X, y=None):
        cols = list(pd.DataFrame(X).columns)
        seen: dict[str, int] = {}
        out = []
        for c in cols:
            s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_") or "f"
            if s in seen:
                seen[s] += 1
                s = f"{s}_{seen[s]}"
            else:
                seen[s] = 0
            out.append(s)
        self.names_ = out
        self.columns_ = out
        return self

    def transform(self, X):
        Xd = pd.DataFrame(X).copy()
        Xd.columns = self.names_
        return Xd


class _CatBoostBase(BaseEstimator):
    """CatBoost wants cat_features by name and refuses pandas `category` dtype.

    Split into classifier/regressor subclasses so sklearn's mixins supply the
    correct estimator tags -- scorers like roc_auc dispatch on them.
    """

    _task = "binary"

    def __init__(self, params: dict | None = None):
        self.params = params

    @staticmethod
    def _cat_cols(X):
        return [c for c in X.columns if str(X[c].dtype) == "category"]

    def _prep(self, X):
        X = pd.DataFrame(X).copy()
        for c in self._cat_cols(X):
            X[c] = X[c].astype(str)
        return X

    def fit(self, X, y=None, **kw):
        from catboost import CatBoostClassifier, CatBoostRegressor

        X = pd.DataFrame(X)
        self.cat_features_ = self._cat_cols(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        cls = CatBoostRegressor if self._task == "regression" else CatBoostClassifier
        self.model_ = cls(cat_features=self.cat_features_, **(self.params or {}))
        self.model_.fit(self._prep(X), y)
        if self._task != "regression":
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X)).ravel()

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


class CatBoostClassifierAdapter(ClassifierMixin, _CatBoostBase):
    _task = "classification"

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-15, 1.0))


class CatBoostRegressorAdapter(RegressorMixin, _CatBoostBase):
    _task = "regression"


# --------------------------------------------------------------------------
# Estimator construction
# --------------------------------------------------------------------------


def build_estimator(model_type: str, params: dict, task: str, class_weight: str):
    p = dict(MODEL_REGISTRY[model_type]["defaults"])
    p.update(params or {})
    balanced = class_weight == "balanced" and task != "regression"
    clf = task != "regression"

    if model_type == "dummy":
        from sklearn.dummy import DummyClassifier, DummyRegressor
        return DummyClassifier(strategy="prior") if clf else DummyRegressor(strategy="mean")

    if model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(class_weight="balanced" if balanced else None, **p)

    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**p)

    if model_type in ("random_forest", "extra_trees"):
        from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                                      RandomForestClassifier, RandomForestRegressor)
        cls = {("random_forest", True): RandomForestClassifier, ("random_forest", False): RandomForestRegressor,
               ("extra_trees", True): ExtraTreesClassifier, ("extra_trees", False): ExtraTreesRegressor}[(model_type, clf)]
        if clf and balanced:
            p["class_weight"] = "balanced"
        return cls(random_state=0, **p)

    if model_type == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        if clf and balanced:
            p["class_weight"] = "balanced"
        return cls(random_state=0, **p)

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if clf else LGBMRegressor
        if clf and balanced:
            p["class_weight"] = "balanced"
        return cls(random_state=0, **p)

    if model_type == "catboost":
        from .schema import DEFAULT_N_JOBS
        p.setdefault("random_seed", 0)
        p.setdefault("thread_count", DEFAULT_N_JOBS)
        if balanced:
            p["auto_class_weights"] = "Balanced"
        return (CatBoostRegressorAdapter if task == "regression" else CatBoostClassifierAdapter)(params=p)

    if model_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        return (KNeighborsClassifier if clf else KNeighborsRegressor)(**p)

    raise CompileError(f"model '{model_type}' has no builder", f"Available: {sorted(MODEL_REGISTRY)}")


# --------------------------------------------------------------------------
# The compiler
# --------------------------------------------------------------------------


def compile_pipeline(recipe: dict, X: pd.DataFrame, task: str) -> tuple[Pipeline, list[str]]:
    """Turn a validated recipe into a fitted-per-fold Pipeline. Returns (pipe, warnings)."""
    warnings: list[str] = []
    num_cfg, cat_cfg = recipe["numeric"], recipe["categorical"]
    model_type = recipe["model"]["type"]
    encode = cat_cfg["encode"]

    if encode == "native" and not MODEL_REGISTRY[model_type]["handles_native_cat"]:
        encode = "ordinal"

    # --- what the data looks like after drops + feature engineering --------
    probe = FeatureEngineer(recipe["features"]).fit_transform(
        ColumnDropper(recipe["drop_columns"]).fit_transform(X.head(200)))
    raw_dt = [c for c in probe.columns if pd.api.types.is_datetime64_any_dtype(probe[c])]
    if raw_dt:
        warnings.append(
            f"Dropping raw datetime column(s) {raw_dt}: a model cannot consume them directly. "
            f"Add a datetime_parts feature to keep the signal.")
    probe = probe.drop(columns=raw_dt)
    cat_cols = [c for c in probe.columns
                if not pd.api.types.is_numeric_dtype(probe[c]) and not pd.api.types.is_bool_dtype(probe[c])]

    if encode == "onehot" and cat_cols:
        width = int(sum(min(X[c].nunique(), 1000) for c in cat_cols if c in X.columns))
        worst = max(((c, int(X[c].nunique())) for c in cat_cols if c in X.columns),
                    key=lambda t: t[1], default=(None, 0))
        if worst[1] > cat_cfg["max_onehot_cardinality"]:
            raise CompileError(
                f"one-hot encoding '{worst[0]}' would create {worst[1]} columns, over the "
                f"max_onehot_cardinality limit of {cat_cfg['max_onehot_cardinality']} (total width ~{width})",
                "Use categorical.encode='native' with lightgbm/catboost, or 'target', or raise "
                "categorical.rare_threshold to collapse rare levels first.")

    # --- numeric branch ----------------------------------------------------
    num_steps: list[tuple[str, Any]] = []
    if num_cfg["impute"] != "none":
        num_steps.append(("impute", SimpleImputer(
            strategy=num_cfg["impute"], fill_value=0 if num_cfg["impute"] == "constant" else None)))
    if num_cfg.get("winsorize"):
        num_steps.append(("winsorize", Winsorizer(num_cfg["winsorize"])))
    if num_cfg["scale"] != "none":
        num_steps.append(("scale", {"standard": StandardScaler, "robust": RobustScaler,
                                    "minmax": MinMaxScaler}[num_cfg["scale"]]()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    # --- categorical branch ------------------------------------------------
    cat_steps: list[tuple[str, Any]] = []
    if cat_cfg["impute"] != "none":
        cat_steps.append(("impute", SimpleImputer(strategy=cat_cfg["impute"], fill_value="__MISSING__")))
    if cat_cfg.get("rare_threshold"):
        cat_steps.append(("rare", RareGrouper(cat_cfg["rare_threshold"])))
    if encode == "onehot":
        cat_steps.append(("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                  sparse_output=False, min_frequency=1)))
    elif encode == "target":
        tt = {"binary": "binary", "multiclass": "multiclass", "regression": "continuous"}[task]
        cat_steps.append(("encode", TargetEncoder(target_type=tt, random_state=0)))
    elif encode == "frequency":
        cat_steps.append(("encode", FrequencyEncoder()))
    else:  # native / ordinal
        cat_steps.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                   unknown_value=-1, encoded_missing_value=-1)))
        if encode == "native":
            cat_steps.append(("to_category", ToCategory()))
    cat_pipe = Pipeline(cat_steps)

    ct = ColumnTransformer(
        [("num", num_pipe, make_column_selector(dtype_include=["number", "bool"])),
         ("cat", cat_pipe, make_column_selector(dtype_exclude=["number", "bool", "datetime"]))],
        remainder="drop", verbose_feature_names_out=True,
    ).set_output(transform="pandas")

    pipe = Pipeline([
        ("drop", ColumnDropper(recipe["drop_columns"])),
        ("features", FeatureEngineer(recipe["features"])),
        ("drop_datetime", DropDatetime()),
        ("prep", ct),
        ("names", SanitizeNames()),
        ("model", build_estimator(model_type, recipe["model"]["params"], task, recipe["class_weight"])),
    ])
    return pipe, warnings
