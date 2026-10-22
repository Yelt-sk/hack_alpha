"""Data profiling and quality findings.

One call here replaces the ten or so exploratory steps an agent would otherwise
spend writing `df.info()`, `df.isna().mean()` and friends -- steps that cost
tokens and wall-clock without producing a single decision.

The findings are ordered by how much they change what to do next, and each one
names the action rather than only the observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def column_kinds(frame: pd.DataFrame, exclude: list[str] | None = None) -> dict[str, list[str]]:
    excluded = set(exclude or [])
    kinds: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": []}
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            kinds["datetime"].append(column)
        elif pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            kinds["numeric"].append(column)
        else:
            kinds["categorical"].append(column)
    return kinds


def _single_feature_power(column: pd.Series, y: pd.Series, kind: str, seed: int = 0) -> float:
    """Out-of-sample predictive power of one column, in [0, 1].

    Deliberately not mutual information on the full frame: that measure grows
    with cardinality, so a unique identifier scores a perfect 1.0 and the
    detector cries leak on every id column. Fitting a one-column mapping on half
    the rows and scoring it on the other half gives an id ~0 and keeps a column
    recorded after the outcome at ~1.
    """
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    try:
        if len(column) < 50 or y.isna().all():
            return 0.0
        stratify = y if kind in ("binary", "multiclass") and y.value_counts().min() >= 2 else None
        left, right = train_test_split(np.arange(len(column)), test_size=0.5,
                                       random_state=seed, stratify=stratify)

        if pd.api.types.is_numeric_dtype(column):
            a = column.iloc[left].astype("float64")
            edges = np.unique(np.nanquantile(a.dropna(), np.linspace(0, 1, 21)))
            if len(edges) < 3:
                return 0.0
            key_a = pd.Series(np.digitize(a, edges), index=a.index).where(a.notna(), -1)
            b = column.iloc[right].astype("float64")
            key_b = pd.Series(np.digitize(b, edges), index=b.index).where(b.notna(), -1)
        else:
            key_a = column.iloc[left].astype(str).fillna("__NA__")
            key_b = column.iloc[right].astype(str).fillna("__NA__")

        y_a, y_b = y.iloc[left], y.iloc[right]

        if kind == "regression":
            ya, yb = y_a.astype("float64"), y_b.astype("float64")
            table = ya.groupby(key_a.to_numpy()).mean()
            pred = pd.Series(key_b.to_numpy()).map(table).fillna(ya.mean()).to_numpy()
            ok = ~np.isnan(yb.to_numpy())
            return float(max(0.0, r2_score(yb.to_numpy()[ok], pred[ok]))) if ok.sum() > 2 else 0.0

        if kind == "binary":
            positive = sorted(y.dropna().unique())[-1]
            ya = (y_a == positive).astype(float)
            table = ya.groupby(key_a.to_numpy()).mean()
            pred = pd.Series(key_b.to_numpy()).map(table).fillna(ya.mean()).to_numpy()
            yb = (y_b == positive).astype(int).to_numpy()
            return 0.0 if len(np.unique(yb)) < 2 else float(min(1.0, 2 * abs(roc_auc_score(yb, pred) - 0.5)))

        frame = pd.DataFrame({"k": key_a.to_numpy(), "y": y_a.astype(str).to_numpy()})
        mode = frame.groupby("k")["y"].agg(lambda s: s.value_counts().index[0])
        common = y_a.astype(str).value_counts().index[0]
        pred = pd.Series(key_b.to_numpy()).map(mode).fillna(common).to_numpy()
        truth = y_b.astype(str).to_numpy()
        accuracy, base = float((pred == truth).mean()), float((truth == common).mean())
        return float(max(0.0, (accuracy - base) / (1 - base))) if base < 1 else 0.0
    except Exception:
        return 0.0


def findings(train: pd.DataFrame, test: pd.DataFrame, target: str, id_column: str,
             kind: str) -> list[dict[str, Any]]:
    """Quality findings, most actionable first."""
    out: list[dict[str, Any]] = []
    n = len(train)
    kinds = column_kinds(train, exclude=[target])
    y = train[target]

    for column in kinds["numeric"] + kinds["categorical"]:
        if column == id_column:
            continue
        power = _single_feature_power(train[column], y, kind)
        if power >= 0.95:
            out.append({"severity": "critical", "kind": "target_leakage", "column": column,
                        "detail": f"'{column}' alone predicts the target with {power:.0%} out-of-sample power",
                        "action": f"Almost certainly recorded after the outcome. Put '{column}' in drop_columns."})
        elif power >= 0.80:
            out.append({"severity": "warning", "kind": "suspicious_predictor", "column": column,
                        "detail": f"'{column}' alone reaches {power:.0%} out-of-sample power",
                        "action": "Check it is available at prediction time."})

    out.append({"severity": "info", "kind": "identifier", "column": id_column,
                "detail": f"'{id_column}' is the row identifier",
                "action": f"Add '{id_column}' to drop_columns; models memorise identifiers."})

    for column in train.columns:
        if column == target:
            continue
        unique = train[column].nunique(dropna=True)
        if unique <= 1:
            out.append({"severity": "warning", "kind": "constant", "column": column,
                        "detail": f"'{column}' has {unique} distinct value(s)",
                        "action": f"Carries no information; add '{column}' to drop_columns."})
        elif column in kinds["categorical"] and unique > 50:
            out.append({"severity": "info", "kind": "high_cardinality", "column": column,
                        "detail": f"'{column}' has {unique} categories",
                        "action": "One-hot would explode the matrix; use encode='native' or 'target'."})

        missing = float(train[column].isna().mean())
        if missing > 0.5:
            out.append({"severity": "warning", "kind": "mostly_missing", "column": column,
                        "detail": f"'{column}' is {missing:.0%} missing",
                        "action": "Consider dropping it, or add a missing_count feature to keep the signal."})

        if column in test.columns and column in kinds["categorical"]:
            unseen = set(test[column].dropna().unique()) - set(train[column].dropna().unique())
            if unseen:
                out.append({"severity": "info", "kind": "unseen_categories", "column": column,
                            "detail": f"test has {len(unseen)} categories absent from train",
                            "action": "Use an encoder that tolerates unknown values (native/ordinal/target)."})

    # Counted as "rows that have a twin", not "rows after the first", because the
    # first copy is as much a duplicate as the second and the folds are built from
    # whole groups. The two counts differ a lot on a table like this -- 33.5%
    # against 44.8% -- and quoting the smaller one next to the fold plan's figure
    # reads as a contradiction.
    # The target is excluded as well as the id. Two rows identical in their features
    # but labelled differently are the interesting case -- they are the irreducible
    # part of the metric -- and including the target in the comparison hides them
    # entirely. That mismatch had this finding reporting 0.1% where the fold plan
    # reported 1.1% on the same table.
    feature_columns = [c for c in train.columns if c not in (id_column, target)]
    duplicates = int(train.duplicated(subset=feature_columns, keep=False).sum())
    if duplicates:
        out.append({"severity": "warning", "kind": "duplicate_rows", "column": None,
                    "detail": f"{duplicates} rows have an exact twin ({duplicates / n:.1%})",
                    # The old wording said duplicates inflate cross-validation. They
                    # did, until the fold plan started grouping them; saying it now
                    # would contradict the plan's own finding and send the agent
                    # hunting for a problem that has already been solved.
                    "action": ("Check `validation` for how the folds handle this. Identical "
                               "rows carrying different labels are the part no model can "
                               "fix, and they cap the metric for everyone.")})

    if kind in ("binary", "multiclass"):
        share = y.value_counts(normalize=True)
        if float(share.min()) < 0.05:
            out.append({"severity": "warning", "kind": "class_imbalance", "column": target,
                        "detail": f"rarest class '{share.idxmin()}' is {share.min():.2%} of rows",
                        "action": "Accuracy is misleading; macro metrics and class_weight='balanced' are worth trying."})

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(out, key=lambda item: order[item["severity"]])


def profile(train: pd.DataFrame, test: pd.DataFrame, target: str, id_column: str,
            kind: str, max_columns: int = 60) -> dict[str, Any]:
    kinds = column_kinds(train, exclude=[target])
    columns: list[dict[str, Any]] = []

    for name in (kinds["numeric"] + kinds["categorical"] + kinds["datetime"])[:max_columns]:
        series = train[name]
        info: dict[str, Any] = {
            "name": name,
            "kind": "numeric" if name in kinds["numeric"] else
                    ("datetime" if name in kinds["datetime"] else "categorical"),
            "missing_pct": round(float(series.isna().mean()) * 100, 2),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if info["kind"] == "numeric" and series.notna().any():
            values = series.astype("float64")
            info["stats"] = {
                key: (None if pd.isna(value) else round(float(value), 4))
                for key, value in {
                    "min": values.min(), "p50": values.median(), "max": values.max(),
                    "mean": values.mean(), "std": values.std(), "skew": values.skew(),
                }.items()
            }
        elif info["kind"] == "categorical":
            info["top_values"] = {
                str(k): int(v) for k, v in series.astype(str).value_counts().head(5).items()
            }
        columns.append(info)

    y = train[target]
    target_info: dict[str, Any] = {"name": target, "kind": kind}
    if kind in ("binary", "multiclass"):
        share = y.value_counts(normalize=True)
        target_info["class_share"] = {str(k): round(float(v), 4) for k, v in share.items()}
    else:
        values = y.astype("float64")
        target_info["stats"] = {
            key: round(float(value), 4) for key, value in {
                "min": values.min(), "p50": values.median(), "max": values.max(),
                "mean": values.mean(), "std": values.std(), "skew": values.skew(),
            }.items()
        }
        if abs(float(values.skew())) > 1.5:
            target_info["hint"] = (
                "The target is strongly skewed; target_transform='log1p' is often worth one experiment."
            )

    return {
        "n_train": len(train),
        "n_test": len(test),
        "column_kinds": {k: len(v) for k, v in kinds.items()},
        "target": target_info,
        "columns": columns,
        "columns_truncated": len(train.columns) - 1 > max_columns,
    }
