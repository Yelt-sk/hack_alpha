"""Data understanding tools."""

from __future__ import annotations

from typing import Any

import agent_profiling as profiling
from agent_session import session
from agent_registry import tool


@tool
def profile_data(max_columns: int = 25) -> dict[str, Any]:
    """Describe the data and list its quality problems in one call.

    Returns shapes, per-column types, missing rates, cardinality, target
    distribution, and findings such as target leakage, identifier columns,
    constant columns, unseen test categories and class imbalance. Each finding
    names the action to take.

    Read this before your first experiment. Doing the same exploration by hand
    costs roughly ten steps and tells you no more.

    Args:
        max_columns: How many columns to describe individually. Raise it only if
            you need the rest; the per-column block dominates the reply, and on a
            wide table it can crowd out your own working context.
    """
    current = session()
    spec, store = current.spec, current.store

    # The profile is deterministic, so a second identical call returns nothing
    # new -- it only spends the context the agent still needs. Observed live: a
    # model re-profiled at step 4 and overflowed its window one step later.
    cached = getattr(current, "_profile_cache", None)
    if cached is not None and cached[0] == max_columns:
        return {
            "status": "unchanged",
            "note": "You already profiled this data; nothing about it has changed.",
            "findings": cached[1]["findings"],
            "hint": "Use inspect_columns for specific columns, or get_context for the run state.",
        }

    report = profiling.profile(
        store.train, store.test, spec.target_column, spec.id_column, spec.kind, max_columns
    )
    report["findings"] = profiling.findings(
        store.train, store.test, spec.target_column, spec.id_column, spec.kind
    )
    # The split diagnosis belongs here as well as in get_context. This is the tool
    # the agent reaches for first, and "your test set is a later period than
    # anything you can train on" is the single most consequential fact about a
    # task -- more so than any column's missing rate. Measured on this project's
    # regression task, not knowing it was worth twelve RMSE.
    report["validation"] = store.plan.summary()
    if store.plan.name == "forward_chain":
        report["findings"] = list(report.get("findings") or []) + [{
            "finding": "the test set lies beyond the training set in time",
            "detail": store.plan.rationale,
            "action": ("Favour models that generalise rather than models that fit the "
                       "training periods closely: a smoother target, less capacity, "
                       "features that describe the subject rather than the period. "
                       "Judge results fold by fold -- the spread between folds is how "
                       "hard each period was, not how noisy your model is."),
        }]
    elif store.plan.name == "grouped":
        report["findings"] = list(report.get("findings") or []) + [{
            "finding": f"{store.plan.diagnosis.get('duplicate_share', 0):.1%} of training "
                       f"rows are exact duplicates of another row",
            "detail": store.plan.rationale,
            "action": ("Scores will read lower than a random split would give; that is the "
                       "split being honest. Groups whose members carry different labels "
                       f"({store.plan.diagnosis.get('conflicting_groups', 0)} of them) cap "
                       "the metric for everyone."),
        }]
    report["dictionary_available"] = store.dictionary is not None
    if store.dtype_fixes:
        report["dtype_reconciliation"] = store.dtype_fixes
    if report.get("columns_truncated"):
        report["note"] = (
            f"Only {max_columns} of {len(store.X.columns)} columns are described here. "
            f"Use inspect_columns for any others you care about."
        )
    current._profile_cache = (max_columns, report)
    return report


@tool
def inspect_columns(columns: list[str], top_k: int = 10) -> dict[str, Any]:
    """Look closely at specific columns: distribution, missingness, link to the target.

    Use it after `profile_data` when a column looks suspicious or promising, or to
    read its description from the data dictionary.

    Args:
        columns: Column names to inspect.
        top_k: How many frequent values to list for a categorical column.
    """
    current = session()
    spec, store = current.spec, current.store
    unknown = [c for c in columns if c not in store.train.columns]
    if unknown:
        return {"status": "error", "problem": f"unknown columns: {unknown}",
                "hint": "See profile_data for the column list."}

    descriptions: dict[str, str] = {}
    if store.dictionary is not None and {"feature"}.issubset(store.dictionary.columns):
        text_column = next(
            (c for c in ("description_ru", "description") if c in store.dictionary.columns), None
        )
        if text_column:
            descriptions = dict(zip(store.dictionary["feature"], store.dictionary[text_column]))

    out: dict[str, Any] = {}
    for name in columns:
        series = store.train[name]
        entry: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_pct": round(float(series.isna().mean()) * 100, 2),
            "n_unique": int(series.nunique(dropna=True)),
            "predictive_power": round(
                profiling._single_feature_power(series, store.y, spec.kind), 4
            ),
        }
        if name in descriptions:
            entry["description"] = str(descriptions[name])
        if series.dtype.kind in "biufc":
            entry["quantiles"] = {
                str(q): round(float(series.quantile(q)), 4) for q in (0.01, 0.25, 0.5, 0.75, 0.99)
            }
        else:
            entry["top_values"] = {
                str(k): int(v) for k, v in series.astype(str).value_counts().head(top_k).items()
            }
            if name in store.test.columns:
                entry["unseen_in_test"] = int(
                    len(set(store.test[name].dropna().unique()) - set(series.dropna().unique()))
                )
        out[name] = entry
    return out
