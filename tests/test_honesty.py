#!/usr/bin/env python
"""
Tests for the properties the server actually claims.

The interesting claims here are not "the code runs" but "the agent cannot cheat".
Those deserve tests, so they are assertions rather than README prose.

Run: .venv/bin/python tests/test_honesty.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mlforge import data as D  # noqa: E402
from mlforge import guards as G  # noqa: E402
from mlforge.pipeline import compile_pipeline  # noqa: E402
from mlforge.schema import validate_recipe  # noqa: E402
from mlforge.session import Dataset  # noqa: E402

PASS, FAIL = [], []


def check(name: str):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  PASS  {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
        return fn
    return deco


def make_ds(**kw) -> Dataset:
    df, tgt = D.make_synthetic_churn(n=1200, seed=1)
    train, hold = D.split_holdout(df, tgt, "binary")
    return Dataset(id="t", source="synthetic", target=tgt, task="binary", train=train,
                   holdout=hold, issues=D.detect_issues(train, tgt, "binary"), **kw)


def recipe(**kw):
    base = {"name": "t", "model": {"type": "lightgbm"}}
    base.update(kw)
    norm, errors, _ = validate_recipe(base, task="binary")
    assert not errors, f"recipe invalid: {errors}"
    return norm


print("\nLeak detection")


@check("flags a column recorded after the outcome")
def _():
    ds = make_ds()
    leaks = [i["column"] for i in ds.issues if i["kind"] == "target_leakage"]
    assert "support_ticket_after_cancel" in leaks, f"missed the planted leak; flagged {leaks}"


@check("does not flag a unique ID as leakage")
def _():
    ds = make_ds()
    leaks = [i["column"] for i in ds.issues if i["kind"] == "target_leakage"]
    assert "customer_id" not in leaks, "ID column false-positived as a leak"
    ids = [i["column"] for i in ds.issues if i["kind"] == "identifier"]
    assert "customer_id" in ids, "ID column was not reported as an identifier either"


@check("does not flag honest columns on a clean dataset")
def _():
    df, tgt = D.load_builtin("wine")
    train, _ = D.split_holdout(df, tgt, "multiclass")
    leaks = [i for i in D.detect_issues(train, tgt, "multiclass") if i["kind"] == "target_leakage"]
    assert not leaks, f"false positives on a clean dataset: {leaks}"


print("\nGuard rails")


@check("blocks an experiment that keeps a flagged leak")
def _():
    ds = make_ds()
    try:
        G.check_experiment(ds, recipe(drop_columns=["customer_id"]), [], "")
        raise AssertionError("the leak guard did not fire")
    except G.GuardBlock as e:
        assert e.code == "LEAK_NOT_DROPPED", e.code
        assert "drop_columns" in e.hint, e.hint


@check("exempts the dummy baseline, which uses no features")
def _():
    ds = make_ds()
    G.check_experiment(ds, recipe(model={"type": "dummy"}), [], "")


@check("the agent cannot waive a leak flag on its own authority")
def _():
    ds = make_ds()
    try:
        G.check_experiment(ds, recipe(), ["LEAK_NOT_DROPPED"],
                           "This column is genuinely available at prediction time in production.")
        raise AssertionError("a self-authorised override was accepted")
    except G.GuardBlock as e:
        assert e.code == "LEAK_NOT_DROPPED" and not e.overridable, (e.code, e.overridable)
        assert "trusted_columns" in e.hint, "the hint does not point at the operator escape hatch"


@check("an operator-trusted column can be overridden, with the reason recorded")
def _():
    ds = make_ds(trusted_columns=["support_ticket_after_cancel"])
    acks = G.check_experiment(ds, recipe(), ["LEAK_NOT_DROPPED"],
                              "Operator confirmed this field is populated before scoring.")
    assert acks and "operator-trusted" in acks[0], acks
    assert "Operator confirmed" in acks[0], "the justification was not retained"


@check("override without a justification is itself refused")
def _():
    ds = make_ds()
    try:
        ds.trusted_columns = ["support_ticket_after_cancel"]
        G.check_experiment(ds, recipe(), ["LEAK_NOT_DROPPED"], "ok")
        raise AssertionError("accepted a one-word justification")
    except G.GuardBlock as e:
        assert e.code == "JUSTIFICATION_REQUIRED", e.code


@check("blocks random folds when the data is declared grouped")
def _():
    ds = make_ds(group_column="region")
    try:
        G.check_experiment(ds, recipe(drop_columns=["customer_id", "support_ticket_after_cancel"]), [], "")
        raise AssertionError("group leakage was allowed")
    except G.GuardBlock as e:
        assert e.code == "GROUP_LEAKAGE" and not e.overridable


@check("blocks random folds when the data is declared time-ordered")
def _():
    ds = make_ds(time_column="signup_date")
    try:
        G.check_experiment(ds, recipe(drop_columns=["customer_id", "support_ticket_after_cancel"]), [], "")
        raise AssertionError("time leakage was allowed")
    except G.GuardBlock as e:
        assert e.code == "TIME_LEAKAGE" and not e.overridable


@check("blocks a feature engineered from the target")
def _():
    ds = make_ds()
    r = recipe(features=[{"op": "ratio", "name": "cheat", "a": "churn", "b": "tenure_months"}])
    try:
        G.check_experiment(ds, r, [], "")
        raise AssertionError("a target-derived feature was allowed")
    except G.GuardBlock as e:
        assert e.code == "TARGET_IN_FEATURE" and not e.overridable


@check("a duplicate recipe is recognised regardless of its label")
def _():
    from mlforge.session import recipe_fingerprint
    a = recipe(name="try one", notes="first attempt")
    b = recipe(name="try two", notes="completely different wording")
    c = recipe(name="try three", drop_columns=["customer_id"])
    assert recipe_fingerprint(a) == recipe_fingerprint(b), "name/notes changed the fingerprint"
    assert recipe_fingerprint(a) != recipe_fingerprint(c), "a real change was not detected"


@check("an implausible score is tainted")
def _():
    assert G.taint_check(1.0, "roc_auc", "binary"), "roc_auc 1.0 was not tainted"
    assert G.taint_check(0.9999, "r2", "regression"), "r2 0.9999 was not tainted"
    assert not G.taint_check(0.87, "roc_auc", "binary"), "an ordinary score was tainted"


print("\nHoldout custody")


@check("the holdout shares no rows with the training split")
def _():
    ds = make_ds()
    overlap = set(map(tuple, ds.train.astype(str).to_numpy())) & \
              set(map(tuple, ds.holdout.astype(str).to_numpy()))
    assert not overlap, f"{len(overlap)} rows appear in both splits"


@check("the budget stops a fourth holdout evaluation")
def _():
    from mlforge.session import HOLDOUT_BUDGET, Run
    ds = make_ds()
    ds.holdout_used = HOLDOUT_BUDGET
    run = Run(id="r", dataset_id="t", recipe=recipe(), primary_metric="roc_auc",
              score=0.8, result={}, diagnosis=[], warnings=[])
    try:
        G.check_final_evaluate(ds, run, [])
        raise AssertionError("spent budget was not enforced")
    except G.GuardBlock as e:
        assert e.code == "HOLDOUT_EXHAUSTED" and not e.overridable


print("\nFold safety")


@check("every fitted transform lives inside the Pipeline, so CV refits it")
def _():
    from sklearn.pipeline import Pipeline
    ds = make_ds()
    pipe, _ = compile_pipeline(recipe(categorical={"encode": "target"},
                                      drop_columns=["customer_id", "support_ticket_after_cancel"]),
                               ds.X, "binary")
    assert isinstance(pipe, Pipeline)
    names = [n for n, _ in pipe.steps]
    assert "prep" in names and names[-1] == "model", names
    # ColumnTransformer.transformers holds (name, transformer, columns) triples.
    cat = next(t for n, t, _ in pipe.named_steps["prep"].transformers if n == "cat")
    assert any(n == "encode" for n, _ in cat.steps), "the encoder is not inside the pipeline"


@check("target encoding is cross-fitted, not fitted on the full target")
def _():
    from sklearn.preprocessing import TargetEncoder
    ds = make_ds()
    pipe, _ = compile_pipeline(recipe(categorical={"encode": "target"},
                                      drop_columns=["customer_id", "support_ticket_after_cancel"]),
                               ds.X, "binary")
    cat = next(t for n, t, _ in pipe.named_steps["prep"].transformers if n == "cat")
    enc = cat.named_steps["encode"]
    assert isinstance(enc, TargetEncoder), type(enc)
    assert getattr(enc, "cv", 5) >= 2, "TargetEncoder is not cross-fitting"


@check("a leaky recipe scores ~1.0 while the clean one does not -- the planted leak is real")
def _():
    from mlforge.train import run_cv
    ds = make_ds()
    leaky, _ = compile_pipeline(recipe(drop_columns=["customer_id"],
                                       model={"type": "logreg"},
                                       numeric={"scale": "standard"},
                                       categorical={"encode": "onehot"}), ds.X, "binary")
    clean, _ = compile_pipeline(recipe(drop_columns=["customer_id", "support_ticket_after_cancel"],
                                       model={"type": "logreg"},
                                       numeric={"scale": "standard"},
                                       categorical={"encode": "onehot"}), ds.X, "binary")
    r = recipe()
    a = run_cv(leaky, ds.X, ds.y, "binary", r)["metrics"]["roc_auc"]["mean"]
    b = run_cv(clean, ds.X, ds.y, "binary", r)["metrics"]["roc_auc"]["mean"]
    assert a > 0.99, f"the planted leak did not produce a perfect score (got {a})"
    assert b < 0.90, f"the clean pipeline is suspiciously good (got {b})"


print("\nRecipe validation")


@check("reports every error in one pass, with suggestions")
def _():
    _, errors, _ = validate_recipe(
        {"model": {"type": "lightgm"}, "numeric": {"scale": "standarde"}, "metric": "rmse"},
        task="binary", columns=["a"])
    fields = {e["field"] for e in errors}
    assert {"model.type", "numeric.scale", "metric"} <= fields, fields
    assert any("lightgbm" in e["hint"] for e in errors), "no did-you-mean for the model typo"


@check("accepts a minimal recipe and fills the defaults")
def _():
    norm, errors, _ = validate_recipe({"model": {"type": "lightgbm"}}, task="binary")
    assert not errors, errors
    assert norm["validation"]["folds"] == 5 and norm["metric"] == "auto"


print("\nRetrieval")


@check("the playbook answers the questions the diagnosis provokes")
def _():
    from mlforge.rag import search
    cases = [("my model scores 1.0 roc auc what is wrong", "too good to be true"),
             ("overfitting gap between train and validation", "OVERFITTING"),
             ("rows repeated per patient how to validate", "Grouped data"),
             ("how to encode high cardinality categorical", "cardinality")]
    for q, expect in cases:
        hits = search(q, top_k=2)
        assert hits, f"no hits for {q!r}"
        assert any(expect.lower() in h["title"].lower() for h in hits), \
            f"{q!r} -> {[h['title'] for h in hits]}, expected something matching {expect!r}"


print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
for n, e in FAIL:
    print(f"  FAILED: {n}\n          {e}")
sys.exit(1 if FAIL else 0)
