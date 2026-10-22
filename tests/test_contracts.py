#!/usr/bin/env python
"""Tests for the properties this system actually promises.

The interesting claims are not "the code runs" but "the answer is always valid"
and "the agent cannot quietly cheat". Those deserve assertions rather than prose
in a README, so they are here.

Run: .venv/bin/python tests/test_contracts.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from typing import Literal  # noqa: E402

import numpy as np  # noqa: E402


def _quiet(*_: object) -> None:
    """Swallow the fold plan's rationale; the suite reports its own lines."""
import pandas as pd  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    def decorate(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except AssertionError as err:
            FAILED.append((name, str(err) or "assertion failed"))
            print(f"  FAIL  {name}: {err}")
        except Exception as err:  # noqa: BLE001
            FAILED.append((name, f"{type(err).__name__}: {err}"))
            print(f"  ERROR {name}: {type(err).__name__}: {err}")
        return fn

    return decorate


REG = ROOT / "data" / "Регрессия" / "task.md"
CLS = ROOT / "data" / "Многоклассовая классификация" / "task.md"
HAVE_DATA = REG.is_file() and CLS.is_file()


# ---------------------------------------------------------------------------
print("\nTool registry")
# ---------------------------------------------------------------------------

from agent_registry import Registry, ToolError  # noqa: E402


def _fresh_registry() -> Registry:
    registry = Registry()

    @registry.tool
    def demo(mode: Literal["a", "b"] = "a", count: int = 1) -> dict:
        """A demo tool.

        Args:
            mode: Which mode to use.
            count: How many times.
        """
        return {"mode": mode, "count": count}

    @registry.tool(experimental=True)
    def broken() -> dict:
        """An experimental tool that fails."""
        raise RuntimeError("boom")

    return registry


@check("the signature is the schema, with enums and defaults")
def _():
    schema = _fresh_registry().schemas()[0]["function"]
    properties = schema["parameters"]["properties"]
    assert properties["mode"]["enum"] == ["a", "b"], properties["mode"]
    assert properties["mode"]["description"] == "Which mode to use."
    assert properties["count"]["type"] == "integer"
    assert schema["parameters"]["required"] == [], schema["parameters"]["required"]


@check("a signature whose annotations cannot be resolved is refused, not degraded")
def _():
    registry = Registry()

    def hidden(mode: "Nowhere" = "a") -> dict:  # noqa: F821
        """A tool annotated with a name that does not exist."""
        return {}

    try:
        registry.tool(hidden)
        raise AssertionError("a tool with unresolvable annotations was registered untyped")
    except ValueError as err:
        assert "annotations" in str(err), err


@check("a tool without a docstring is refused at registration")
def _():
    registry = Registry()
    try:
        registry.tool(lambda: None)
        raise AssertionError("a tool with no description was accepted")
    except ValueError:
        pass


@check("an unknown tool name comes back with a suggestion")
def _():
    out = _fresh_registry().dispatch("dem", {})
    assert "unknown_tool" in out and "demo" in out, out


@check("bad arguments are reported, not raised")
def _():
    out = _fresh_registry().dispatch("demo", {"nope": 1})
    assert "bad_arguments" in out and "nope" in out, out


@check("malformed JSON arguments are reported, not raised")
def _():
    out = _fresh_registry().dispatch("demo", "{broken")
    assert "bad_json" in out, out


@check("a failing experimental mod disables itself instead of ending the run")
def _():
    registry = _fresh_registry()
    out = registry.dispatch("broken", {})
    assert "exception" in out, out
    assert "broken" in registry.disabled(), registry.disabled()
    assert "broken" not in registry.names(), "a disabled tool is still offered"


@check("ToolError carries a correction the model can resend")
def _():
    payload = ToolError("bad", "fix it", fix={"a": 1}).payload()
    assert payload["corrected_input"] == {"a": 1} and "next_step" in payload


# ---------------------------------------------------------------------------
print("\nRecipe vocabulary")
# ---------------------------------------------------------------------------

from agent_recipe_schema import repair, validate  # noqa: E402


@check("a flattened recipe is repaired and the repair is reported")
def _():
    fixed, notes = repair({"model": "lightgbm", "learning_rate": 0.05, "params": {"num_leaves": 31}})
    assert fixed["model"]["type"] == "lightgbm", fixed
    assert fixed["model"]["params"] == {"num_leaves": 31, "learning_rate": 0.05}, fixed
    assert notes, "the repair was silent"


@check("every error is reported in one pass, with suggestions")
def _():
    _, errors, _ = validate(
        {"model": {"type": "lightgm"}, "numeric": {"scale": "standarde"}}, "regression", ["a"]
    )
    fields = {e["field"] for e in errors}
    assert {"model.type", "numeric.scale"} <= fields, fields
    assert any("lightgbm" in e["hint"] for e in errors), "no did-you-mean for the model typo"


@check("a model is refused for a task kind it cannot do")
def _():
    _, errors, _ = validate({"model": {"type": "ridge"}}, "multiclass")
    assert any(e["field"] == "model.type" for e in errors), errors


@check("an AutoML model without a time budget is refused")
def _():
    _, errors, _ = validate({"model": {"type": "flaml"}}, "regression")
    assert any(e["field"] == "time_budget_s" for e in errors), errors


@check("every target transform is refused on a classification target")
def _():
    # Written against the rule, not against a value. The check in the schema used
    # to name log1p specifically, so widening the vocabulary to sqrt and cbrt let
    # them through on a class label -- a failure that would have surfaced deep
    # inside the fold loop, after the fit.
    from agent_recipe_schema import TARGET_TRANSFORM

    for transform in TARGET_TRANSFORM:
        for kind in ("multiclass", "binary"):
            _, errors, _ = validate(
                {"name": "x", "target_transform": transform, "model": {"type": "lightgbm"}},
                kind, ["a", "b"],
            )
            if transform == "none":
                assert not errors, f"'none' should be fine on {kind}: {errors}"
            else:
                assert errors, f"'{transform}' was accepted on {kind}"


@check("log1p is refused on a classification target")
def _():
    _, errors, _ = validate({"target_transform": "log1p"}, "multiclass")
    assert any(e["field"] == "target_transform" for e in errors), errors


@check("an unscaled linear model is warned about, not blocked")
def _():
    _, errors, warnings_ = validate({"model": {"type": "ridge"}}, "regression")
    assert not errors, errors
    assert any("scale" in w for w in warnings_), warnings_


# ---------------------------------------------------------------------------
print("\nPlausibility guard")
# ---------------------------------------------------------------------------

from agent_guards import implausible  # noqa: E402


@check("a perfect score is flagged and an ordinary one is not")
def _():
    assert implausible(1.0, "macro_f1"), "a perfect macro F1 was not flagged"
    assert implausible(0.0, "rmse"), "a zero RMSE was not flagged"
    assert not implausible(0.87, "macro_f1"), "an ordinary score was flagged"
    assert not implausible(43.2, "rmse"), "an ordinary RMSE was flagged"


# ---------------------------------------------------------------------------
print("\nLedger")
# ---------------------------------------------------------------------------

from agent_state import Clock, Experiment, Ledger  # noqa: E402


def _experiment(identifier: str, value: float, folds: list[float]) -> Experiment:
    return Experiment(identifier, identifier, "recipe", value, folds, "rmse", "minimize", 1.0)


@check("a gain inside the fold noise is named as noise")
def _():
    ledger = Ledger("minimize")
    ledger.add(_experiment("a", 10.0, [10.1, 9.9, 10.0]))
    verdict = ledger.verdict(ledger.add(_experiment("b", 9.99, [10.0, 9.98, 9.99])))
    assert "noise" in verdict.lower(), verdict


@check("a gain beyond the noise is named as a new best")
def _():
    ledger = Ledger("minimize")
    ledger.add(_experiment("a", 10.0, [10.1, 9.9, 10.0]))
    verdict = ledger.verdict(ledger.add(_experiment("b", 8.0, [8.1, 7.9, 8.0])))
    assert "NEW BEST" in verdict, verdict


@check("a tainted experiment never becomes the best")
def _():
    ledger = Ledger("maximize")
    ledger.add(_experiment("honest", 0.7, [0.7]))
    cheat = _experiment("cheat", 1.0, [1.0])
    cheat.tainted = "implausible"
    ledger.add(cheat)
    assert ledger.best().id == "honest", ledger.best().id


@check("the clock reserves time for finalising")
def _():
    clock = Clock(total_seconds=1000, reserve_seconds=400)
    assert clock.usable_remaining <= 600 + 1, clock.usable_remaining
    assert not clock.affordable(900), "an unaffordable operation was allowed"
    assert Clock(total_seconds=10, reserve_seconds=20).must_finalize


# ---------------------------------------------------------------------------
print("\nTask understanding" + ("" if HAVE_DATA else " (skipped: no data/)"))
# ---------------------------------------------------------------------------

if HAVE_DATA:
    from agent_task import load_task  # noqa: E402

    @check("the regression task is read correctly")
    def _():
        spec = load_task(REG)
        assert spec.kind == "regression", spec.kind
        assert spec.metric == "rmse", spec.metric
        assert (spec.id_column, spec.target_column) == ("id", "site_eui")
        assert not spec.warnings, spec.warnings

    @check("the classification task picks the primary metric, not a reference one")
    def _():
        spec = load_task(CLS)
        assert spec.kind == "multiclass", spec.kind
        # The statement also mentions Accuracy and Weighted F1; only Macro F1 is scored.
        assert spec.metric == "macro_f1", f"chose {spec.metric}"
        assert len(spec.classes) == 10, spec.classes
        assert not spec.warnings, spec.warnings

    @check("the target is read from the submission template, not from prose")
    def _():
        for path, target in ((REG, "site_eui"), (CLS, "attack_cat")):
            spec = load_task(path)
            header = pd.read_csv(spec.files["sample_submission"], nrows=0).columns.tolist()
            assert header[-1] == target == spec.target_column, (header, spec.target_column)


# ---------------------------------------------------------------------------
print("\nSubmission custody" + ("" if HAVE_DATA else " (skipped: no data/)"))
# ---------------------------------------------------------------------------

if HAVE_DATA:
    from agent_submission import SubmissionManager, frame_from_predictions  # noqa: E402

    def _manager():
        spec = load_task(CLS)
        test_ids = pd.read_csv(spec.files["test"], usecols=[spec.id_column])[spec.id_column]
        return spec, SubmissionManager(spec, test_ids, known_labels=spec.classes)

    def _good(spec, manager):
        return frame_from_predictions(
            spec, manager.test_ids, np.full(len(manager.test_ids), "Normal")
        )

    @check("a well-formed submission is accepted")
    def _():
        spec, manager = _manager()
        assert manager.validate(_good(spec, manager)) == []

    @check("every malformed submission is rejected with a reason")
    def _():
        spec, manager = _manager()
        good = _good(spec, manager)
        cases = {
            "renamed column": lambda f: f.rename(columns={"id": "ID"}),
            "extra column": lambda f: f.assign(extra=1),
            "reordered rows": lambda f: f.iloc[::-1].reset_index(drop=True),
            "missing rows": lambda f: f.iloc[:-5],
            "null target": lambda f: f.assign(attack_cat=[None] + list(f["attack_cat"][1:])),
            "unknown label": lambda f: f.assign(attack_cat=["Nope"] + list(f["attack_cat"][1:])),
        }
        for label, mutate in cases.items():
            problems = manager.validate(mutate(good.copy()))
            assert problems, f"{label} was accepted"

    @check("a worse score does not replace the saved answer")
    def _():
        spec, manager = _manager()
        manager.save(_good(spec, manager), score=0.60, source="first")
        result = manager.save(_good(spec, manager), score=0.10, source="worse")
        assert result["status"] == "kept_previous", result
        spec.submission_path.unlink(missing_ok=True)

    @check("a forced write without a score clears the recorded score")
    def _():
        spec, manager = _manager()
        manager.save(_good(spec, manager), score=0.60, source="first")
        manager.save(_good(spec, manager), score=None, source="forced", force=True)
        assert manager.state.score is None, (
            "the recorded score outlived the file it described; a worse model "
            "could then be rejected for not beating a number that no longer applies"
        )
        spec.submission_path.unlink(missing_ok=True)

    @check("the floor is scored, so the ladder can climb past it")
    def _():
        # The regression this prevents: a floor written with no score, once an
        # unknown score started locking the file, stopped the ladder on its first
        # rung -- the constant stayed on disk and every real model was refused.
        # A floor nothing can beat is not a floor.
        spec, manager = _manager()
        y = pd.read_csv(spec.files["train"], usecols=[spec.target_column])[spec.target_column]
        floor = manager.save_constant(y)
        assert floor["validation_score"] is not None, (
            "the floor has no score, so nothing can be shown to beat it"
        )
        better = 0.9 if spec.direction == "maximize" else floor["validation_score"] / 2
        assert manager.save(_good(spec, manager), score=better,
                            source="model")["status"] == "saved", "the ladder cannot climb"
        spec.submission_path.unlink(missing_ok=True)

    @check("an unscored submission cannot be silently overwritten")
    def _():
        # Observed in a full three-hour run: the agent hand-built an ensemble
        # worth 0.6632 and saved it as a file, which left the recorded score
        # unknown. The fallback pipeline then ran out the clock with ordinary
        # recipes, and every one of them -- including a constant at 0.048 --
        # replaced it, because an unknown score switched the comparison off.
        # The run earned 0.6632 and would have submitted 0.5984.
        spec, manager = _manager()
        manager.save(_good(spec, manager), score=None, source="hand_built")
        result = manager.save(_good(spec, manager), score=0.10, source="fallback")
        assert result["status"] == "kept_previous", (
            "a scored result replaced an unscored one; nothing established it was "
            "better, and the run's best answer is what gets destroyed"
        )
        assert "evaluate_predictions" in result.get("hint", ""), result
        spec.submission_path.unlink(missing_ok=True)

    @check("an unscored submission can still be replaced deliberately")
    def _():
        # The lock must not become a trap: whoever knows better can still act.
        spec, manager = _manager()
        manager.save(_good(spec, manager), score=None, source="hand_built")
        result = manager.save(_good(spec, manager), score=0.9, source="better", force=True)
        assert result["status"] == "saved", result
        assert manager.state.score == 0.9
        spec.submission_path.unlink(missing_ok=True)

    @check("the constant floor produces a file that validates")
    def _():
        spec, manager = _manager()
        y = pd.read_csv(spec.files["train"], usecols=[spec.target_column])[spec.target_column]
        assert manager.save_constant(y)["status"] == "saved"
        written = pd.read_csv(spec.submission_path)
        assert manager.validate(written) == [], manager.validate(written)
        spec.submission_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
print("\nFold safety" + ("" if HAVE_DATA else " (skipped: no data/)"))
# ---------------------------------------------------------------------------

if HAVE_DATA:
    from agent_recipe_pipeline import compile_pipeline  # noqa: E402

    @check("every fitted transform sits inside the pipeline, so folds refit it")
    def _():
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import TargetEncoder

        spec = load_task(REG)
        X = pd.read_csv(spec.files["train"], nrows=500).drop(columns=[spec.target_column])
        recipe, errors, _ = validate(
            {"categorical": {"encode": "target"}, "numeric": {"winsorize": 0.01},
             "model": {"type": "lightgbm"}},
            "regression", list(X.columns),
        )
        assert not errors, errors
        pipeline, _ = compile_pipeline(recipe, X, "regression")
        assert isinstance(pipeline, Pipeline)
        categorical = next(t for n, t, _ in pipeline.named_steps["prep"].transformers if n == "cat")
        encoder = categorical.named_steps["encode"]
        assert isinstance(encoder, TargetEncoder), type(encoder)
        assert "winsorize" in dict(
            next(t for n, t, _ in pipeline.named_steps["prep"].transformers if n == "num").steps
        ), "winsorisation bounds are learned outside the pipeline"

    @check("folds are identical for every experiment in a run")
    def _():
        from agent_state import DataStore

        spec = load_task(REG)
        cache = ROOT / ".test_cache"
        one, two = DataStore(spec, cache, log=_quiet), DataStore(spec, cache, log=_quiet)
        assert (one.fold_id == two.fold_id).all(), "fold assignment is not reproducible"
        assert one.plan.name == two.plan.name, "the chosen scheme is not reproducible"
        # The count is whatever the plan decided, not a fixed five: this task is a
        # forecast, so it is validated on its last few periods rather than on five
        # random slices. What has to hold is that every split is non-empty and that
        # no row is validated twice.
        assert one.n_splits >= 2, f"only {one.n_splits} split(s)"
        assert all(len(a) and len(b) for a, b in one.splits()), "an empty split"
        seen = np.concatenate([b for _, b in one.splits()])
        assert len(seen) == len(set(seen.tolist())), "a row is validated by two folds"
        assert set(seen.tolist()) == set(np.where(one.evaluable)[0].tolist()), \
            "fold_id and the splits disagree about which rows are validated"


    @check("a forecast plan never trains on the period it validates, or later")
    def _():
        # The defect this prevents is the one that cost two submissions: random
        # folds on a task whose test set is a later period measured RMSE 41.9 where
        # the honest arrangement measured 59.4. If a training half ever contains the
        # validation period, the plan has silently become interpolation again.
        from agent_state import DataStore

        store = DataStore(load_task(REG), ROOT / ".test_cache", log=_quiet)
        if store.plan.name != "forward_chain":
            return
        axis = store.plan.diagnosis["separating_feature"]["feature"]
        values = store.train[axis].to_numpy()
        for train_idx, val_idx in store.splits():
            origin = values[val_idx][0]
            assert (values[val_idx] == origin).all(), "a fold mixes two periods"
            assert values[train_idx].max() < origin, (
                f"training half reaches {values[train_idx].max()} while validating {origin}"
            )

    @check("no tool is invisible in the regime the agent actually uses")
    def _():
        # evaluate_predictions used to be custom-only while the default regime is
        # standard, so the one bridge from hand-written code to the ledger was
        # absent from the tool list in all four runs we have logs for -- while three
        # docstrings told the agent to call it. A tool the prompt depends on cannot
        # be conditional.
        import agent_tools  # noqa: F401
        import agent_rag  # noqa: F401
        import agent_mods

        agent_mods.load_all()
        from agent_registry import REGISTRY

        offered = set(REGISTRY.names("standard"))
        for name in REGISTRY.names(None):
            assert name in offered, f"'{name}' is not offered in the default regime"

    @check("a tool named in a docstring exists")
    def _():
        import agent_tools  # noqa: F401
        import agent_rag  # noqa: F401
        import agent_mods

        agent_mods.load_all()
        from agent_registry import REGISTRY

        # Every backticked name shaped like one of our tools must resolve. Catches a
        # rename, or a regime gate, that leaves a docstring pointing at something the
        # agent cannot call.
        import re

        known = set(REGISTRY.tools)

        for path in sorted(ROOT.glob("agent_*.py")) + sorted(ROOT.glob("mod_*.py")):
            for ref in set(re.findall(r"`(\w+)`", path.read_text())):
                if ref.startswith(("run_", "save_", "get_", "set_", "evaluate_",
                                   "profile_", "inspect_", "tune_", "search_", "blend_")):
                    assert ref in known, f"{path.name} refers to `{ref}`, which is not a tool"


print("\nEngine adapters")
# ---------------------------------------------------------------------------
# No engine needed: these check the glue, which is where the bugs were.


@check("an engine's class order is read from the engine, not assumed")
def _():
    # The defect this prevents produced a plausible-looking wrong number rather
    # than an error. LightAutoML's reader numbers the classes in its own order --
    # measured as {'mid': 0, 'alpha': 1, 'zebra': 2} where alphabetical order is
    # alpha, mid, zebra -- so assuming sorted(unique(y)) permutes every
    # prediction. It turned a classification benchmark into macro F1 0.17, which
    # reads as a weak engine and is in fact scrambled labels.
    import mod_engine

    class Reader:
        class_mapping = {"mid": 0, "alpha": 1, "zebra": 2}

    class Model:
        reader = Reader()

    order = mod_engine._lama_classes(Model(), ["alpha", "mid", "zebra"])
    assert order == ["mid", "alpha", "zebra"], order
    # And it must fall back rather than crash when the attribute is absent, because
    # a future version may rename it.
    assert mod_engine._lama_classes(object(), ["a", "b"]) == ["a", "b"]


@check("an engine is refused rather than attempted when it cannot be imported")
def _():
    import mod_engine

    reason = mod_engine._available("autogluon")
    # Either it imports here or it says why; what must never happen is a silent
    # truthy value that the caller reads as "ready".
    assert isinstance(reason, str)
    assert mod_engine._available("flaml") == "" or "Error" in mod_engine._available("flaml")


print("\nSplit recognition")
# ---------------------------------------------------------------------------
# No data/ needed: these tables are generated here on purpose, so the
# recogniser is tested against tasks it was not written against.

def _synthetic(n: int, duplicates: float = 0.0, period: list | None = None, seed: int = 0):
    """A table with no connection to either hackathon dataset.

    The point of these cases is that the recogniser must be driven by what the
    data shows, not by the two tasks it was written against. Two of the five
    exist to check that it declines: a split along unordered categories gives no
    direction to extrapolate in, and a row identifier looks exactly like a period
    unless you notice it has one value per row.
    """
    r = np.random.default_rng(seed)
    frame = pd.DataFrame({f"f{i}": r.normal(size=n) for i in range(6)})
    frame["cat"] = r.choice(list("abcd"), n)
    if duplicates:
        k = int(n * duplicates)
        frame.iloc[n - k:] = frame.iloc[r.choice(n - k, k)].to_numpy()
    if period is not None:
        frame["period"] = r.choice(period, n)
    return frame


@check("an ordinary task keeps ordinary folds")
def _():
    import agent_split

    train, test = _synthetic(4000, seed=1), _synthetic(2000, seed=2)
    y = pd.Series((train.f0 + train.f1 > 0).astype(int))
    plan = agent_split.plan(train, test, y, list(train.columns), True, log=_quiet)
    assert plan.name == "stratified", plan.name
    assert not plan.full_fit_for_test


@check("duplicates are grouped")
def _():
    import agent_split

    train, test = _synthetic(4000, duplicates=0.30, seed=3), _synthetic(2000, seed=4)
    y = pd.Series(np.random.default_rng(5).choice(list("XYZ"), len(train)))
    plan = agent_split.plan(train, test, y, list(train.columns), True, log=_quiet)
    assert plan.name == "grouped", plan.name
    groups = agent_split.duplicate_groups(train, list(train.columns))
    for train_idx, val_idx in plan.splits:
        shared = set(groups[train_idx]) & set(groups[val_idx])
        assert not shared, f"{len(shared)} duplicate groups straddle a fold"


@check("a later period is recognised as a forecast")
def _():
    import agent_split

    train = _synthetic(4000, period=[2019, 2020, 2021, 2022], seed=6)
    test = _synthetic(2000, period=[2023], seed=7)
    y = pd.Series(np.random.default_rng(8).normal(size=len(train)))
    plan = agent_split.plan(train, test, y, list(train.columns), False, log=_quiet)
    assert plan.name == "forward_chain", plan.name
    assert plan.diagnosis["separating_feature"]["feature"] == "period"
    assert plan.full_fit_for_test, "fold models are blind to the newest period"


@check("unordered categories separate the sets but do not order them")
def _():
    import agent_split

    train = _synthetic(3000, seed=9)
    train["site"] = np.random.default_rng(10).choice(["msk", "spb"], len(train))
    test = _synthetic(1500, seed=11)
    test["site"] = "kzn"
    y = pd.Series(np.random.default_rng(12).normal(size=len(train)))
    plan = agent_split.plan(train, test, y, list(train.columns), False, log=_quiet)
    # Reported, because it matters, but not acted on: there is no "earlier" to
    # train on when the values have no order.
    assert plan.name == "kfold", plan.name
    assert plan.diagnosis["separating_feature"]["kind"] == "disjoint_categories"


@check("a row identifier is not mistaken for a period")
def _():
    import agent_split

    train, test = _synthetic(3000, seed=13), _synthetic(1500, seed=14)
    train["row_id"] = np.arange(len(train))
    test["row_id"] = np.arange(len(train), len(train) + len(test))
    y = pd.Series(np.random.default_rng(15).normal(size=len(train)))
    plan = agent_split.plan(train, test, y, list(train.columns), False, log=_quiet)
    assert plan.name == "kfold", plan.name
    assert "separating_feature" not in plan.diagnosis, "an id was read as a forecast axis"


print(f"\n{'=' * 60}\n{len(PASSED)} passed, {len(FAILED)} failed")
for name, reason in FAILED:
    print(f"  FAILED: {name}\n          {reason}")
sys.exit(1 if FAILED else 0)
