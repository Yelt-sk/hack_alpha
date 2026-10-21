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

from agent.tools.registry import Registry, ToolError  # noqa: E402


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

from agent.recipes.schema import repair, validate  # noqa: E402


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

from agent.guards import implausible  # noqa: E402


@check("a perfect score is flagged and an ordinary one is not")
def _():
    assert implausible(1.0, "macro_f1"), "a perfect macro F1 was not flagged"
    assert implausible(0.0, "rmse"), "a zero RMSE was not flagged"
    assert not implausible(0.87, "macro_f1"), "an ordinary score was flagged"
    assert not implausible(43.2, "rmse"), "an ordinary RMSE was flagged"


# ---------------------------------------------------------------------------
print("\nLedger")
# ---------------------------------------------------------------------------

from agent.state import Clock, Experiment, Ledger  # noqa: E402


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
    from agent.task import load_task  # noqa: E402

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
    from agent.submission import SubmissionManager, frame_from_predictions  # noqa: E402

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
    from agent.recipes.pipeline import compile_pipeline  # noqa: E402

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
        from agent.state import DataStore

        spec = load_task(REG)
        cache = ROOT / ".test_cache"
        first = DataStore(spec, cache).fold_id.copy()
        second = DataStore(spec, cache).fold_id
        assert (first == second).all(), "fold assignment is not reproducible"
        assert len(set(first.tolist())) == 5, "unexpected number of folds"


print(f"\n{'=' * 60}\n{len(PASSED)} passed, {len(FAILED)} failed")
for name, reason in FAILED:
    print(f"  FAILED: {name}\n          {reason}")
sys.exit(1 if FAILED else 0)
