# AutoML agent — tool space

An AI agent that reads an ML task, investigates the data, runs experiments and
leaves a file of predictions. Built for the Alfa hackathon track *«AI-агент для
разработки ML-моделей»*.

This repository is the **substrate**: the tool space the agent acts through, the
guarantees around it, and the gateway for hanging further work on it. Retrieval
is developed separately and plugs in through a fixed seam.

---

## The two facts that shaped the design

**There is no MCP in the container.** `baseline.py` demonstrates the idea with
`FastMCP`, but the organisers confirmed the package is not installed: the
submission script offers tools to the model and executes the chosen action
itself. So the tool layer is in-process. The contracts stay MCP-shaped — `name`,
`description`, `parameters` is exactly what an MCP server advertises — so the
same registry can be served over real MCP for local debugging.

**The expensive failure is zero, not a weak model.** Three hours with a hard
kill, and `S = 100·max(0, 1 − RMSE/B_RMSE)`: no file, a malformed file, or a
result at baseline level all score nothing. So a valid submission exists from the
first seconds of the run and is only ever replaced by something that validates
and scores better.

---

## How a run goes

```
solution.py task.md
  ├── read the task           derive paths, target, kind, metric — nothing hard-coded
  ├── load data, fix folds    one fold assignment reused by every experiment
  ├── rung 1: constant        a valid submission exists within seconds
  ├── rung 2: default LGBM    a real floor within minutes
  ├── watchdog thread         writes the best known answer if time runs out
  └── the agent               improves on the floor, or the autopilot does
```

If `OPENAI_API_KEY` and `OPENAI_MODEL` are absent, or the loop fails, the run
falls through to a deterministic pipeline instead of to nothing.

---

## The tool space

Twelve tools. The vocabulary is wide — the model driving this is capable — but
the things that are expensive to get wrong are not left to it.

| Tool | What it is for |
|---|---|
| `get_context` | task, metric, time left, best so far, saved submission |
| `set_strategy` | declare the regime: recipes, or your own code |
| `profile_data` | shapes, types, missing rates, and quality findings with actions |
| `inspect_columns` | distribution, dictionary description, single-feature power |
| `get_recipe_schema` | the recipe vocabulary and worked examples |
| `run_recipe` | compile, cross-validate honestly, score, record, save if best |
| `tune_recipe` | Optuna on the same folds, inside a budget it actually honours |
| `evaluate_predictions` | score your own predictions on the run's folds |
| `run_python` | arbitrary code in a subprocess, with the data pre-cached |
| `leaderboard` | every attempt, ranked, with the fold-noise band |
| `save_submission` | validate and write the answer |
| `search_knowledge` | the retrieval seam (no-op until the retriever is installed) |

### Hybrid by default

Recipes are the fast path: the agent decides *what to try*, proven algorithms do
the work. `run_python` stays available for anything a recipe cannot express,
which matters because two of the four graded tasks are never shown to us.

`evaluate_predictions` is what makes the hybrid one system rather than two. It
scores hand-written predictions on the same folds as recipes, so the leaderboard
can compare them. Without it the two regimes produce incomparable numbers.

---

## What the agent cannot do

* **Choose its own validation split.** Folds belong to the run, not to a recipe.
* **Leak across folds.** Imputers, encoders, rare-category lists and
  winsorisation bounds all live inside the `Pipeline` and are refitted per fold.
  There is no recipe field that would place them outside it.
* **Break the answer.** Every write is validated against `test.csv` first —
  exact columns, every id once, test order, no missing values, known labels —
  and written atomically. A rejected file leaves the previous answer intact.
* **Pass off a perfect score.** Results beyond a plausibility threshold are
  flagged and never become the best until explained.
* **Overrun the clock.** Work that cannot finish before the reserve is refused.

---

## Experimental mods

Drop a `mod_*.py` beside `solution.py` and it becomes a tool:

```python
# mod_my_idea.py
from agent_session import session
from agent_registry import tool

@tool(experimental=True)
def my_idea(window: int = 3) -> dict:
    """One line the model will read as the tool description.

    Args:
        window: What it controls.
    """
    ...
```

Discovered at startup by filename prefix; the signature becomes the JSON schema.
A mod that raises at import or at call time is disabled and logged rather than
allowed to end the run — an unfinished idea must not cost a submission.

The prefix, rather than a folder, is what makes this work on the platform: only
root files are copied next to the task, so a mod in a subfolder would be left
behind.

---

## Retrieval seam

Knowledge retrieval is built separately. Implement `search(query, top_k)` in
`rag_impl.py`, in the archive root beside `solution.py`, returning `[{"title", "source", "relevance", "content"}]`
and `search_knowledge` starts using it. Absent or failing, the tool says so and
the run continues.

Worth indexing: `task.md`, `data_dictionary.csv` (it carries `description_ru`
per field), and a playbook covering when a recipe suffices and when the task
needs hand-written code.

---

## Layout

Everything that runs lives in the archive root. This is not a style choice:
the platform copies **only the root `.py` files and `requirements.txt`** into the
task directory, so a module in a subfolder is simply absent at grading time. The
names keep the hierarchy readable instead.

```
solution.py                 entry point: ladder, watchdog, fallbacks
requirements.txt            the provided dependency list, unmodified

agent_task.py               derive paths, target, kind, metric from the task files
agent_state.py              Clock, DataStore (data + shared folds), Ledger
agent_session.py            the one live session every tool reads
agent_submission.py         validation and atomic custody of submission.csv
agent_metrics.py            one scoring path for every regime
agent_profiling.py          profile and quality findings
agent_guards.py             plausibility check
agent_prompt.py             the system prompt
agent_loop.py               the agent loop, context budgeting, spend accounting
agent_autopilot.py          deterministic pipeline, used when there is no model
agent_bootstrap.py          the guarantee ladder
agent_rag.py                retrieval seam

agent_recipe_schema.py      vocabulary, repair, validation, model registry
agent_recipe_pipeline.py    recipe -> leak-safe pipeline, cross-validated

agent_registry.py           functions -> OpenAI tool schemas, and dispatch
agent_tools.py              imports the core tools, registering them
agent_tool_context.py       get_context, set_strategy
agent_tool_data.py          profile_data, inspect_columns
agent_tool_modeling.py      get_recipe_schema, run_recipe, tune_recipe
agent_tool_execution.py     run_python
agent_tool_submit.py        evaluate_predictions, save_submission, leaderboard
agent_mods.py               experimental gateway (discovers mod_*.py)
mod_blend_best.py           an experimental mod: blend_top_experiments

scripts/package.py          build the ZIP and verify it the way the platform runs it
scripts/drive.py            drive the tools by hand, with no model in the loop
tests/test_contracts.py     assertions on the guarantees above
REPORT.md                   the 27B run: actions, cost, results, what to improve
```

`scripts/` and `tests/` are development-only and deliberately stay out of the
archive.

---

## Running it

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# deterministic, no model needed
AGENT_TOTAL_SECONDS=600 .venv/bin/python solution.py "data/Регрессия/task.md"

# with a model
OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... \
  .venv/bin/python solution.py "data/Регрессия/task.md"

.venv/bin/python tests/test_contracts.py
```

`AGENT_TOTAL_SECONDS`, `AGENT_RESERVE_SECONDS` and `AGENT_MAX_STEPS` override the
defaults, which is how a three-hour run is rehearsed in ten minutes.

The case data is not committed. Unpack the provided archive into `data/` to get
`data/Регрессия/` and `data/Многоклассовая классификация/`.

---

## What a live run showed

Driven by a local **Qwen3-1.7B** (llama.cpp, 16k context) on the regression task.
Far weaker than the Qwen3-27B the platform provides, which is the point: it
stresses the tool surface rather than the modelling.

The first attempt died at step 4 — the window overflowed and the loop gave up
with thirteen minutes unused. Three fixes came out of that, each of which
applies to any model:

* **Budget the whole request, not the recent history.** The system prompt and
  tool schemas are most of it. Dense JSON also tokenises at roughly three
  characters per token, not four — optimism there is what overflowed a 16k
  window. An overflow now shrinks the history and retries instead of ending.
* **A refusal needs a correction attached.** Told only which keys were legal,
  the model resubmitted the same invented recipe three times. Given back a
  repaired recipe it could resend verbatim, it recovered on the next call.
* **Leftover time belongs to the autopilot.** Whatever ends the agent, the
  remaining minutes still buy experiments, and a worse result is refused anyway.

After the fixes it ran the full 25 steps, and its last action — `tune_recipe` —
produced the best result of the run: **RMSE 42.71** against the automatic floor
of 42.80.

It also showed the limit of a 1.7B: from step 11 it oscillated between two
recipes it had already run, fourteen times. The ledger refused every repeat, so
nothing was corrupted and no time was wasted on refits, but a bare "you already
ran this" reads to a weak model as "try the adjacent one". A duplicate refusal
now escalates and names the untried axes instead.

## Measured on this hardware

Eight cores, no GPU — the same shape as the grading container.

| | regression (53k × 64) | multiclass (88k × 44) |
|---|---|---|
| load + cache | 2.3 s | 1.0 s |
| 5-fold LightGBM | ~6 s | ~85 s |
| bootstrap floor | RMSE 42.80 | — |

`n_jobs=-1` is a trap here: cross-validation already iterates folds, so letting
each booster claim every core oversubscribes them. Measured, an identical 5-fold
LightGBM run took **47 s at `n_jobs=-1` and 2.2 s at `n_jobs=4`**, for the same
score. Workers are capped at half the core count throughout.

---

## Open questions for the organisers

* Exact launch command — `python solution.py task.md`, as in `baseline.py`?
* Is there a step limit as well as the time limit? `baseline.py` hard-codes 100.
* Token budget per run inside the container.
* `baseline.py` runs `run_python` with `env={"PATH": os.defpath}`, so the API key
  is not inherited by the subprocess. Deliberate, or an artefact of the example?
