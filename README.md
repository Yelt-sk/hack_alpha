# MLForge

An MCP server that lets an LLM agent build tabular ML models — by describing experiments, not by writing code.

Hackathon track: *teach an AI agent to build and improve ML solutions.*

## The idea being tested

Three pieces, as proposed:

1. **MCP space over an open-source ML framework** — a narrowed, opinionated surface over scikit-learn / LightGBM / CatBoost.
2. **RAG over a framework guide** — `search_knowledge`, retrieving a curated playbook.
3. **A system prompt** explaining the agent's job — `prompts/system.md`.

What the build changed about the idea:

**The agent does not write training code.** It emits a declarative JSON *recipe*; the server validates it, compiles it into a leak-safe sklearn `Pipeline`, cross-validates it honestly and scores it. This is what makes a small local model viable — the hard parts (fold-safe encoding, CV scheme, holdout hygiene) live in the server, not in the model's head.

**The RAG corpus is decision rules, not API documentation.** A capable model already knows sklearn better than any retrieved doc chunk; dumping upstream docs adds noise. What it does *not* know is which choices this server rewards. So `knowledge/` is a hand-written playbook: which encoder for which model, which metric for which task, how to react to each diagnosis.

**Honesty is enforced, not requested.** The holdout is split at load time and never exposed. Every fitted transform lives inside the `Pipeline`, so it is refitted per fold. Guard rails refuse suspicious actions outright, and an override demands a written justification that is permanently attached to the run and reproduced in the final report.

## Setup

The virtualenv is not committed — it is machine-specific. Recreate it:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tests/test_honesty.py      # 21 assertions; confirms the install
```

To attach an MCP client, copy `mcp_config.json` and replace `<MLFORGE_DIR>` with the
absolute path to your checkout. Nothing else in the repository contains a local path.

## Quick start

```bash
.venv/bin/python scripts/demo_agent.py   # scripted walkthrough of every tool, no LLM
.venv/bin/python scripts/smoke_mcp.py    # same server over real MCP stdio
.venv/bin/python tests/test_honesty.py   # 21 assertions on the anti-cheating claims
```

`demo_agent.py` calls the tool functions directly — it proves the ML. `smoke_mcp.py` connects
as an MCP client over stdio, so it proves the protocol: 12 tools discovered, guards firing
over the wire, the system prompt delivered as server instructions.

Point an MCP client at the server with `mcp_config.json`, or drive it with a local model:

```bash
.venv/bin/pip install openai
.venv/bin/python scripts/run_agent.py --base-url http://localhost:11434/v1 --model qwen2.5:14b-instruct --dataset synthetic_churn
```

## The twelve tools

| Tool | Purpose |
|---|---|
| `list_datasets` | What can be loaded |
| `load_dataset` | Load, lock away a holdout, report quality issues |
| `profile_dataset` | Column profile and the full issue list (training split only) |
| `search_knowledge` | BM25 over the curated playbook |
| `get_recipe_schema` | Every field, every allowed value, worked examples |
| `validate_recipe_dry_run` | Check a recipe without training it |
| `run_experiment` | Train, cross-validate, score, diagnose |
| `tune_hyperparams` | Optuna over the registered search space |
| `inspect_run` | Feature importances, per-fold detail |
| `leaderboard` | All runs ranked, with deltas |
| `final_evaluate` | The locked holdout — budget-limited, logged |
| `export_solution` | Standalone script, model artifact, report |

Twelve is a ceiling, not a target. Every extra tool is a way for a small model to get lost.

## Guard rails

Three tiers, because a uniform hard block is worse than useless — it bricks the agent on false positives.

**block** — invalid whatever the intent. A feature built from the target; a flagged leak still in the feature set; random folds on data declared grouped or time-ordered; dropping every feature; spending holdout budget that is gone; calling `final_evaluate` before two real candidates exist.

**quarantine** — refused until the agent passes `override_guards` *and* a justification of at least 20 characters, which is attached to the run permanently and reproduced in the final report.

**taint** — applied after the fact. A score beyond the plausibility threshold (`roc_auc ≥ 0.995`, `r2 ≥ 0.999`, …) marks the run and excludes it from the leaderboard's best until explained.

The goal is not an obedient agent. It is that silent self-deception becomes impossible: anything questionable either does not run, or runs with its reasoning on the record.

### Why the agent cannot waive a leak flag

The leak guard was originally overridable. The live run with a local 1.7B model killed that design in one step: blocked, the model simply wrote *"this column is known to be available at prediction time and is not a leak"* — a fabrication — and walked straight through.

Whether a column is knowable at prediction time is a fact about the business process, not something inferable from the data. An agent cannot establish it, so it may not certify it. The override now belongs to the operator, who passes `trusted_columns=[...]` to `load_dataset`; the agent can only ask for it.

The taint layer did catch that run afterwards — `final_evaluate` was blocked — but only because 1.000 is obviously absurd. A subtler leak scoring 0.93 would have passed both layers. Defence in depth is not a substitute for putting the decision in the right hands.

## Keeping a weak model in the loop

A local **Qwen3-1.7B** (llama.cpp, 12k context, 4 GB GPU) drives the full loop unaided:

```
list_datasets -> load_dataset -> profile_dataset -> search_knowledge
-> dummy (0.500) -> BLOCKED: leak not dropped -> clean lightgbm (0.721)
-> also drop the id column -> tune (0.751) -> holdout (0.745) -> export
```

Its closing summary is accurate: *"ROC AUC 0.75 cross-validated and 0.7453 on the holdout... not overfitting... not leaking, the critical leakage column was dropped."* The 0.751 matches what the scripted expert path reaches (0.749).

The model did not change between the first attempt and this one — the server did. Four server-side affordances, each added in response to an observed failure rather than anticipated:

- **One tool call per turn** (`--parallel-tools` to opt out). The first attempt emitted the entire eighteen-call workflow in a single turn, before any result existed, and trained nothing.

- **Shape repair.** The 1.7B reliably emitted `{"type": "lightgbm", "learning_rate": 0.05}` instead of the nested form, burned eighteen calls on the resulting errors and never recovered. The intent is unambiguous, so `validate_recipe` now repairs flattened recipes and returns a loud warning — repaired, never silent.
- **Duplicate refusal.** It re-submitted an identical recipe four times in nine steps. Recipes are fingerprinted (ignoring `name`/`notes`), and a repeat is refused with a pointer to the run that already scored it.
- **`next_step`.** Every result now names one concrete next action. A capable model infers this from the diagnosis; a small one does not, and naming it is what turns a repeat into a search.

- **A corrected recipe inside every refusal.** This was the one that mattered most. Told in words to add a column to `drop_columns`, the model took ten consecutive blocks and never complied; handed the repaired recipe object with `"next_step": "send this back"`, it recovered on the next call. For a small model, a refusal without an attached fix is a dead end.

The runner also compacts tool JSON before truncating — pretty-printing costs roughly a third of the tokens in whitespace, which matters against a 12k window.

The general lesson: *"the agent can't do it"* usually means *"the tool was designed for a stronger model."* Competence moves from the model into the server, and the bar on the model drops sharply.

## Leak detection

`load_dataset` measures each column's **out-of-sample** single-feature predictive power: fit a one-column mapping on half the training split, score it on the other half. Columns above 0.95 are flagged critical.

The obvious implementation — mutual information over the full frame — is wrong here, and instructively so. It is inflated by cardinality, so a unique `customer_id` scores a perfect 1.0 and the detector cries leak on every ID column. Out-of-sample, an ID generalises to nothing and scores ~0, while a column recorded after the outcome still scores ~1.

## Honesty by construction

- The holdout is split at load time; `profile_dataset` and every diagnostic read the training split only.
- Imputation statistics, rare-category lists, winsorisation bounds and target encodings all live inside the `Pipeline`, so `cross_validate` refits them per fold. There is no recipe field that would let the agent leak across folds.
- Target encoding uses sklearn's internally cross-fitted `TargetEncoder`.
- `final_evaluate` is capped at 3 calls per dataset, every call logged with its justification.
- Optuna only ever sees cross-validation scores.

## Repository layout

```
.
├── server.py                    MCP server: the 12 tools, guard wiring, next-step hints   (714 lines)
├── requirements.txt             pinned runtime dependencies
├── mcp_config.json              MCP client stanza; <MLFORGE_DIR> is a placeholder
├── README.md                    this file
├── REPORT.md                    write-up of the build: findings, experiments, conclusions
│
├── prompts/
│   └── system.md                the agent's system prompt                                  (39 lines)
│
├── knowledge/                   the RAG corpus — 8 files, 45 chunks of decision rules
│   ├── 00-workflow.md             the loop, what counts as a real improvement, when to stop
│   ├── 10-tasks-and-metrics.md    metric per task, validation scheme per data shape
│   ├── 20-data-quality.md         leakage, identifiers, missingness, outliers, imbalance
│   ├── 30-preprocessing.md        encoder decision table, scaling, feature engineering
│   ├── 40-models.md               which model when, and how to read the comparison
│   ├── 50-tuning.md               what actually moves the metric; how not to overfit CV
│   ├── 60-diagnosis.md            each diagnosis line mapped to a concrete next action
│   └── 70-recipes.md              ready-to-adapt recipe templates
│
├── mlforge/                     the library the server is a thin shell over
│   ├── schema.py                recipe vocabulary, validation, shape repair, model registry (479 lines)
│   ├── pipeline.py              recipe -> leak-safe sklearn Pipeline                        (420 lines)
│   ├── train.py                 cross-validation, metrics, diagnosis                        (144 lines)
│   ├── guards.py                the three-tier gate and the plausibility check              (214 lines)
│   ├── data.py                  loading, profiling, leak detection, holdout custody         (354 lines)
│   ├── rag.py                   BM25 retriever, stdlib only                                 (139 lines)
│   └── session.py               datasets, runs, leaderboard, holdout budget, fingerprints   (155 lines)
│
├── scripts/
│   ├── demo_agent.py            scripted walkthrough of every tool, no LLM                  (102 lines)
│   ├── smoke_mcp.py             protocol check over real MCP stdio                          (75 lines)
│   └── run_agent.py             LLM loop against any OpenAI-compatible endpoint             (174 lines)
│
├── tests/
│   └── test_honesty.py          21 assertions on leak detection, guards, fold safety        (304 lines)
│
└── workspace/                   (gitignored) run state and exported solutions
```

### Where to look first

| If you want to understand… | Read |
|---|---|
| what the agent is allowed to say | `mlforge/schema.py` — the recipe vocabulary is the whole interface |
| why it cannot leak | `mlforge/pipeline.py` — every fitted transform is inside the `Pipeline` |
| why it cannot cheat | `mlforge/guards.py`, then `tests/test_honesty.py` |
| what it is told to do | `prompts/system.md` and `knowledge/00-workflow.md` |
| what actually happened | `REPORT.md` |

## Known limits

- Tabular supervised learning only — no CV, NLP or time-series forecasting beyond `time_series` CV.
- No ensembling or stacking across runs; each recipe is a single pipeline.
- The retriever is lexical. For this corpus that is a feature; for a much larger one, embeddings would start to pay off.
- Driving the loop well still needs a capable model. A local Qwen3-1.7B (llama.cpp, 12k context) operates the tools correctly but does not conduct a search: it reaches the right *conclusion* ("this score is implausible, the column is a leak") while failing to act on it.
- `tune_hyperparams` can overrun its budget by roughly the length of one trial (observed: 105s against a 90s budget). It reports the actual `search_seconds`.
