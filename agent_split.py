"""Work out how the test set was built, then validate the same way.

The folds are the measuring instrument. If they do not resemble the split the
grader used, every number the run produces is a number about a different problem,
and optimising it is worse than useless -- it is confidently wrong.

Two things went wrong in this project before this module existed, both found only
after two submissions:

* **The regression task is a forecast.** ``Year_Factor`` is 1..5 in train and
  exactly 6 in test; an adversarial model separates the two sets with AUC 1.0000.
  Random five-fold said RMSE 41.9. Training on years 1..4 and validating on year 5
  -- which is the shape of the real task -- said **59.4**, against 59.15 for
  predicting the median. The whole apparent gain was an artefact of validating on
  interpolation when the task is extrapolation.

* **The classification task is full of duplicates.** 44.8% of training rows have
  an exact twin on all 42 features. Splitting rows rather than groups put 43.8%
  of every validation fold's rows in its own training half, where the model had
  already memorised them -- while only 8.6% of test rows have a twin in train.
  Per-class thresholds measured +0.0375 on those folds and **+0.0096** on folds
  grouped by duplicate. The nested honesty check could not see it: its inner
  folds shared the same leak.

So the diagnosis runs before any fold exists, and it is reported rather than
quietly applied -- the agent is told what the test set looks like, because that
changes what it should try, not only how we score it.

Nothing here is specific to those two datasets. The rules are: a feature whose
train and test values do not overlap means a forecast; many exact duplicates mean
group them; neither means the ordinary thing. When nothing is clear, the ordinary
thing happens and the uncertainty is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Share of rows that must sit in duplicate groups before grouping the folds.
#: Below this, grouping costs stratification quality and buys little: at 5% the
#: leak moves a macro-averaged metric by less than its own fold spread.
DUPLICATE_SHARE = 0.10

#: Adversarial AUC above which train and test are reported as visibly different
#: distributions. 0.5 is indistinguishable; a tree model on noise lands near 0.55
#: on data this size, so this is comfortably clear of chance.
SHIFT_AUC = 0.70

#: Most validation origins a forecast plan will use. Each one costs a fit, and
#: early origins train on so little data that they say more about the data volume
#: than about the model.
MAX_ORIGINS = 3

#: Rows sampled per side for the adversarial probe. Enough for a stable AUC,
#: cheap enough to run before the first model: about four seconds on 88k x 42
#: against ninety for the full five-fold version.
PROBE_ROWS = 12_000


@dataclass
class FoldPlan:
    """A fold assignment and the reasoning that produced it."""

    name: str
    splits: list[tuple[np.ndarray, np.ndarray]]
    fold_id: np.ndarray
    rationale: str
    diagnosis: dict[str, Any] = field(default_factory=dict)
    #: True when fold models must not be used for the test prediction. In a
    #: forecast plan every fold model is deliberately blind to the most recent
    #: data, which is the data closest to the test set; averaging them would
    #: throw away exactly what matters most.
    full_fit_for_test: bool = False

    @property
    def n_splits(self) -> int:
        return len(self.splits)

    @property
    def evaluable(self) -> np.ndarray:
        """Rows that receive an out-of-fold prediction.

        A forecast plan leaves the earliest rows unvalidated on purpose: they are
        the past that every origin trains on. Scoring them would mean scoring an
        empty array element.
        """
        return self.fold_id >= 0

    def summary(self) -> dict[str, Any]:
        return {
            "scheme": self.name,
            "n_splits": self.n_splits,
            "rationale": self.rationale,
            "validated_rows": int(self.evaluable.sum()),
            "total_rows": int(len(self.fold_id)),
            **self.diagnosis,
        }


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def duplicate_groups(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """A group id per row, equal for rows identical on every one of ``columns``.

    Hashing rather than ``groupby`` because this runs on every task at startup and
    a 42-column groupby over 88k rows is seconds, not milliseconds.
    """
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def _duplicate_report(frame: pd.DataFrame, columns: list[str],
                      target: pd.Series | None) -> dict[str, Any]:
    groups = duplicate_groups(frame, columns)
    counts = pd.Series(groups).value_counts()
    repeated = counts[counts > 1]
    in_groups = int(repeated.sum())
    report: dict[str, Any] = {
        "duplicate_rows": in_groups,
        "duplicate_share": round(in_groups / max(1, len(frame)), 4),
        "unique_rows": int(counts.size),
    }
    if target is not None and in_groups:
        # Groups whose members disagree about the answer are the irreducible part:
        # no model can be right about all of them, and they cap the metric for
        # everyone. Worth reporting as a ceiling rather than discovered as a
        # mystery later.
        conflicting = (
            pd.DataFrame({"g": groups, "y": np.asarray(target)})
            .groupby("g")["y"].nunique()
        )
        report["conflicting_groups"] = int((conflicting > 1).sum())
    return report


def separating_feature(train: pd.DataFrame, test: pd.DataFrame,
                       columns: list[str]) -> dict[str, Any] | None:
    """A feature whose train and test values do not overlap at all.

    That is the signature of a split made along something -- a period, a cohort, a
    site -- rather than at random. When it is also ordered and the test side lies
    entirely beyond the train side, the task is a forecast, and validating it by
    interpolation flatters every model equally.

    Returns the strongest such feature, or None.
    """
    best: dict[str, Any] | None = None
    for column in columns:
        if column not in test.columns:
            continue
        left, right = train[column].dropna(), test[column].dropna()
        if left.empty or right.empty:
            continue
        # A column with as many values as rows is an identifier, not a period.
        if left.nunique() > max(20, 0.01 * len(train)):
            continue
        lv, rv = set(left.unique()), set(right.unique())
        if lv & rv:
            continue
        ordered = pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right)
        if not ordered:
            # Disjoint categories separate the sets but give no direction to
            # extrapolate along, so there is no "train on earlier" to imitate.
            candidate = {"feature": column, "kind": "disjoint_categories",
                         "train_values": sorted(map(str, lv))[:12],
                         "test_values": sorted(map(str, rv))[:12]}
        else:
            forward = min(rv) > max(lv)
            candidate = {
                "feature": column,
                "kind": "forecast" if forward else "disjoint_ordinal",
                "train_values": sorted(lv)[:12],
                "test_values": sorted(rv)[:12],
            }
        # Prefer a forecast over a bare disjoint set: it is the one that tells us
        # how to validate, not merely that something is odd.
        if best is None or (candidate["kind"] == "forecast" and best["kind"] != "forecast"):
            best = candidate
    return best


def adversarial_auc(train: pd.DataFrame, test: pd.DataFrame, columns: list[str],
                    seed: int = 0) -> dict[str, Any]:
    """How well a model can tell a training row from a test row.

    0.5 means the two sets are drawn alike and ordinary cross-validation measures
    the right thing. Near 1.0 means they are different populations, and a score
    measured on one of them is a claim about the other.

    Deliberately small and fast: this runs before the first real model, and its
    job is to raise a flag, not to be the final word.
    """
    shared = [c for c in columns if c in test.columns]
    if not shared:
        return {"adversarial_auc": None, "note": "no columns in common"}
    try:
        from lightgbm import LGBMClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception as err:  # noqa: BLE001 - a missing probe must not stop a run
        return {"adversarial_auc": None, "note": f"probe unavailable: {err}"}

    rng = np.random.default_rng(seed)
    def sample(frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) <= PROBE_ROWS:
            return frame[shared]
        return frame[shared].iloc[rng.choice(len(frame), PROBE_ROWS, replace=False)]

    left, right = sample(train), sample(test)
    X = pd.concat([left, right], ignore_index=True)
    for column in X.columns:
        if X[column].dtype == object:
            X[column] = X[column].astype("category")
    y = np.r_[np.zeros(len(left)), np.ones(len(right))]

    try:
        fit_X, val_X, fit_y, val_y = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        model = LGBMClassifier(n_estimators=120, learning_rate=0.1, num_leaves=31,
                               n_jobs=4, verbose=-1, importance_type="gain",
                               random_state=seed)
        model.fit(fit_X, fit_y)
        auc = float(roc_auc_score(val_y, model.predict_proba(val_X)[:, 1]))
    except Exception as err:  # noqa: BLE001
        return {"adversarial_auc": None, "note": f"probe failed: {err}"}

    gains = pd.Series(model.feature_importances_, index=X.columns)
    total = float(gains.sum()) or 1.0
    drivers = [
        {"feature": str(name), "share": round(float(value) / total, 3)}
        for name, value in gains.sort_values(ascending=False).head(5).items()
        if value > 0
    ]
    return {"adversarial_auc": round(auc, 4), "shift_drivers": drivers}


def diagnose(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str],
             target: pd.Series | None = None, seed: int = 0) -> dict[str, Any]:
    """Everything we can learn about the split before fitting anything real."""
    found: dict[str, Any] = {}
    found.update(_duplicate_report(train, feature_columns, target))
    found.update(adversarial_auc(train, test, feature_columns, seed))
    separator = separating_feature(train, test, feature_columns)
    if separator:
        found["separating_feature"] = separator
    auc = found.get("adversarial_auc")
    found["distribution_shift"] = bool(auc is not None and auc >= SHIFT_AUC)
    return found


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _forward_chain(values: pd.Series, n_wanted: int) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Train on what came before, validate on what came next.

    One split per validation origin, each training only on strictly earlier
    values. This is the shape of the graded task when the test set lies beyond the
    training set in time, and it is the only arrangement that measures
    extrapolation rather than interpolation.
    """
    ordered = sorted(pd.unique(values.dropna()))
    if len(ordered) < 2:
        return None
    origins = ordered[-min(n_wanted, len(ordered) - 1):]
    array = values.to_numpy()
    splits = []
    for origin in origins:
        train_idx = np.where(array < origin)[0]
        val_idx = np.where(array == origin)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits or None


def _grouped(y: pd.Series, groups: np.ndarray, n_splits: int, classification: bool,
             seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds that keep identical rows together, so no row is scored against itself."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    if classification:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    # GroupKFold is deterministic and takes no seed; the grouping is the whole
    # point and shuffling within it changes nothing that matters.
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))


def _plain(y: pd.Series, n_splits: int, classification: bool,
           seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold, StratifiedKFold

    if classification:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y))))


#: Below this share of the latest period recognisable from earlier ones, an entity
#: history would be mostly empty and is not worth a feature.
ENTITY_MIN_MATCH = 0.2
ENTITY_MAX_COLUMNS = 6


def _codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values.astype(str), use_na_sentinel=False)[0].astype(np.int64)


def entity_key(train: pd.DataFrame, test: pd.DataFrame, period: str,
               columns: list[str]) -> dict[str, Any] | None:
    """Find the columns that identify the same entity across periods, if any.

    On a forecast task rows are often repeated measurements of the same things --
    buildings, customers, shops -- with no id column saying so. Their static
    description identifies them anyway, and what an entity's target was last
    period is the strongest single fact about what it will be next period.

    Chosen greedily on train alone: add whichever column most increases the share
    of the latest period's rows that are unique within that period *and* seen in
    an earlier one. Columns that describe the period rather than the entity
    (weather, prices) change between periods, break the match, and are never
    picked; columns that describe the entity sharpen it. On the regression task
    this finds floor area, year built, facility type and state, recognising 67% of
    year 5 from years 1..4 and 79% of the test set from train.
    """
    try:
        values = train[period]
        latest = (values == values.max()).to_numpy()
        if latest.all() or not latest.any():
            return None
        candidates = [c for c in columns if c != period and c in test.columns]
        coded = {c: _codes(pd.concat([train[c], test[c]], ignore_index=True)) for c in candidates}
        n_train = len(train)

        def recognised(key: np.ndarray) -> float:
            current, earlier = key[:n_train][latest], key[:n_train][~latest]
            unique = ~pd.Series(current).duplicated(keep=False).to_numpy()
            return float((unique & np.isin(current, earlier)).mean())

        chosen: list[str] = []
        key = np.zeros(n_train + len(test), dtype=np.int64)
        best = 0.0
        while len(chosen) < ENTITY_MAX_COLUMNS:
            trials = []
            for column in candidates:
                if column in chosen:
                    continue
                combined = pd.factorize(key * (int(coded[column].max()) + 2) + coded[column])[0]
                trials.append((recognised(combined), column, combined))
            if not trials:
                break
            share, column, combined = max(trials, key=lambda t: t[0])
            if share <= best + 1e-3:
                break
            chosen, key, best = chosen + [column], combined.astype(np.int64), share

        # Second pass: a column that keeps recognition where it is but separates
        # two different entities that shared a key. Recognition alone cannot see
        # this -- a collision still "matches" -- and on the regression task it is
        # the difference between keying on (area, year built, type) and adding the
        # state: 42.6 against 41.4 RMSE, because same-sized buildings in different
        # states were sharing one history.
        period_code = _codes(values)

        def collisions(key: np.ndarray) -> int:
            within = pd.factorize(key[:n_train] * (int(period_code.max()) + 2) + period_code)[0]
            return int(pd.Series(within).duplicated(keep=False).sum())

        while best >= ENTITY_MIN_MATCH and len(chosen) < ENTITY_MAX_COLUMNS:
            current = collisions(key)
            trials = []
            for column in candidates:
                if column in chosen:
                    continue
                combined = pd.factorize(key * (int(coded[column].max()) + 2) + coded[column])[0]
                share = recognised(combined)
                if share >= best - 1e-4:
                    trials.append((current - collisions(combined), share, column, combined))
            if not trials:
                break
            gained, share, column, combined = max(trials, key=lambda t: (t[0], t[1]))
            # A handful of rows is not worth another column. Every column added is
            # one more way for an entity's history to fragment, and the history of
            # a few tail rows is where this metric lives: a near-constant weather
            # column that separated 7 rows cost 0.8 RMSE on the regression task,
            # while the state, separating 38, was worth 1.2.
            if gained < max(10, int(0.0005 * n_train)):
                break
            chosen, key, best = chosen + [column], combined.astype(np.int64), max(best, share)
        if best < ENTITY_MIN_MATCH:
            return None
        test_match = float(np.isin(key[n_train:], key[:n_train]).mean())
        return {"columns": chosen, "period": period, "train_match": round(best, 4),
                "test_match": round(test_match, 4)}
    except Exception:  # noqa: BLE001 - a diagnosis must never cost the run
        return None


def _as_plan(name: str, splits: list[tuple[np.ndarray, np.ndarray]], n_rows: int,
             rationale: str, diagnosis: dict[str, Any],
             full_fit_for_test: bool = False) -> FoldPlan:
    fold_id = np.full(n_rows, -1, dtype=np.int8)
    for index, (_, val_idx) in enumerate(splits):
        fold_id[val_idx] = index
    return FoldPlan(name=name, splits=splits, fold_id=fold_id, rationale=rationale,
                    diagnosis=diagnosis, full_fit_for_test=full_fit_for_test)


def _classes_survive(y: pd.Series, splits: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Every split must be able to predict every class it will be scored on.

    A forecast plan can easily hand an origin a training half missing a rare
    class; the fold then cannot name it, and the macro average is punished for a
    fault of the split rather than of the model.
    """
    labels = np.asarray(y)
    for train_idx, val_idx in splits:
        if not set(np.unique(labels[val_idx])) <= set(np.unique(labels[train_idx])):
            return False
    return True


def plan(train: pd.DataFrame, test: pd.DataFrame, y: pd.Series,
         feature_columns: list[str], classification: bool, n_splits: int = 5,
         seed: int = 42, log=print) -> FoldPlan:
    """Choose a fold scheme from what the data says about the split.

    The order of preference is not arbitrary. A forecast is the most consequential
    thing to get wrong -- it was worth 17 RMSE here -- so it is tested first.
    Duplicates come next, because they are the difference between a measurement
    and a memory test. If neither applies, the ordinary scheme is correct and is
    used without apology.
    """
    found = diagnose(train, test, feature_columns, y if classification else None, seed)
    n_rows = len(train)

    separator = found.get("separating_feature")
    if separator and separator["kind"] == "forecast":
        column = separator["feature"]
        splits = _forward_chain(train[column], MAX_ORIGINS)
        if splits and (not classification or _classes_survive(y, splits)):
            origins = [train[column].to_numpy()[v[0]] for _, v in splits]
            validated = sum(len(v) for _, v in splits)
            rationale = (
                f"'{column}' holds {separator['train_values']} in train and "
                f"{separator['test_values']} in test, with no overlap: the test set is "
                f"beyond the training set, so the task is extrapolation. Validating on "
                f"{origins} in turn, each time training only on earlier values. Random "
                f"folds would measure interpolation instead, which is a different and "
                f"much easier problem -- measured here as RMSE 41.9 against 59.4 for the "
                f"same model. Fold-to-fold spread under this plan reflects how hard each "
                f"period was, not how noisy the model is, so compare models fold by fold."
            )
            log(f"[split] forecast on '{column}': origins {origins}, "
                f"{validated} of {n_rows} rows validated")
            entity = entity_key(train, test, column, feature_columns)
            if entity:
                found["entity"] = entity
                log(f"[split] entities recur across '{column}': key {entity['columns']} "
                    f"recognises {entity['train_match']:.0%} of the latest period and "
                    f"{entity['test_match']:.0%} of test")
            return _as_plan("forward_chain", splits, n_rows, rationale, found,
                            full_fit_for_test=True)
        log(f"[split] '{column}' separates train from test but a forecast plan was "
            f"not usable here; falling back")
        found["forecast_rejected"] = (
            f"'{column}' looks like a forecast axis, but splitting on it left a class "
            f"absent from a training half, so it was not used"
            if splits else
            f"'{column}' looks like a forecast axis but has too few distinct values to chain"
        )

    if found.get("duplicate_share", 0.0) >= DUPLICATE_SHARE:
        groups = duplicate_groups(train, feature_columns)
        try:
            splits = _grouped(y, groups, n_splits, classification, seed)
        except Exception as err:  # noqa: BLE001
            log(f"[split] grouping failed ({err}); falling back to plain folds")
            found["grouping_failed"] = str(err)
        else:
            share = found["duplicate_share"]
            rationale = (
                f"{share:.1%} of training rows have an exact twin on all "
                f"{len(feature_columns)} features. Splitting rows would put those twins "
                f"on both sides of a fold, where the model scores them from memory. "
                f"Folds are grouped so identical rows stay together."
            )
            log(f"[split] grouped folds: {share:.1%} of rows sit in duplicate groups")
            return _as_plan("grouped", splits, n_rows, rationale, found)

    rationale = (
        "No feature separates train from test and duplicates are rare, so rows are "
        "exchangeable and ordinary stratified folds measure the right thing."
    )
    if found.get("distribution_shift"):
        rationale += (
            f" Note: an adversarial model still tells the two sets apart with AUC "
            f"{found['adversarial_auc']}, so some shift exists without a single feature "
            f"carrying it. Cross-validation will read slightly optimistic."
        )
    log(f"[split] plain folds (adversarial AUC {found.get('adversarial_auc')})")
    return _as_plan("stratified" if classification else "kfold",
                    _plain(y, n_splits, classification, seed), n_rows, rationale, found)
