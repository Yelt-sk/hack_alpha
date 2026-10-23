"""Understanding the task from its files, with nothing hard-coded.

Two of the four scored tasks are never shown to us: the platform re-runs this
same code against them. So every fact about a task has to be *derived*, and the
derivations have to prefer signals that are structural rather than textual.

The strongest such signal is ``sample_submission.csv``. Its header is literally
``<id>,<target>`` -- it names the identifier and the target column without any
parsing of prose, in any language, for any task. Everything else keys off that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

TaskKind = Literal["regression", "binary", "multiclass"]

#: Canonical metric name -> (direction, patterns matched against the task text).
METRICS: dict[str, tuple[str, tuple[str, ...]]] = {
    # RMSLE first in spirit: "root mean squared log error" must not read as RMSE,
    # whose pattern therefore refuses a following "log".
    "rmsle": ("minimize", (r"\brmsle\b", r"root[-\s]mean[-\s]squared[-\s]log",
                           r"squared[-\s]log(?:arithmic)?[-\s]error", r"\bmsle\b")),
    "rmse": ("minimize", (r"\brmse\b", r"root[-\s]mean[-\s]squared(?![-\s]log)")),
    "mse": ("minimize", (r"\bmse\b", r"(?<!root )mean[-\s]squared[-\s]error")),
    "smape": ("minimize", (r"\bsmape\b", r"symmetric mean absolute percentage")),
    "mae": ("minimize", (r"\bmae\b", r"mean absolute error")),
    "mape": ("minimize", (r"(?<!s)\bmape\b", r"(?<!symmetric )mean absolute percentage")),
    "r2": ("maximize", (r"\br2\b", r"r\^?2\b", r"коэффициент детерминации")),
    "macro_f1": ("maximize", (r"macro[-\s]?f1", r"f1[-\s]?macro", r"macro[-\s]averag\w*\s+f1",
                              r"f1[-\s]?\(\s*macro", r"f1[-\s]score\s*\(\s*macro")),
    "weighted_f1": ("maximize", (r"weighted[-\s]?f1",)),
    "f1": ("maximize", (r"\bf1\b",)),
    "accuracy": ("maximize", (r"\baccuracy\b", r"\bточность\b")),
    "roc_auc": ("maximize", (r"roc[-\s]?auc", r"\bauc\b")),
    "logloss": ("minimize", (r"log[-\s]?loss", r"логарифмическая функция потерь")),
}

DEFAULT_METRIC: dict[str, str] = {
    "regression": "rmse",
    "binary": "roc_auc",
    "multiclass": "macro_f1",
}

#: Canonical roles -> the filenames that usually carry them.
_FILE_ROLES: dict[str, tuple[str, ...]] = {
    "train": ("train.csv",),
    "test": ("test.csv",),
    "dictionary": ("data_dictionary.csv", "dictionary.csv"),
    "sample_submission": ("sample_submission.csv", "submission_example.csv"),
}


@dataclass
class TaskSpec:
    """Everything the rest of the system needs to know about the task."""

    task_path: Path
    task_dir: Path
    text: str
    files: dict[str, Path]
    id_column: str
    target_column: str
    kind: TaskKind
    metric: str
    direction: str
    classes: list[Any] | None = None
    n_train: int = 0
    n_test: int = 0
    warnings: list[str] = field(default_factory=list)
    #: The answer column holds P(positive class) rather than a label. Only for a
    #: binary task scored on probabilities whose own files say so -- writing
    #: probabilities where labels are expected fails validation outright.
    wants_probability: bool = False

    @property
    def submission_path(self) -> Path:
        """Where the answer goes: next to the task markdown, per the task statement."""
        return self.task_dir / "submission.csv"

    @property
    def is_classification(self) -> bool:
        return self.kind in ("binary", "multiclass")

    def summary(self) -> dict[str, Any]:
        """Compact description for the model. Paths are relative -- absolute ones
        waste tokens and tell the model nothing it can use."""
        return {
            "kind": self.kind,
            "metric": self.metric,
            "direction": self.direction,
            "id_column": self.id_column,
            "target_column": self.target_column,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_classes": len(self.classes) if self.classes else None,
            "classes": self.classes if self.classes and len(self.classes) <= 25 else None,
            "files": {k: str(v.relative_to(self.task_dir)) for k, v in self.files.items()},
            "submission_path": str(self.submission_path.relative_to(self.task_dir)),
            "warnings": self.warnings,
            "answer_format": ("probability of the positive class "
                              f"'{self.classes[-1] if self.classes else 1}'"
                              if self.wants_probability else "label"),
        }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_files(task_dir: Path, text: str) -> tuple[dict[str, Path], list[str]]:
    """Locate the task's CSVs.

    The public tasks already disagree with themselves here: the markdown quotes
    ``multiclass/participants/train.csv`` while the delivered archive is flat. So
    trust neither source alone -- take paths mentioned in the text when they
    exist, and fall back to a recursive search by filename.
    """
    warnings: list[str] = []
    mentioned = {Path(m) for m in re.findall(r"[\w./\\-]+\.csv", text)}
    found: dict[str, Path] = {}

    for role, names in _FILE_ROLES.items():
        candidates: list[Path] = []

        for rel in mentioned:
            if rel.name in names:
                full = (task_dir / rel).resolve()
                if full.is_file():
                    candidates.append(full)

        if not candidates:
            for name in names:
                candidates.extend(sorted(task_dir.rglob(name)))

        if candidates:
            # Shallowest wins: a nested copy is more likely to be an artifact.
            found[role] = min(candidates, key=lambda p: (len(p.parts), str(p)))
        elif role in ("train", "test"):
            raise FileNotFoundError(
                f"could not locate {role}.csv under {task_dir}. "
                f"CSV files present: {[str(p.relative_to(task_dir)) for p in task_dir.rglob('*.csv')][:20]}"
            )
        else:
            warnings.append(f"{role} file not found; continuing without it")

    return found, warnings


# ---------------------------------------------------------------------------
# Target, kind, metric
# ---------------------------------------------------------------------------


def _id_and_target(files: dict[str, Path], train_columns: list[str]) -> tuple[str, str]:
    """Read the identifier and target column names off the submission template."""
    sample = files.get("sample_submission")
    if sample is not None:
        header = pd.read_csv(sample, nrows=0).columns.tolist()
        if len(header) >= 2:
            return header[0], header[-1]

    # No template: the target is whatever train has and test does not.
    raise ToolMissingTarget(
        "sample_submission.csv is absent, so the target column cannot be read from it. "
        f"Train columns: {train_columns[:40]}"
    )


class ToolMissingTarget(RuntimeError):
    """Raised when the target cannot be determined; the caller decides what to do."""


def _infer_kind(y: pd.Series, text: str) -> TaskKind:
    lowered = text.lower()
    n_unique = int(y.nunique(dropna=True))
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    says_regression = bool(re.search(r"регресси|regress", lowered))
    says_classification = bool(re.search(r"классификац|classif", lowered))

    # The text decides when it is unambiguous; the target's shape alone cannot.
    # A 1..5 rating is a regression when the task says so (dtype-only rules made
    # it ten-way classification, and the MAE metric then had nowhere to go), and
    # labels coded 1.0 .. 10.0 are classes when the task says so.
    if numeric and says_regression and not says_classification and n_unique > 2:
        return "regression"
    if says_classification and not says_regression and n_unique >= 2:
        integral = numeric and bool((y.dropna() % 1 == 0).all())
        if n_unique == 2:
            return "binary"
        if n_unique <= max(50, int(0.01 * len(y))) and (not numeric or integral):
            return "multiclass"

    if says_regression and n_unique > 20:
        return "regression"
    if pd.api.types.is_float_dtype(y) and n_unique > 20:
        return "regression"
    if n_unique == 2:
        return "binary"
    if n_unique <= max(50, int(0.01 * len(y))) and not pd.api.types.is_float_dtype(y):
        return "multiclass"
    return "regression"


#: Metric spelled as code. This is the strongest evidence available: a statement
#: that shows how the score is computed cannot be misread, and the signal does
#: not depend on the language the prose is written in.
_CODE_SIGNALS: tuple[tuple[str, str], ...] = (
    (r"f1_score\([^)]*average\s*=\s*[\"']macro[\"']", "macro_f1"),
    (r"f1_score\([^)]*average\s*=\s*[\"']weighted[\"']", "weighted_f1"),
    (r"f1_score\([^)]*average\s*=\s*[\"']binary[\"']", "f1"),
    (r"root_mean_squared_error|sqrt\s*\(\s*mean|mean_squared_error[^)]*\)\s*\*\*\s*0?\.5", "rmse"),
    (r"mean_absolute_percentage_error", "mape"),
    (r"mean_squared_log_error|root_mean_squared_log_error", "rmsle"),
    (r"mean_absolute_error", "mae"),
    (r"\br2_score\b", "r2"),
    (r"roc_auc_score", "roc_auc"),
    (r"\blog_loss\b", "logloss"),
    (r"accuracy_score", "accuracy"),
)

#: Phrases that declare the primary metric. Both orders occur in the public
#: tasks: "Основная метрика — Macro F1" and "основной метрикой выбран Macro F1".
_PRIMARY_DECLARATIONS: tuple[str, ...] = (
    r"основн\w*\s+метрик\w*[^.\n]{0,80}",
    r"метрика\s+оценки[^.\n]{0,80}",
    r"(?:main|primary|evaluation)\s+metric[^.\n]{0,80}",
    r"[^.\n]{0,80}выбран\w*\s+основн\w*",
)

#: Phrases marking a metric as secondary. Without this, "Дополнительно
#: рассчитываются Accuracy и Weighted F1" reads as an endorsement.
_SECONDARY_MARKERS = re.compile(
    r"дополнительн|справочн|не отражают|а не\b|вспомогательн|reference|additionally"
    r"|not used|not for ranking|for information|secondary|also report",
    re.IGNORECASE,
)


def _infer_metric(text: str, kind: TaskKind) -> tuple[str, str, list[str]]:
    """Pick the primary metric, weighing evidence from strongest to weakest.

    Statements name several metrics and mark one as primary, so presence alone is
    not enough -- the classification task mentions Macro F1, Weighted F1 and
    Accuracy within three lines of each other, and only the first is scored.
    """
    warnings: list[str] = []
    lowered = text.lower()
    valid = _metrics_for(kind)
    evidence: dict[str, list[tuple[int, str]]] = {}

    def add(metric: str, weight: int, reason: str) -> None:
        if metric in valid:
            evidence.setdefault(metric, []).append((weight, reason))

    # 1. How the score is computed, in code.
    code = " ".join(
        re.findall(r"```.*?```", text, flags=re.DOTALL) + re.findall(r"`([^`\n]+)`", text)
    )
    for pattern, metric in _CODE_SIGNALS:
        if re.search(pattern, code, flags=re.IGNORECASE):
            add(metric, 10, "shown as code")

    # 2. An explicit "the primary metric is X" declaration.
    declarations = " ".join(
        m.group(0) for pattern in _PRIMARY_DECLARATIONS for m in re.finditer(pattern, lowered)
    )
    for name, (_, patterns) in METRICS.items():
        if any(re.search(p, declarations) for p in patterns):
            add(name, 6, "declared primary")

    # 3. Emphasis, demoted when the sentence frames the metric as secondary.
    for sentence in re.split(r"(?<=[.!?])\s+|\n\n", lowered):
        secondary = bool(_SECONDARY_MARKERS.search(sentence))
        bolded = " ".join(re.findall(r"\*\*(.+?)\*\*", sentence))
        for name, (_, patterns) in METRICS.items():
            if bolded and any(re.search(p, bolded) for p in patterns):
                add(name, -3 if secondary else 3, "demoted as secondary" if secondary else "bold")
            elif any(re.search(p, sentence) for p in patterns):
                add(name, -2 if secondary else 1, "demoted as secondary" if secondary else "mentioned")

    if not evidence:
        fallback = DEFAULT_METRIC[kind]
        warnings.append(
            f"no metric recognised in the task text; defaulting to '{fallback}' for a {kind} task"
        )
        return fallback, METRICS[fallback][0], warnings

    totals = {name: sum(w for w, _ in items) for name, items in evidence.items()}
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    chosen, top = ranked[0]
    if kind == "multiclass" and chosen == "f1":
        # A bare "F1" on more than two classes has no single positive class; the
        # averaged form every multiclass leaderboard means is macro.
        warnings.append("'F1' on a multiclass task read as macro F1")
        chosen = "macro_f1"

    if len(ranked) > 1 and ranked[1][1] == top:
        warnings.append(
            f"metric is ambiguous between {[n for n, _ in ranked if totals[n] == top]}; using '{chosen}'"
        )
    if top <= 1:
        warnings.append(f"metric '{chosen}' inferred from weak evidence; verify against the task text")

    return chosen, METRICS[chosen][0], warnings


def _metrics_for(kind: TaskKind) -> set[str]:
    if kind == "regression":
        return {"rmse", "mae", "mape", "r2", "rmsle", "mse", "smape"}
    if kind == "binary":
        return {"roc_auc", "f1", "accuracy", "logloss", "macro_f1", "weighted_f1"}
    return {"macro_f1", "weighted_f1", "accuracy", "logloss", "f1"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_task(task_path: str | Path) -> TaskSpec:
    """Read a task markdown file and derive the full specification."""
    task_path = Path(task_path).resolve()
    if not task_path.is_file():
        raise FileNotFoundError(f"task file not found: {task_path}")

    text = task_path.read_text(encoding="utf-8")
    task_dir = task_path.parent
    files, warnings = _discover_files(task_dir, text)

    train_head = pd.read_csv(files["train"], nrows=2000)
    id_column, target_column = _id_and_target(files, train_head.columns.tolist())

    if target_column not in train_head.columns:
        raise RuntimeError(
            f"target '{target_column}' from sample_submission.csv is absent in train.csv "
            f"(columns: {train_head.columns.tolist()[:40]})"
        )

    # The head may miss rare classes, so read the target column in full -- one
    # narrow column is cheap even on the largest task here.
    y = pd.read_csv(files["train"], usecols=[target_column])[target_column]
    kind = _infer_kind(y, text)
    metric, direction, metric_warnings = _infer_metric(text, kind)
    warnings += metric_warnings

    classes = None
    if kind in ("binary", "multiclass"):
        classes = sorted(y.dropna().unique().tolist(), key=str)

    n_test = sum(1 for _ in open(files["test"], encoding="utf-8")) - 1

    wants_probability = False
    if kind == "binary" and metric in ("roc_auc", "logloss"):
        wants_probability = _asks_for_probability(files, target_column, text)
        if wants_probability:
            warnings.append(f"answers are written as P({target_column} = {classes[-1]!r})")

    return TaskSpec(
        task_path=task_path,
        task_dir=task_dir,
        text=text,
        files=files,
        id_column=id_column,
        target_column=target_column,
        kind=kind,
        metric=metric,
        direction=direction,
        classes=classes,
        n_train=len(y),
        n_test=n_test,
        warnings=warnings,
        wants_probability=wants_probability,
    )


def _asks_for_probability(files: dict[str, Path], target: str, text: str) -> bool:
    """Does this binary task want a probability column? Evidence from its own files.

    The sample submission is the strongest witness: values strictly between 0 and
    1 can only be probabilities. Failing that, the statement naming probabilities
    near the answer. A metric that ranks (ROC-AUC) is not enough on its own --
    some graders score AUC on hard labels and reject anything else.
    """
    try:
        sample = pd.read_csv(files["sample_submission"], usecols=[target], nrows=5000)[target]
        values = pd.to_numeric(sample, errors="coerce")
        if values.notna().all() and ((values > 0) & (values < 1)).any():
            return True
    except Exception:  # noqa: BLE001 - an unreadable sample is no evidence either way
        pass
    return bool(re.search(r"вероятност|probabilit", text.lower()))
