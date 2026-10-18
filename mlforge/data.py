"""
Dataset loading, profiling, leakage detection and holdout custody.

The holdout is split here, at load time, and is never handed back to the agent.
profile_dataset and every diagnostic read the *train* frame only. This is the
single most important property of the server: it makes test-set fishing
impossible by construction rather than by instruction.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column kinds. pandas 3 gives string columns a real string dtype, so the old
# select_dtypes('object') trick silently misses them -- classify explicitly.
# --------------------------------------------------------------------------


def column_kinds(df: pd.DataFrame, exclude: list[str] | None = None) -> dict[str, list[str]]:
    exclude = set(exclude or [])
    numeric, categorical, datetime = [], [], []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime.append(c)
        elif pd.api.types.is_bool_dtype(s):
            numeric.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        else:
            categorical.append(c)
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime}


def infer_task(y: pd.Series) -> str:
    """binary / multiclass / regression, with the usual tabular heuristics."""
    if pd.api.types.is_float_dtype(y) and y.nunique(dropna=True) > 20:
        return "regression"
    n = y.nunique(dropna=True)
    if n == 2:
        return "binary"
    if n <= 20 and not pd.api.types.is_float_dtype(y):
        return "multiclass"
    return "regression"


# --------------------------------------------------------------------------
# Built-in catalogue. Offline-first: the synthetic sets and the sklearn bundled
# sets need no network, so a demo never dies on a flaky connection.
# --------------------------------------------------------------------------

CATALOG: dict[str, dict[str, Any]] = {
    "synthetic_churn": {
        "task": "binary", "offline": True,
        "doc": "Synthetic telecom churn: mixed types, missing values, a datetime column, "
               "class imbalance and one deliberately leaky column. Good for exercising the full loop.",
    },
    "breast_cancer": {"task": "binary", "offline": True, "doc": "Wisconsin breast cancer. 569x30, all numeric, easy. Sanity check only."},
    "wine": {"task": "multiclass", "offline": True, "doc": "Wine cultivars. 178x13, 3 classes, all numeric."},
    "digits": {"task": "multiclass", "offline": True, "doc": "8x8 handwritten digits. 1797x64, 10 classes."},
    "diabetes": {"task": "regression", "offline": True, "doc": "Diabetes progression. 442x10, all numeric."},
    "california_housing": {"task": "regression", "offline": False, "doc": "California house prices. 20640x8 numeric. Downloads on first use."},
    "adult": {"task": "binary", "offline": False, "doc": "Census income >50K. ~48k rows, mixed numeric/categorical with missing values. Downloads from OpenML."},
    "credit_g": {"task": "binary", "offline": False, "doc": "German credit risk. 1000x20, heavily categorical, imbalanced. Downloads from OpenML."},
}


def make_synthetic_churn(n: int = 6000, seed: int = 0) -> tuple[pd.DataFrame, str]:
    """A deliberately messy dataset.

    It contains `support_ticket_after_cancel`, which is recorded *after* the
    outcome it is supposed to predict. A competent agent must notice the
    leak-detector warning and drop it; the holdout score collapses if it does not.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = np.round(rng.normal(70, 25, n).clip(15, 200), 2)
    contract = rng.choice(["month-to-month", "one_year", "two_year"], n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["card", "bank_transfer", "e_check", "mailed_check"], n, p=[0.35, 0.25, 0.25, 0.15])
    region = rng.choice([f"region_{i}" for i in range(18)], n)
    support_calls = rng.poisson(1.2, n)

    logit = (
        -1.1
        - 0.035 * tenure
        + 0.013 * monthly
        + np.where(contract == "month-to-month", 1.15, -0.45)
        + np.where(payment == "e_check", 0.55, 0.0)
        + 0.28 * support_calls
        + rng.normal(0, 0.55, n)
    )
    churn = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST{i:06d}" for i in range(n)],
        "signup_date": pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": np.round(tenure * monthly * rng.normal(1.0, 0.05, n), 2),
        "contract": contract,
        "payment_method": payment,
        "region": region,
        "support_calls": support_calls,
        "satisfaction_score": rng.integers(1, 11, n).astype(float),
        # Recorded only after the customer already churned -> pure leakage.
        "support_ticket_after_cancel": np.where(
            churn == 1, rng.choice(["refund", "retention_offer"], n), "none"),
        "churn": churn,
    })
    # Realistic missingness.
    for col, frac in [("total_charges", 0.04), ("satisfaction_score", 0.12), ("payment_method", 0.03)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df, "churn"


def load_builtin(name: str) -> tuple[pd.DataFrame, str]:
    from sklearn import datasets as skd

    if name == "synthetic_churn":
        return make_synthetic_churn()
    if name in ("breast_cancer", "wine", "digits", "diabetes"):
        loader = {"breast_cancer": skd.load_breast_cancer, "wine": skd.load_wine,
                  "digits": skd.load_digits, "diabetes": skd.load_diabetes}[name]
        b = loader(as_frame=True)
        df = b.frame.copy()
        return df, b.target.name or "target"
    if name == "california_housing":
        b = skd.fetch_california_housing(as_frame=True)
        return b.frame.copy(), b.target.name
    if name in ("adult", "credit_g"):
        oml = {"adult": "adult", "credit_g": "credit-g"}[name]
        b = skd.fetch_openml(oml, version=1, as_frame=True)
        df = b.frame.copy()
        return df, b.target.name
    raise ValueError(f"Unknown built-in dataset '{name}'. Available: {sorted(CATALOG)}")


def load_csv(path: str, target: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in CSV. Columns: {list(df.columns)[:40]}")
    # Opportunistically parse anything that looks like a date.
    for c in df.columns:
        if c == target or pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            sample = df[c].dropna().head(200)
            if len(sample) and sample.astype(str).str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > 0.8:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df, target


# --------------------------------------------------------------------------
# Profiling and quality checks
# --------------------------------------------------------------------------


def _target_determination(col: pd.Series, y: pd.Series, task: str, seed: int = 0) -> float:
    """Out-of-sample predictive power of one column, in [0, 1].

    Deliberately *not* mutual information on the full frame: that measure is
    inflated by cardinality, so a unique customer_id scores a perfect 1.0 and
    the detector cries leak on every ID column. Instead we fit a single-feature
    mapping on one half and score it on the other. An ID generalises to nothing
    and scores ~0; a column recorded after the outcome still scores ~1.
    """
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    try:
        n = len(col)
        if n < 50 or y.isna().all():
            return 0.0
        strat = y if task in ("binary", "multiclass") and y.value_counts().min() >= 2 else None
        ia, ib = train_test_split(np.arange(n), test_size=0.5, random_state=seed, stratify=strat)

        if pd.api.types.is_numeric_dtype(col):
            a = col.iloc[ia].astype("float64")
            edges = np.unique(np.nanquantile(a.dropna(), np.linspace(0, 1, 21)))
            if len(edges) < 3:
                return 0.0
            key_a = pd.Series(np.digitize(a, edges), index=a.index).where(a.notna(), -1)
            b = col.iloc[ib].astype("float64")
            key_b = pd.Series(np.digitize(b, edges), index=b.index).where(b.notna(), -1)
        else:
            key_a = col.iloc[ia].astype(str).fillna("__NA__")
            key_b = col.iloc[ib].astype(str).fillna("__NA__")

        ya, yb = y.iloc[ia], y.iloc[ib]

        if task == "regression":
            ya_f, yb_f = ya.astype("float64"), yb.astype("float64")
            table = ya_f.groupby(key_a.to_numpy()).mean()
            pred = pd.Series(key_b.to_numpy()).map(table).fillna(ya_f.mean()).to_numpy()
            ok = ~np.isnan(yb_f.to_numpy())
            return float(max(0.0, r2_score(yb_f.to_numpy()[ok], pred[ok]))) if ok.sum() > 2 else 0.0

        if task == "binary":
            pos = sorted(y.dropna().unique())[-1]
            ya_b = (ya == pos).astype(float)
            table = ya_b.groupby(key_a.to_numpy()).mean()
            pred = pd.Series(key_b.to_numpy()).map(table).fillna(ya_b.mean()).to_numpy()
            yb_b = (yb == pos).astype(int).to_numpy()
            if len(np.unique(yb_b)) < 2:
                return 0.0
            return float(min(1.0, 2 * abs(roc_auc_score(yb_b, pred) - 0.5)))

        # multiclass: accuracy lift over the majority-class baseline
        table = pd.DataFrame({"k": key_a.to_numpy(), "y": ya.astype(str).to_numpy()})
        mode = table.groupby("k")["y"].agg(lambda s: s.value_counts().index[0])
        glob = ya.astype(str).value_counts().index[0]
        pred = pd.Series(key_b.to_numpy()).map(mode).fillna(glob).to_numpy()
        yb_s = yb.astype(str).to_numpy()
        acc = float((pred == yb_s).mean())
        base = float((yb_s == glob).mean())
        return float(max(0.0, (acc - base) / (1 - base))) if base < 1 else 0.0
    except Exception:
        return 0.0


def profile_frame(df: pd.DataFrame, target: str, task: str, max_cols: int = 60) -> dict[str, Any]:
    kinds = column_kinds(df, exclude=[target])
    n = len(df)
    cols: list[dict[str, Any]] = []
    for c in list(kinds["numeric"] + kinds["categorical"] + kinds["datetime"])[:max_cols]:
        s = df[c]
        info: dict[str, Any] = {
            "name": c,
            "kind": "numeric" if c in kinds["numeric"] else ("datetime" if c in kinds["datetime"] else "categorical"),
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean()) * 100, 2),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if info["kind"] == "numeric" and s.notna().any():
            d = s.astype("float64")
            info["stats"] = {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in
                             {"min": d.min(), "p50": d.median(), "max": d.max(),
                              "mean": d.mean(), "std": d.std(), "skew": d.skew()}.items()}
        elif info["kind"] == "categorical":
            vc = s.astype(str).value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in vc.items()}
        cols.append(info)

    tgt: dict[str, Any] = {"name": target, "task": task,
                           "missing_pct": round(float(df[target].isna().mean()) * 100, 2)}
    if task in ("binary", "multiclass"):
        vc = df[target].value_counts(normalize=True)
        tgt["class_balance"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        tgt["minority_pct"] = round(float(vc.min()) * 100, 2)
    else:
        d = df[target].astype("float64")
        tgt["stats"] = {k: round(float(v), 4) for k, v in
                        {"min": d.min(), "p50": d.median(), "max": d.max(),
                         "mean": d.mean(), "std": d.std(), "skew": d.skew()}.items()}

    return {
        "n_rows": n,
        "n_columns": len(df.columns) - 1,
        "column_kinds": {k: len(v) for k, v in kinds.items()},
        "target": tgt,
        "columns": cols,
        "columns_truncated": len(kinds["numeric"] + kinds["categorical"] + kinds["datetime"]) > max_cols,
    }


def detect_issues(df: pd.DataFrame, target: str, task: str) -> list[dict[str, Any]]:
    """Data-quality findings, most severe first. Each one names the fix."""
    issues: list[dict[str, Any]] = []
    n = len(df)
    kinds = column_kinds(df, exclude=[target])
    y = df[target]

    for c in kinds["numeric"] + kinds["categorical"]:
        det = _target_determination(df[c], y, task)
        if det >= 0.95:
            issues.append({
                "severity": "critical", "kind": "target_leakage", "column": c,
                "detail": f"'{c}' alone predicts the target with {det:.1%} out-of-sample power.",
                "action": f"Almost certainly recorded after the outcome. Put '{c}' in drop_columns "
                          f"unless you can justify it being known at prediction time.",
            })
        elif det >= 0.75:
            issues.append({
                "severity": "warning", "kind": "suspicious_predictor", "column": c,
                "detail": f"'{c}' alone reaches {det:.1%} out-of-sample predictive power -- unusually high.",
                "action": "Check whether this column is available at prediction time in production.",
            })

    for c in kinds["numeric"] + kinds["categorical"] + kinds["datetime"]:
        nu = df[c].nunique(dropna=True)
        if nu <= 1:
            issues.append({"severity": "warning", "kind": "constant", "column": c,
                           "detail": f"'{c}' has {nu} distinct value(s).",
                           "action": f"Carries no information. Add '{c}' to drop_columns."})
        elif nu >= 0.98 * n and c in kinds["categorical"]:
            issues.append({"severity": "warning", "kind": "identifier", "column": c,
                           "detail": f"'{c}' is unique on {nu}/{n} rows.",
                           "action": f"Looks like an ID. Add '{c}' to drop_columns -- models memorise IDs."})
        elif nu > 50 and c in kinds["categorical"]:
            issues.append({"severity": "info", "kind": "high_cardinality", "column": c,
                           "detail": f"'{c}' has {nu} categories.",
                           "action": "One-hot would explode the matrix. Use encode='native' (lightgbm/catboost) "
                                     "or encode='target'."})

    for c in kinds["numeric"] + kinds["categorical"]:
        m = float(df[c].isna().mean())
        if m > 0.5:
            issues.append({"severity": "warning", "kind": "mostly_missing", "column": c,
                           "detail": f"'{c}' is {m:.0%} missing.",
                           "action": "Consider dropping it, or add a missing_count feature to keep the signal."})

    dup = int(df.duplicated().sum())
    if dup:
        issues.append({"severity": "warning", "kind": "duplicate_rows", "column": None,
                       "detail": f"{dup} fully duplicated rows ({dup / n:.1%}).",
                       "action": "Duplicates spanning folds inflate CV scores. Worth investigating upstream."})

    if task in ("binary", "multiclass"):
        vc = y.value_counts(normalize=True)
        if float(vc.min()) < 0.10:
            issues.append({"severity": "warning", "kind": "class_imbalance", "column": target,
                           "detail": f"Minority class is {vc.min():.1%} of rows.",
                           "action": "Accuracy is misleading here. Use roc_auc or pr_auc, and try class_weight='balanced'."})

    if n < 500:
        issues.append({"severity": "info", "kind": "small_sample", "column": None,
                       "detail": f"Only {n} rows.",
                       "action": "Use more folds (10) and treat small score differences as noise."})

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(issues, key=lambda i: order[i["severity"]])


def split_holdout(df: pd.DataFrame, target: str, task: str, seed: int = 20240617,
                  frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    strat = df[target] if task in ("binary", "multiclass") else None
    train, hold = train_test_split(df, test_size=frac, random_state=seed, shuffle=True, stratify=strat)
    return train.reset_index(drop=True), hold.reset_index(drop=True)


def frame_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df.head(1000), index=False).values.tobytes())
    return h.hexdigest()[:16]
