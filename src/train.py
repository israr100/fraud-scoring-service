"""Training pipeline: data -> XGBoost -> evaluated, threshold-tuned artifact.

Run with:
    python -m src.train                     # fast default (single fit, random split)
    python -m src.train --split temporal    # time-based split (more honest here)
    python -m src.train --search            # cross-validated hyperparameter search
    python -m src.train --calibrate         # isotonic probability calibration
    python -m src.train --search --calibrate --split temporal   # the works

Design notes (the things an ML-engineering reviewer looks for):
  * Stratified (or temporal) train/val/test split so the tiny positive class is
    represented and, optionally, no "future" leaks into training.
  * `scale_pos_weight` to handle extreme imbalance instead of naive resampling.
  * Model + preprocessing bundled in one sklearn Pipeline -> one artifact, no
    train/serve skew.
  * Optional RandomizedSearchCV over hyperparameters, scored by average precision.
  * Optional isotonic calibration so predicted scores are true probabilities.
  * Decision threshold tuned on validation via cost-weighted F-beta, then frozen
    into the artifact (serving must not re-tune).
  * Metrics reported on a held-out test set the tuner never saw.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import MODELS_DIR, settings
from .data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset

# Recall matters more than precision in fraud (a missed fraud costs more than a
# false alarm), so we optimize F-beta with beta > 1.
FBETA = 2.0

# The search space for --search. Names are "<step>__<param>" for the Pipeline.
SEARCH_SPACE = {
    "clf__n_estimators": [200, 300, 400, 600],
    "clf__max_depth": [3, 4, 5, 6, 8],
    "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "clf__subsample": [0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "clf__min_child_weight": [1, 3, 5],
}


@dataclass
class Metrics:
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    threshold: float
    n_train: int
    n_test: int
    positive_rate: float


def _fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def tune_threshold(y_true: np.ndarray, scores: np.ndarray, beta: float = FBETA) -> float:
    """Pick the probability cutoff maximizing F-beta on validation data."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (scores >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        s = _fbeta(p, r, beta)
        if s > best_score:
            best_score, best_t = s, float(t)
    return best_t


def build_pipeline(scale_pos_weight: float, n_jobs: int = -1) -> Pipeline:
    """StandardScaler + XGBoost as one fit/predict object.

    n_jobs is exposed so hyperparameter search can set it to 1 and parallelize
    across CV folds instead (avoids CPU oversubscription).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=n_jobs,
            random_state=42,
        )),
    ])


def search_hyperparams(X_train, y_train, spw: float, n_iter: int, seed: int):
    """Cross-validated random search; returns (best_pipeline, best_params).

    Scored by average precision (PR-AUC), the right objective under imbalance.
    Refits the best configuration on the full training set.
    """
    base = build_pipeline(spw, n_jobs=1)  # let the search parallelize the folds
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base, SEARCH_SPACE, n_iter=n_iter, scoring="average_precision",
        cv=cv, n_jobs=-1, random_state=seed, refit=True, verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def _split(df, split: str, seed: int):
    """Return X_train, X_val, X_test, y_train, y_val, y_test as a 60/20/20 split.

    split="random":   stratified random split (default).
    split="temporal": order by the Time column and cut chronologically, so the
                      test set is strictly "later" than training — no leakage.
    """
    if split == "temporal" and "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)
        X = df[FEATURE_COLUMNS].values
        y = df[TARGET_COLUMN].values
        n = len(df)
        i1, i2 = int(0.6 * n), int(0.8 * n)
        return (X[:i1], X[i1:i2], X[i2:], y[:i1], y[i1:i2], y[i2:])

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train(search: bool = False, calibrate: bool = False, split: str = "random",
          n_iter: int = 15, seed: int = 42) -> Metrics:
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = _split(df, split, seed)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(pos, 1)

    t0 = time.time()
    best_params = None
    if search:
        pipe, best_params = search_hyperparams(X_train, y_train, spw, n_iter, seed)
    else:
        pipe = build_pipeline(spw)
        pipe.fit(X_train, y_train)

    # Optional calibration: turn raw scores into well-behaved probabilities,
    # fitted on the validation set. sklearn >=1.6 wraps the already-fitted model
    # in FrozenEstimator; older versions use cv="prefit". Support both.
    model = pipe
    if calibrate:
        try:
            from sklearn.frozen import FrozenEstimator
            model = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
        except ImportError:  # sklearn < 1.6
            model = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
        model.fit(X_val, y_val)
    fit_secs = time.time() - t0

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_scores)

    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)

    metrics = Metrics(
        roc_auc=round(float(roc_auc_score(y_test, test_scores)), 4),
        average_precision=round(float(average_precision_score(y_test, test_scores)), 4),
        precision=round(float(precision_score(y_test, test_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_test, test_pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
        threshold=round(threshold, 4),
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        positive_rate=round(float((y_train.sum() + y_val.sum() + y_test.sum())
                                  / (len(y_train) + len(y_val) + len(y_test))), 5),
    )

    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "pipeline": model,                      # calibrated model if --calibrate
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "metrics": asdict(metrics),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fit_seconds": round(fit_secs, 2),
        "config": {"search": search, "calibrated": calibrate, "split": split,
                   "best_params": best_params},
    }
    joblib.dump(artifact, settings.model_path)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2))
    return metrics


def _parse_args():
    p = argparse.ArgumentParser(description="Train the fraud-scoring model.")
    p.add_argument("--search", action="store_true",
                   help="cross-validated hyperparameter search (slower)")
    p.add_argument("--calibrate", action="store_true",
                   help="isotonic probability calibration")
    p.add_argument("--split", choices=["random", "temporal"], default="random",
                   help="train/test split strategy")
    p.add_argument("--n-iter", type=int, default=15, dest="n_iter",
                   help="search iterations when --search is set")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    m = train(search=args.search, calibrate=args.calibrate,
              split=args.split, n_iter=args.n_iter)
    print(json.dumps(asdict(m), indent=2))