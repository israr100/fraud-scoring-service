"""Training-pipeline tests: the model must actually learn signal."""
import numpy as np

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, make_synthetic
from src.train import build_pipeline, tune_threshold


def test_synthetic_schema_and_imbalance():
    df = make_synthetic(n_rows=5_000, fraud_rate=0.01)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns
    rate = df[TARGET_COLUMN].mean()
    assert 0.0 < rate < 0.05  # genuinely imbalanced


def test_threshold_tuner_prefers_recall():
    # Perfect scores -> tuner should pick a cutoff that catches every positive.
    y = np.array([0, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8])
    t = tune_threshold(y, scores)
    pred = (scores >= t).astype(int)
    assert pred.tolist() == y.tolist()


def test_pipeline_learns_signal():
    df = make_synthetic(n_rows=8_000, fraud_rate=0.02, seed=1)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    pos = int(y.sum())
    pipe = build_pipeline(scale_pos_weight=(len(y) - pos) / pos)
    pipe.fit(X, y)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    assert auc > 0.85  # clearly better than chance
