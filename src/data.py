"""Dataset loading and a self-contained synthetic generator.

The service is modeled on the classic Kaggle "Credit Card Fraud Detection"
dataset (columns: Time, V1..V28 anonymized PCA components, Amount, Class),
which is highly imbalanced (~0.17% positives).

To keep the repo runnable with zero external downloads, `load_dataset` falls
back to a synthetic generator that reproduces the same schema and a similar
class imbalance. To train on the real data instead, drop `creditcard.csv`
into data/ (see README) and it will be picked up automatically.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
REAL_CSV = DATA_DIR / "creditcard.csv"

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def make_synthetic(n_rows: int = 60_000, fraud_rate: float = 0.004, seed: int = 42) -> pd.DataFrame:
    """Generate an imbalanced, fraud-like tabular dataset.

    Fraud rows are drawn from a shifted/wider distribution on a subset of the
    PCA-like features so a model can learn a real (but non-trivial) boundary.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    # Legitimate transactions: standard normal PCA components.
    legit = rng.standard_normal((n_legit, 28))
    # Fraud: a modest shift on a few components with heavy overlap -> a genuine,
    # non-separable boundary (realistic AUC in the ~0.95 range, not a giveaway 1.0).
    fraud = rng.standard_normal((n_fraud, 28))
    signal_cols = [0, 2, 3, 9, 13]
    fraud[:, signal_cols] += rng.uniform(1.0, 1.5, size=len(signal_cols))
    fraud[:, signal_cols] *= 1.15
    # Label noise: a small fraction of fraud looks perfectly ordinary.
    n_noise = int(0.10 * n_fraud)
    fraud[:n_noise, signal_cols] = rng.standard_normal((n_noise, len(signal_cols)))

    X = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Amount: fraud skews toward larger, rounder values.
    amount = np.concatenate([
        rng.gamma(2.0, 30.0, n_legit),
        rng.gamma(3.5, 60.0, n_fraud),
    ])
    time = rng.uniform(0, 172_800, n_rows)  # two days in seconds, like the original

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df[TARGET_COLUMN] = y

    # Shuffle so rows aren't ordered by class.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    """Return the real Kaggle CSV if present, else a synthetic stand-in."""
    if REAL_CSV.exists():
        return pd.read_csv(REAL_CSV)
    DATA_DIR.mkdir(exist_ok=True)
    return make_synthetic()
