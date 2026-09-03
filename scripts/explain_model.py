"""Explain a trained fraud-scoring model with five diagnostic views.

Run (from the repo root):
    python -m scripts.explain_model                 # uses models/model.joblib
    python -m scripts.explain_model --split temporal # match a temporal-split model

Outputs PNGs into model_report/ and prints a short text summary. Only needs
matplotlib on top of the training dependencies (SHAP contributions come from
XGBoost itself, no `shap` package required).

The five views:
  1. feature_importance.png  — which features the trees rely on (gain)
  2. score_distributions.png — P(fraud) for real fraud vs legit (how separable)
  3. pr_roc_curves.png       — precision-recall + ROC, operating point marked
  4. calibration_curve.png   — are the predicted probabilities honest?
  5. single_explanation.png  — per-feature SHAP contributions for one alert
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless: write files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from src.data import FEATURE_COLUMNS, load_dataset
from src.train import _split

LEGIT, FRAUD = "#3b6ea5", "#d1495b"   # colour-blind-safe blue / red
OUT = Path("model_report")


def _unwrap(model):
    """Return (pipeline, xgb_classifier) from a plain or calibrated model.

    Best-effort: handles Pipeline directly, and CalibratedClassifierCV wrapping
    a (possibly FrozenEstimator-wrapped) Pipeline. Returns (None, None) parts it
    cannot find, and the caller degrades gracefully.
    """
    m = model
    if hasattr(m, "calibrated_classifiers_"):          # CalibratedClassifierCV
        cc = m.calibrated_classifiers_[0]
        m = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", m)
    if hasattr(m, "estimator") and not hasattr(m, "named_steps"):  # FrozenEstimator
        m = m.estimator
    pipe = m if hasattr(m, "named_steps") else None
    xgb = pipe.named_steps.get("clf") if pipe is not None else None
    return pipe, xgb


def _prob(model, X):
    return model.predict_proba(X)[:, 1]


def plot_importance(xgb, path):
    if xgb is None:
        print("  (skipped feature importance — could not reach the XGBoost model)")
        return
    imp = np.asarray(xgb.feature_importances_)
    order = np.argsort(imp)[::-1][:15][::-1]
    names = [FEATURE_COLUMNS[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, imp[order], color=LEGIT)
    ax.set_title("Feature importance (gain) — what the trees split on")
    ax.set_xlabel("relative importance")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_scores(p, y, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(p[y == 0], bins=50, color=LEGIT, alpha=0.7, label="legit", log=True)
    ax.hist(p[y == 1], bins=50, color=FRAUD, alpha=0.8, label="fraud", log=True)
    ax.set_title("Predicted P(fraud) by true class")
    ax.set_xlabel("model probability"); ax.set_ylabel("count (log scale)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pr_roc(p, y, threshold, path):
    prec, rec, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    yhat = (p >= threshold).astype(int)
    # operating point
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    op_prec = tp / (tp + fp) if tp + fp else 0
    op_rec = tp / (tp + fn) if tp + fn else 0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot(rec, prec, color=LEGIT)
    a1.scatter([op_rec], [op_prec], color=FRAUD, zorder=5,
               label=f"threshold {threshold:.2f}\nP={op_prec:.2f} R={op_rec:.2f}")
    a1.set_title(f"Precision-Recall (AP={average_precision_score(y, p):.3f})")
    a1.set_xlabel("recall"); a1.set_ylabel("precision"); a1.legend(loc="lower left")
    a2.plot(fpr, tpr, color=LEGIT); a2.plot([0, 1], [0, 1], "--", color="#999")
    a2.set_title(f"ROC (AUC={roc_auc_score(y, p):.3f})")
    a2.set_xlabel("false positive rate"); a2.set_ylabel("true positive rate")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_calibration(p, y, path):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=FRAUD, label="this model")
    ax.set_title("Calibration (reliability) curve")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraud rate")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_single_explanation(pipe, xgb, X_row, p_row, path):
    """Per-feature SHAP contributions for one transaction, via XGBoost's
    native pred_contribs (no `shap` package needed)."""
    if pipe is None or xgb is None:
        print("  (skipped single-transaction explanation — no tree model reachable)")
        return
    import xgboost as xgblib
    scaler = pipe.named_steps.get("scaler")
    Xs = scaler.transform(X_row.reshape(1, -1)) if scaler is not None else X_row.reshape(1, -1)
    booster = xgb.get_booster()
    contribs = booster.predict(xgblib.DMatrix(Xs, feature_names=FEATURE_COLUMNS),
                               pred_contribs=True)[0]
    feat_contribs = contribs[:-1]  # last entry is the bias/base value
    order = np.argsort(np.abs(feat_contribs))[::-1][:12][::-1]
    names = [FEATURE_COLUMNS[i] for i in order]
    vals = feat_contribs[order]
    colors = [FRAUD if v > 0 else LEGIT for v in vals]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_title(f"Why this transaction scored {p_row:.2f}\n(red pushes toward fraud, blue toward legit; log-odds)")
    ax.set_xlabel("contribution to the raw score (log-odds)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Explain a trained fraud model.")
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    args = ap.parse_args()

    art = joblib.load(args.model)
    model = art["pipeline"]
    threshold = art["threshold"]
    pipe, xgb = _unwrap(model)

    df = load_dataset()
    _, _, X_test, _, _, y_test = _split(df, args.split, seed=42)
    p = _prob(model, X_test)

    OUT.mkdir(exist_ok=True)
    print(f"Model: {args.model}  |  threshold {threshold}  |  test rows {len(y_test)}")
    print(f"Test AUC {roc_auc_score(y_test, p):.3f}  AP {average_precision_score(y_test, p):.3f}")

    plot_importance(xgb, OUT / "feature_importance.png")
    plot_scores(p, y_test, OUT / "score_distributions.png")
    plot_pr_roc(p, y_test, threshold, OUT / "pr_roc_curves.png")
    plot_calibration(p, y_test, OUT / "calibration_curve.png")

    # pick the highest-scoring true fraud as the example to explain
    fraud_idx = np.where(y_test == 1)[0]
    if len(fraud_idx):
        j = fraud_idx[np.argmax(p[fraud_idx])]
        plot_single_explanation(pipe, xgb, X_test[j], p[j], OUT / "single_explanation.png")

    print(f"Wrote 5 charts to {OUT}/")


if __name__ == "__main__":
    main()
