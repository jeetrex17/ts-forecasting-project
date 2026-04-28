"""Weighted skill score and helpers.

The weighted skill score is the primary evaluation metric:

    score = sqrt( 1 - clip_{0,1}( sum_i w_i (y_i - yhat_i)^2 / sum_i w_i y_i^2 ) )

A score of 1 is a perfect forecast and 0 means no improvement over the
clipped reference. The metric is dominated by high-weight rows because
the panel weight distribution is extremely concentrated.
"""
from __future__ import annotations
import numpy as np


def weighted_skill_score(y_true, y_pred, weights):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    num = np.sum(weights * (y_true - y_pred) ** 2)
    den = np.sum(weights * y_true ** 2)
    if den <= 0:
        return 0.0
    ratio = np.clip(num / den, 0.0, 1.0)
    return float(np.sqrt(1.0 - ratio))


def weighted_rmse(y_true, y_pred, weights):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights)))


def bootstrap_skill(y_true, y_pred, weights, n_boot=500, seed=0):
    """Return (mean, lo, hi) 95% CI on the weighted skill score by resampling rows."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = weighted_skill_score(y_true[idx], y_pred[idx], weights[idx])
    return float(scores.mean()), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
