"""Tests for the threshold functions and the metrics."""

import numpy as np
import pytest

from wine_quality.evaluate import apply_threshold, choose_threshold, compute_metrics


def test_apply_threshold():
    out = apply_threshold(np.array([0.2, 0.5, 0.9]), 0.5)
    np.testing.assert_array_equal(out, [0, 1, 1])


def test_choose_threshold_reaches_min_recall():
    y_true = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    threshold = choose_threshold(y_true, scores, min_recall=0.75)
    y_pred = apply_threshold(scores, threshold)
    recall = (y_pred[y_true == 1] == 1).mean()
    assert recall >= 0.75
    assert threshold == pytest.approx(0.6)


def test_choose_threshold_rejects_invalid_recall():
    with pytest.raises(ValueError):
        choose_threshold(np.array([0, 1]), np.array([0.2, 0.8]), min_recall=1.5)


def test_compute_metrics_keys_and_range():
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.4, 0.6])
    metrics = compute_metrics(y_true, scores, threshold=0.5)
    assert {"average_precision", "roc_auc", "precision", "recall", "f1"} <= set(metrics)
    assert all(0 <= value <= 1 for value in metrics.values())
