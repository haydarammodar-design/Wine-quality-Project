"""Tests for the Trainer: predictions, threshold, save and load."""

import numpy as np

from wine_quality.models import Trainer


def test_comparison_has_baseline_and_three_models(fitted_trainer):
    models = set(fitted_trainer.comparison_["model"])
    assert models == {"dummy", "logistic", "knn", "linear_svm"}


def test_predict_proba_shape_and_range(fitted_trainer, split):
    proba = fitted_trainer.predict_proba(split.X_test)
    assert proba.shape == (len(split.X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_uses_threshold(fitted_trainer, split):
    proba = fitted_trainer.predict_proba(split.X_test)
    pred = fitted_trainer.predict(split.X_test)
    assert set(np.unique(pred)) <= {0, 1}
    np.testing.assert_array_equal(
        pred, (proba >= fitted_trainer.threshold_).astype(int)
    )


def test_save_load_roundtrip(fitted_trainer, split, tmp_path):
    path = tmp_path / "model.joblib"
    fitted_trainer.save(path)
    loaded = Trainer.load(path)
    assert loaded.threshold_ == fitted_trainer.threshold_
    np.testing.assert_allclose(
        loaded.predict_proba(split.X_test), fitted_trainer.predict_proba(split.X_test)
    )
