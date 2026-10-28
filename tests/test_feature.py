"""Tests for the custom feature and the preprocessing."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wine_quality.features import RATIO_COLUMN, SulfurRatioAdder
from wine_quality.models import Trainer


def test_ratio_adder_adds_column(toy_df):
    out = SulfurRatioAdder().fit_transform(toy_df)
    assert RATIO_COLUMN in out.columns
    assert RATIO_COLUMN not in toy_df.columns
    assert out[RATIO_COLUMN].between(0, 1).all()


def test_scaler_fitted_on_train_only(config, split):
    pipe = Trainer(config).build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(split.X_train, split.y_train)
    columns = pipe.named_steps["preprocess"].named_steps["columns"]
    scaler = columns.named_transformers_["num"].named_steps["scaler"]

    numeric = [*config["features"]["numeric"], RATIO_COLUMN]
    train = SulfurRatioAdder().transform(split.X_train)[numeric]
    np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy())

    full = pd.concat([split.X_train, split.X_test])
    everything = SulfurRatioAdder().transform(full)[numeric]
    assert not np.allclose(scaler.mean_, everything.mean().to_numpy())
