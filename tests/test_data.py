"""Tests for loading, cleaning and splitting."""

import pandas as pd
import pytest

from wine_quality.data import TARGET, DatasetLoader


def test_load_shape_and_columns(config):
    df = DatasetLoader(config).load()
    assert df.shape == (6497, 13)
    assert set(df["wine_type"]) == {"red", "white"}
    assert "fixed_acidity" in df.columns


def test_clean_removes_duplicates(config, toy_df):
    cleaned = DatasetLoader(config).clean(toy_df)
    assert len(cleaned) == len(toy_df) - 1
    assert cleaned.duplicated().sum() == 0


def test_target_matches_quality_threshold(config, toy_df):
    cleaned = DatasetLoader(config).clean(toy_df)
    assert set(cleaned[TARGET]) <= {0, 1}
    expected = (cleaned["quality"] >= 7).astype(int)
    pd.testing.assert_series_equal(cleaned[TARGET], expected, check_names=False)


def test_clean_rejects_missing_column(config, toy_df):
    with pytest.raises(ValueError):
        DatasetLoader(config).clean(toy_df.drop(columns=["alcohol"]))


def test_quality_and_target_not_in_features(split):
    for X in (split.X_train, split.X_test):
        assert "quality" not in X.columns
        assert TARGET not in X.columns


def test_split_no_test_row_in_train(split):
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert len(split.X_train) == len(split.y_train)


def test_split_is_stratified(split):
    assert abs(split.y_train.mean() - split.y_test.mean()) < 0.01
