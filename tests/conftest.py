"""Shared pytest fixtures."""

import copy
from typing import Any

import pandas as pd
import pytest

from wine_quality.config import load_config
from wine_quality.data import DatasetLoader, DataSplit
from wine_quality.models import Trainer


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """The real project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def small_config(config: dict[str, Any]) -> dict[str, Any]:
    """The configuration with one value per grid, so training is fast in tests."""
    small = copy.deepcopy(config)
    small["models"] = {
        "logistic": {"model__C": [1.0]},
        "knn": {"model__n_neighbors": [21]},
        "linear_svm": {"model__estimator__C": [1.0]},
    }
    return small


@pytest.fixture
def toy_df(config: dict[str, Any]) -> pd.DataFrame:
    """Four raw wines: the last row is an exact copy of the first one."""
    values = [
        [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, "red", 5],
        [6.3, 0.30, 0.34, 1.6, 0.049, 14.0, 132.0, 0.994, 3.3, 0.49, 12.5, "white", 7],
        [8.1, 0.28, 0.40, 6.9, 0.050, 30.0, 97.0, 0.9951, 3.26, 0.44, 10.1, "white", 6],
        [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, "red", 5],
    ]
    features = config["features"]
    columns = [*features["numeric"], *features["categorical"], "quality"]
    return pd.DataFrame(values, columns=columns)


@pytest.fixture(scope="session")
def split(config: dict[str, Any]) -> DataSplit:
    """The real train/test split."""
    return DatasetLoader(config).load_and_split()


@pytest.fixture(scope="session")
def fitted_trainer(small_config: dict[str, Any], split: DataSplit) -> Trainer:
    """A trainer fitted on the real training set with the small grids."""
    return Trainer(small_config).fit(split.X_train, split.y_train)
