"""Feature engineering and the preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RATIO_COLUMN: str = "free_so2_ratio"


class SulfurRatioAdder(BaseEstimator, TransformerMixin):
    """Add the share of free sulfur dioxide in the total sulfur dioxide."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SulfurRatioAdder":
        """Learn nothing: the ratio needs no statistics from the data.
        Args:
            X: The features.
            y: Ignored, present for scikit-learn compatibility.
        Returns:
            The transformer itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with the new ratio column.
        Args:
            X: Features containing the two sulfur dioxide columns.
        Returns:
            The features plus ``free_so2_ratio`` (NaN where the total is 0, so
            the imputer can fill it).
        """
        X = X.copy()
        total = X["total_sulfur_dioxide"].replace(0, np.nan)
        X[RATIO_COLUMN] = X["free_sulfur_dioxide"] / total
        return X


def build_preprocessor(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Build the (unfitted) preprocessing pipeline.
    Args:
        numeric: Names of the numeric columns (before the ratio is added).
        categorical: Names of the categorical columns.
    Returns:
        A pipeline: add the ratio, then impute + scale numeric columns and
        impute + one-hot encode categorical columns.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, [*numeric, RATIO_COLUMN]),
            ("cat", categorical_pipe, categorical),
        ]
    )
    return Pipeline(steps=[("add_features", SulfurRatioAdder()), ("columns", columns)])
