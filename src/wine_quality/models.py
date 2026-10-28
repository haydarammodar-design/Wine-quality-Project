"""Candidate models, hyperparameter search, threshold tuning and persistence."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from wine_quality.evaluate import apply_threshold, choose_threshold
from wine_quality.features import build_preprocessor


class Trainer:
    """Compare models under one cross-validation and keep the best one."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Store the configuration and create the shared CV splitter.
        Args:
            config: The project configuration (see ``config.yaml``).
        """
        self.config = config
        self.seed: int = config["seed"]
        self.cv = StratifiedKFold(
            n_splits=config["cv"]["n_splits"], shuffle=True, random_state=self.seed
        )
        self.searches_: dict[str, GridSearchCV] = {}
        self.comparison_: pd.DataFrame | None = None
        self.best_name_: str | None = None
        self.best_model_: BaseEstimator | None = None
        self.pipeline_: Pipeline | None = None
        self.threshold_: float | None = None
        self.oof_scores_: np.ndarray | None = None

    def build_pipeline(self, estimator: BaseEstimator) -> Pipeline:
        """Chain the preprocessing and a model into one pipeline.
        Args:
            estimator: The scikit-learn model to put at the end.
        Returns:
            The unfitted pipeline, with steps ``preprocess`` and ``model``.
        """
        features = self.config["features"]
        preprocess = build_preprocessor(features["numeric"], features["categorical"])
        return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])

    def candidate_models(self) -> dict[str, BaseEstimator]:
        """Return the models to compare, by name (names match ``config.yaml``)."""
        return {
            "logistic": LogisticRegression(max_iter=1000, random_state=self.seed),
            "knn": KNeighborsClassifier(),
            "linear_svm": CalibratedClassifierCV(
                LinearSVC(max_iter=5000, random_state=self.seed), cv=3
            ),
        }

    def baseline_score(self, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
        """Score a DummyClassifier under the shared cross-validation.
        Args:
            X: Training features.
            y: Training target.
        Returns:
            Mean and standard deviation of the CV score.
        """
        pipe = self.build_pipeline(DummyClassifier(strategy="prior"))
        scores = cross_val_score(pipe, X, y, cv=self.cv, scoring=self.config["scoring"])
        return float(scores.mean()), float(scores.std())

    def compare(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Tune every candidate with GridSearchCV and build the comparison table.
        Args:
            X: Training features.
            y: Training target.
        Returns:
            One row per model: best hyperparameters, CV mean and std.
        """
        mean, std = self.baseline_score(X, y)
        rows = [{"model": "dummy", "best_params": {}, "cv_mean": mean, "cv_std": std}]
        for name, estimator in self.candidate_models().items():
            search = GridSearchCV(
                self.build_pipeline(estimator),
                param_grid=self.config["models"][name],
                scoring=self.config["scoring"],
                cv=self.cv,
                n_jobs=-1,
            )
            search.fit(X, y)
            self.searches_[name] = search
            rows.append(
                {
                    "model": name,
                    "best_params": search.best_params_,
                    "cv_mean": float(search.best_score_),
                    "cv_std": float(
                        search.cv_results_["std_test_score"][search.best_index_]
                    ),
                }
            )
        table = pd.DataFrame(rows).sort_values("cv_mean", ascending=False)
        self.comparison_ = table.reset_index(drop=True)
        return self.comparison_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Trainer":
        """Compare the models, keep the best one and choose its threshold.
        The threshold is chosen on out-of-fold probabilities of the training set,
        so the test set is never used to make a choice.
        Args:
            X: Training features.
            y: Training target.
        Returns:
            The fitted trainer.
        """
        self.compare(X, y)
        self.best_name_ = max(
            self.searches_, key=lambda n: self.searches_[n].best_score_
        )
        self.pipeline_ = self.searches_[self.best_name_].best_estimator_
        self.best_model_ = self.pipeline_.named_steps["model"]
        self.oof_scores_ = cross_val_predict(
            clone(self.pipeline_), X, y, cv=self.cv, method="predict_proba"
        )[:, 1]
        self.threshold_ = choose_threshold(
            y, self.oof_scores_, self.config["threshold"]["min_recall"]
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the probability that each wine is good.
        Args:
            X: The features.
        Returns:
            The probabilities of class 1.
        Raises:
            RuntimeError: If the trainer has not been fitted or loaded.
        """
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() or Trainer.load() first.")
        return self.pipeline_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 0/1 decisions using the chosen threshold."""
        return apply_threshold(self.predict_proba(X), self.threshold_)

    def save(self, path: Path) -> None:
        """Save the fitted pipeline, its threshold and the config with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": self.pipeline_,
            "threshold": self.threshold_,
            "best_name": self.best_name_,
            "comparison": self.comparison_,
            "config": self.config,
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: Path) -> "Trainer":
        """Rebuild a fitted trainer from a file written by :meth:`save`."""
        bundle = joblib.load(path)
        trainer = cls(bundle["config"])
        trainer.pipeline_ = bundle["pipeline"]
        trainer.threshold_ = bundle["threshold"]
        trainer.best_name_ = bundle["best_name"]
        trainer.comparison_ = bundle["comparison"]
        trainer.best_model_ = trainer.pipeline_.named_steps["model"]
        return trainer
