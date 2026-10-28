"""Command-line tool: ``wine-quality train | evaluate | predict``."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from wine_quality.config import CONFIG_PATH, load_config, resolve
from wine_quality.data import DatasetLoader
from wine_quality.evaluate import (
    apply_threshold,
    compute_metrics,
    plot_confusion_matrix,
)
from wine_quality.models import Trainer


def cmd_train(config: dict[str, Any]) -> None:
    """Train and compare the models, then save the best one."""
    split = DatasetLoader(config).load_and_split()
    trainer = Trainer(config).fit(split.X_train, split.y_train)
    print(trainer.comparison_.to_string(index=False))
    print(f"Best model: {trainer.best_name_} | threshold: {trainer.threshold_:.3f}")
    model_path = resolve(config["paths"]["model_file"])
    trainer.save(model_path)
    print(f"Model saved to {model_path}")


def cmd_evaluate(config: dict[str, Any]) -> None:
    """Print the test metrics of the saved model and save its confusion matrix."""
    split = DatasetLoader(config).load_and_split()
    trainer = Trainer.load(resolve(config["paths"]["model_file"]))
    scores = trainer.predict_proba(split.X_test)
    metrics = compute_metrics(split.y_test, scores, trainer.threshold_)
    print(f"Test metrics of '{trainer.best_name_}':")
    for name, value in metrics.items():
        print(f"  {name:>17}: {value:.3f}")
    figure_path = resolve(config["paths"]["figures_dir"]) / "confusion_matrix.png"
    y_pred = apply_threshold(scores, trainer.threshold_)
    plot_confusion_matrix(split.y_test, y_pred, figure_path)
    print(f"Confusion matrix saved to {figure_path}")


def cmd_predict(config: dict[str, Any], input_path: Path) -> None:
    """Predict one new wine described in a JSON file."""
    case = json.loads(input_path.read_text(encoding="utf-8"))
    features = config["features"]
    required = [*features["numeric"], *features["categorical"]]
    missing = [name for name in required if name not in case]
    if missing:
        raise SystemExit(f"Missing fields in {input_path}: {', '.join(missing)}")

    X_new = pd.DataFrame([case])[required]
    trainer = Trainer.load(resolve(config["paths"]["model_file"]))
    proba = float(trainer.predict_proba(X_new)[0])
    if proba >= trainer.threshold_:
        decision = "GOOD -> send to the tasting panel"
    else:
        decision = "NOT GOOD -> standard line"
    print(f"P(good) = {proba:.3f} | threshold = {trainer.threshold_:.3f} | {decision}")


def build_parser() -> argparse.ArgumentParser:
    """Create the parser with the train, evaluate and predict sub-commands."""
    parser = argparse.ArgumentParser(
        prog="wine-quality", description="Predict whether a wine is good."
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH, help="Path to config.yaml."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train, compare and save the model.")
    sub.add_parser("evaluate", help="Print the test metrics of the saved model.")
    predict = sub.add_parser("predict", help="Predict one new wine.")
    predict.add_argument(
        "--input", type=Path, required=True, help="JSON file describing the wine."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of the ``wine-quality`` command.

    Args:
        argv: Arguments to parse; None means the real command line.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "train":
        cmd_train(config)
    elif args.command == "evaluate":
        cmd_evaluate(config)
    elif args.command == "predict":
        cmd_predict(config, args.input)


if __name__ == "__main__":
    main()
