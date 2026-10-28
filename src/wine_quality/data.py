"""Download, load, clean and split the Wine Quality data."""

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from wine_quality.config import resolve

TARGET: str = "is_good"


@dataclass
class DataSplit:
    """Features and target, split into a train part and a test part."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class DatasetLoader:
    """Load the red and white wine files, clean them and split them."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Store the configuration and compute the data folder.
        Args:
            config: The project configuration (see ``config.yaml``).
        """
        self.config = config
        self.raw_dir: Path = resolve(config["paths"]["raw_dir"])
        self.seed: int = config["seed"]

    def download(self) -> None:
        """Download the UCI zip file and extract the CSV files into ``raw_dir``."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.raw_dir / "wine_quality.zip"
        urllib.request.urlretrieve(self.config["data"]["url"], zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(self.raw_dir)
        zip_path.unlink()

    def load(self) -> pd.DataFrame:
        """Read both CSV files and stack them into one table.
        Returns:
            One row per wine, with a ``wine_type`` column ("red" or "white") and
            snake_case column names.
        """
        paths = self.config["paths"]
        red_path = self.raw_dir / paths["red_file"]
        white_path = self.raw_dir / paths["white_file"]
        if not (red_path.exists() and white_path.exists()):
            self.download()
        sep = self.config["data"]["separator"]
        red = pd.read_csv(red_path, sep=sep)
        white = pd.read_csv(white_path, sep=sep)
        red["wine_type"] = "red"
        white["wine_type"] = "white"
        df = pd.concat([red, white], ignore_index=True)
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check the columns, remove duplicates and build the binary target.
        Args:
            df: The raw table returned by :meth:`load`.
        Returns:
            A new, cleaned table with the target column ``is_good`` (0 or 1).
            ``quality`` is kept (useful for EDA) and removed in :meth:`split`.
        Raises:
            ValueError: If an expected column is missing.
        """
        data_cfg = self.config["data"]
        features = self.config["features"]
        expected = [*features["numeric"], *features["categorical"]]
        expected.append(data_cfg["quality_column"])
        missing = [col for col in expected if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.copy()
        if data_cfg["drop_duplicates"]:
            df = df.drop_duplicates().reset_index(drop=True)
        quality = df[data_cfg["quality_column"]]
        df[TARGET] = (quality >= data_cfg["good_threshold"]).astype(int)
        return df

    def split(self, df: pd.DataFrame) -> DataSplit:
        """Split the cleaned table into stratified train and test parts.
        Args:
            df: The table returned by :meth:`clean`.
        Returns:
            The train/test split. ``quality`` and the target are not in X.
        """
        X = df.drop(columns=[self.config["data"]["quality_column"], TARGET])
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["data"]["test_size"],
            stratify=y,
            random_state=self.seed,
        )
        return DataSplit(X_train, X_test, y_train, y_test)

    def load_and_split(self) -> DataSplit:
        """Run :meth:`load`, :meth:`clean` and :meth:`split` in order."""
        return self.split(self.clean(self.load()))
