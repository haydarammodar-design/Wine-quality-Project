"""Tests for the command-line tool."""

import json

import pytest
import yaml

from wine_quality.cli import main

NEW_CASE = {
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.27,
    "citric_acid": 0.36,
    "residual_sugar": 20.7,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 45.0,
    "total_sulfur_dioxide": 170.0,
    "density": 1.001,
    "pH": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8,
    "wine_type": "white",
}


@pytest.fixture
def cli_files(tmp_path, small_config, fitted_trainer):
    """A config pointing to a model saved in a temporary folder, and a JSON case."""
    model_path = tmp_path / "model.joblib"
    fitted_trainer.save(model_path)
    config = {**small_config, "paths": {**small_config["paths"]}}
    config["paths"]["model_file"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(NEW_CASE), encoding="utf-8")
    return config_path, case_path


def test_predict_prints_probability(cli_files, capsys):
    config_path, case_path = cli_files
    main(["--config", str(config_path), "predict", "--input", str(case_path)])
    out = capsys.readouterr().out
    assert "P(good) =" in out
    proba = float(out.split("P(good) = ")[1].split()[0])
    assert 0 <= proba <= 1


def test_predict_missing_field_exits(cli_files, tmp_path):
    config_path, _ = cli_files
    bad_case = tmp_path / "bad.json"
    bad_case.write_text(json.dumps({"alcohol": 12.0}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--config", str(config_path), "predict", "--input", str(bad_case)])
