# Wine Quality: should this batch go to the tasting panel?
Pre-screening tool for a Vinho Verde cooperative: from 11 lab measurements and the wine type, it
predicts whether a wine is good (quality ≥ 7) so that only promising batches are sent to the
expert panel. Goal: catch at least 80 % of good wines. Final model: k-NN (k = 51) with a
threshold of 0.214; test average precision 0.60, recall 0.79, precision 0.40.
## Installation
    python -m venv .venv
    .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
    pip install -e ".[dev]"
## Usage
    pytest                                               # 19 tests
    wine-quality train                                   # compare models, save models/model.joblib
    wine-quality evaluate                                # test metrics + reports/figures/confusion_matrix.png
    wine-quality predict --input examples/new_case.json  # one new wine
## Project structure
```
wine-quality/
├── README.md                    how to install, train, evaluate, predict + data credit
├── AI_USAGE.md                  declaration of AI use
├── pyproject.toml               package metadata, pinned dependencies, the CLI command, tool settings
├── config.yaml                  every setting: seed, paths, columns, CV, metric, grids, threshold goal
├── .gitignore
├── data/raw/                    winequality-red.csv, winequality-white.csv (downloaded if missing)
├── examples/                    new_case.json, premium_case.json (inputs for predict)
├── src/wine_quality/
│   ├── __init__.py              marks the package, holds the version
│   ├── config.py                PROJECT_ROOT, load_config(), resolve()
│   ├── data.py                  DataSplit (dataclass), DatasetLoader: download, load, clean, split
│   ├── features.py              SulfurRatioAdder (custom transformer), build_preprocessor()
│   ├── evaluate.py              apply_threshold, choose_threshold, compute_metrics, plots
│   ├── models.py                Trainer: pipelines, baseline, GridSearchCV, threshold, save/load
│   └── cli.py                   wine-quality train | evaluate | predict
├── tests/                       conftest.py + 5 test files, 19 tests
├── notebooks/analysis.ipynb     the modelling story, calling the package
├── models/model.joblib          created by `wine-quality train`
└── reports/                     report.pdf, slides.pdf, figures/
```
## Data and credits
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data
mining from physicochemical properties. Decision Support Systems, 47(4):547-553, 2009.
Wine Quality dataset, UCI Machine Learning Repository (id 186), licensed CC BY 4.0.
Libraries: scikit-learn, pandas, NumPy, matplotlib, joblib, PyYAML, pytest, ruff.
## Authors
- Haidra Mohammad
- Medoune Mitchell Fall