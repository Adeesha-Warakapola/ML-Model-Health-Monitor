"""Central place for file paths, column names, and settings shared across the project.

Keeping these in one module means every other file imports the same constants
instead of re-typing paths or column names, so a change only needs to happen once.
"""

from pathlib import Path

# Project root is two levels up from this file: src/monitor/config.py -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_CACHE_PATH = DATA_DIR / "adult_raw.csv"

DOCS_DIR = PROJECT_ROOT / "docs"
EXPLORATION_REPORT_PATH = DOCS_DIR / "dataset_exploration.md"

# The Adult / Census Income dataset is id=2 in the UCI ML Repository.
UCI_DATASET_ID = 2
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/2/adult"

# Fixed seed so any sampling behaves the same way on every run (reproducibility).
RANDOM_SEED = 42

# The raw files mark missing values with "?" rather than an empty cell.
MISSING_VALUE_SYMBOL = "?"

TARGET_COLUMN = "income"
POSITIVE_LABEL = ">50K"
TARGET_LABELS = ["<=50K", ">50K"]

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
