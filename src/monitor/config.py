"""Central place for file paths, column names, and settings shared across the project.

Keeping these in one module means every other file imports the same constants
instead of re-typing paths or column names, so a change only needs to happen once.
"""

from pathlib import Path

from . import schema

# Project root is two levels up from this file: src/monitor/config.py -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_CACHE_PATH = DATA_DIR / "adult_raw.csv"

DOCS_DIR = PROJECT_ROOT / "docs"
EXPLORATION_REPORT_PATH = DOCS_DIR / "dataset_exploration.md"

# Fitted objects are saved here so that every monitoring batch is processed
# with the exact same preprocessor that was fitted on the training data.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

# The baseline model artifact bundles preprocessing and the classifier, so
# loading this one file is enough to make a prediction.
BASELINE_MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"
BASELINE_METADATA_PATH = ARTIFACTS_DIR / "baseline_model_metadata.json"

# Evaluation results are committed, unlike the model binary, because they are
# small, readable, and document what the baseline actually achieved.
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"

# Fraction of the data held back for evaluation.
TEST_SIZE = 0.2

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

# Column names are defined once, in schema.py, which also carries the allowed
# categories and value bounds. These are plain aliases so that older callers
# keep working while there is only ever one definition to keep correct.
NUMERIC_FEATURES = schema.NUMERIC_COLUMNS
CATEGORICAL_FEATURES = schema.CATEGORICAL_COLUMNS
FEATURE_COLUMNS = schema.FEATURE_COLUMNS
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
