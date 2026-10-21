"""Turn validated data into the numeric matrix a model expects.

The central rule here is that the preprocessor is **fitted once, on training
data only**, and every later batch is only ever transformed. Fitting on a
monitoring batch would quietly change the meaning of the features - the
scaler would re-centre on new data and the encoder would learn new
categories - which both corrupts comparisons against the reference period
and leaks information that would not exist at prediction time.

To make that rule hard to break by accident, this module exposes
`fit_preprocessor` and `transform_batch` as separate functions. There is no
`fit_transform` helper for batches.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, schema

MISSING_CATEGORY = "Missing"


def build_preprocessor() -> ColumnTransformer:
    """Create the unfitted preprocessing pipeline.

    Numeric columns: median imputation, then standard scaling. The median is
    used rather than the mean because `capital_gain` and `capital_loss` are
    extremely skewed (over 90% zeros), which drags a mean around.

    Categorical columns: missing values become an explicit "Missing"
    category rather than being filled with the most common value. Phase 1
    found that missingness is concentrated in `occupation`, `workclass` and
    `native_country`, and how often a field is missing is itself a signal
    worth monitoring. Replacing it with the most frequent category would
    throw that signal away.

    `handle_unknown="ignore"` makes the encoder safe against categories it
    has never seen: instead of raising, it emits all-zeros for that column
    group. Unseen categories are reported separately by the validator, so
    they are never silently lost.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY),
            ),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, schema.NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, schema.CATEGORICAL_COLUMNS),
        ],
        remainder="drop",  # unexpected columns are ignored, never passed through
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature columns from the income label."""
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Cannot split features and target: column "
            f"'{config.TARGET_COLUMN}' is not present."
        )
    return df[schema.FEATURE_COLUMNS], df[config.TARGET_COLUMN]


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = config.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test, keeping the class balance of the original data.

    `stratify` makes both halves carry the same roughly 76/24 income split as
    the full dataset. This preserves the natural imbalance rather than
    correcting it, which is what the project requires: a monitor should see
    the distribution that real data actually has.
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[config.TARGET_COLUMN],
    )


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor. Call this on training data and nothing else."""
    missing = [c for c in schema.FEATURE_COLUMNS if c not in X_train.columns]
    if missing:
        raise ValueError(f"Cannot fit preprocessor, columns are missing: {missing}")

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train[schema.FEATURE_COLUMNS])
    return preprocessor


def transform_batch(
    preprocessor: ColumnTransformer, batch: pd.DataFrame
) -> pd.DataFrame:
    """Transform a batch with an already-fitted preprocessor.

    Returns a DataFrame with named columns rather than a bare array, so later
    monitoring phases can refer to individual encoded features by name. The
    batch's original index is kept so rows can be traced back to the source.
    """
    missing = [c for c in schema.FEATURE_COLUMNS if c not in batch.columns]
    if missing:
        raise ValueError(f"Cannot transform batch, columns are missing: {missing}")

    transformed = preprocessor.transform(batch[schema.FEATURE_COLUMNS])
    return pd.DataFrame(
        transformed,
        columns=get_feature_names(preprocessor),
        index=batch.index,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Names of the columns produced by the fitted preprocessor."""
    return list(preprocessor.get_feature_names_out())


def save_preprocessor(
    preprocessor: ColumnTransformer, path: Path = config.PREPROCESSOR_PATH
) -> None:
    """Persist the fitted preprocessor so later batches reuse the exact same one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path = config.PREPROCESSOR_PATH) -> ColumnTransformer:
    """Load the fitted preprocessor saved by `save_preprocessor`."""
    if not path.exists():
        raise FileNotFoundError(
            f"No fitted preprocessor at {path}. "
            "Run `python validate_and_preprocess.py` to create one."
        )
    return joblib.load(path)
