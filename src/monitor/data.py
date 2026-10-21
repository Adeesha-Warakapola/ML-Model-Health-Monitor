"""Load the UCI Adult Census Income dataset.

The dataset is fetched from the official UCI ML Repository using the
`ucimlrepo` package, then saved to a local CSV cache. Later runs read the
cache instead of downloading again, which keeps the project reproducible
and usable without an internet connection.

Three cleaning steps are applied after loading, because the raw data needs them:

1. Column names are converted from hyphens to underscores
   ("education-num" -> "education_num") so they are valid Python identifiers.
2. The income labels appear in two spellings. The rows that came from the
   original test split keep a trailing period ("<=50K."), so the raw target
   column has four distinct values instead of two. The period is stripped.
3. Missing values are encoded inconsistently across the file: rows from the
   original training split use the literal string "?", while rows from the
   test split use a real empty value. Every "?" is converted to NaN so that
   missingness can be counted correctly with the usual pandas tools.
"""

from pathlib import Path

import pandas as pd

from . import config


def fetch_from_uci() -> pd.DataFrame:
    """Download the dataset from the UCI ML Repository.

    Returns the raw, uncleaned data with features and target joined into a
    single DataFrame. Requires an internet connection.
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
    return pd.concat([dataset.data.features, dataset.data.targets], axis=1)


def save_cache(raw_df: pd.DataFrame, path: Path = config.RAW_CACHE_PATH) -> None:
    """Save the raw (uncleaned) dataset to a local CSV so it can be reloaded offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(path, index=False)


def load_raw_data(
    path: Path = config.RAW_CACHE_PATH, refresh: bool = False
) -> pd.DataFrame:
    """Return the raw dataset, downloading and caching it on first use.

    Set `refresh=True` to force a fresh download even if the cache exists.
    """
    if refresh or not path.exists():
        raw_df = fetch_from_uci()
        save_cache(raw_df, path)
        return raw_df
    return pd.read_csv(path)


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the two cleaning steps described in the module docstring."""
    df = raw_df.copy()
    df.columns = [column.replace("-", "_") for column in df.columns]

    # Trim stray whitespace, then treat "?" as a genuine missing value.
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].str.strip()
    df = df.replace(config.MISSING_VALUE_SYMBOL, pd.NA)

    target = config.TARGET_COLUMN
    df[target] = df[target].str.rstrip(".")

    # Guard against silently loading something that is not the expected dataset.
    missing_columns = set(config.ALL_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing_columns)}")

    return df[config.ALL_COLUMNS]


def load_adult_data(refresh: bool = False) -> pd.DataFrame:
    """Load the cleaned, analysis-ready dataset. This is the function to use."""
    return clean_data(load_raw_data(refresh=refresh))
