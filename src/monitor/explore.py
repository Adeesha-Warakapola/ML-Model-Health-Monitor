"""Inspect the Adult dataset and summarise what is actually in it.

Each function answers one question about the data and returns a plain pandas
object, so the results can be printed, tested, or written into a report
without any of them depending on each other.
"""

import pandas as pd

from . import config


def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: data type, missing values, and how many distinct values it holds."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": df.notna().sum().values,
            "missing": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
            "n_unique": df.nunique().values,
        }
    ).reset_index(drop=True)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Only the columns that actually have missing values, worst first."""
    summary = schema_summary(df)
    has_missing = summary[summary["missing"] > 0]
    return has_missing.sort_values("missing", ascending=False).reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Count fully identical rows.

    Note these are not necessarily data errors: the dataset holds anonymous
    census records, so two different people can legitimately share every
    recorded value.
    """
    duplicate_mask = df.duplicated(keep="first")
    return {
        "n_duplicate_rows": int(duplicate_mask.sum()),
        "duplicate_pct": round(float(duplicate_mask.mean() * 100), 4),
        "n_unique_rows": int(len(df) - duplicate_mask.sum()),
    }


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Class counts and percentages for the income target."""
    counts = df[config.TARGET_COLUMN].value_counts(dropna=False)
    return pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "percent": (counts.values / len(df) * 100).round(2),
        }
    ).reset_index(drop=True)


def class_imbalance_ratio(df: pd.DataFrame) -> float:
    """How many majority-class rows exist per minority-class row."""
    counts = df[config.TARGET_COLUMN].value_counts()
    return round(float(counts.max() / counts.min()), 2)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Standard descriptive statistics for the numeric feature columns."""
    return df[config.NUMERIC_FEATURES].describe().T.round(2)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """For each categorical feature: how many categories, and the most common one."""
    rows = []
    for column in config.CATEGORICAL_FEATURES:
        counts = df[column].value_counts(dropna=True)
        rows.append(
            {
                "column": column,
                "n_categories": int(df[column].nunique(dropna=True)),
                "most_common": counts.index[0],
                "most_common_pct": round(float(counts.iloc[0] / df[column].notna().sum() * 100), 2),
            }
        )
    return pd.DataFrame(rows)
