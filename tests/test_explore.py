"""Tests for the dataset inspection helpers.

Most tests use a small handcrafted DataFrame so the expected answers can be
checked by hand, rather than depending on the real dataset's numbers.
"""

import numpy as np
import pandas as pd
import pytest

from monitor import config, explore
from monitor.data import load_adult_data
from monitor.report import build_markdown_report


def make_sample() -> pd.DataFrame:
    """Four rows with one known missing value and one known duplicate pair."""
    base = {
        "age": 30,
        "fnlwgt": 100000,
        "education_num": 10,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "workclass": "Private",
        "education": "HS-grad",
        "marital_status": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
        "income": "<=50K",
    }
    rows = [
        dict(base),
        dict(base),  # exact duplicate of row 0
        {**base, "age": 45, "workclass": None, "income": ">50K"},
        {**base, "age": 50, "income": "<=50K"},
    ]
    return pd.DataFrame(rows)[config.ALL_COLUMNS]


def test_schema_summary_counts_missing_values():
    summary = explore.schema_summary(make_sample())
    workclass = summary[summary["column"] == "workclass"].iloc[0]

    assert workclass["missing"] == 1
    assert workclass["non_null"] == 3
    assert workclass["missing_pct"] == 25.0


def test_missing_value_summary_lists_only_affected_columns():
    summary = explore.missing_value_summary(make_sample())
    assert list(summary["column"]) == ["workclass"]


def test_duplicate_summary_finds_the_identical_pair():
    result = explore.duplicate_summary(make_sample())
    assert result["n_duplicate_rows"] == 1
    assert result["n_unique_rows"] == 3


def test_target_distribution_counts_and_percentages():
    distribution = explore.target_distribution(make_sample())
    counts = dict(zip(distribution["label"], distribution["count"]))

    assert counts == {"<=50K": 3, ">50K": 1}
    assert distribution[distribution["label"] == ">50K"]["percent"].iloc[0] == 25.0


def test_class_imbalance_ratio():
    assert explore.class_imbalance_ratio(make_sample()) == 3.0


def test_numeric_summary_covers_every_numeric_feature():
    summary = explore.numeric_summary(make_sample())
    assert list(summary.index) == config.NUMERIC_FEATURES
    assert summary.loc["age", "max"] == 50


def test_categorical_summary_reports_dominant_category():
    summary = explore.categorical_summary(make_sample())
    workclass = summary[summary["column"] == "workclass"].iloc[0]

    # 3 non-missing values, all "Private"
    assert workclass["n_categories"] == 1
    assert workclass["most_common"] == "Private"
    assert workclass["most_common_pct"] == 100.0


def test_report_contains_the_real_headline_numbers():
    # Generated against the real dataset: the report must state the actual
    # row count and class split rather than placeholder text.
    df = load_adult_data()
    markdown = build_markdown_report(df)

    assert "48,842" in markdown
    assert "76.07" in markdown
    assert "23.93" in markdown
    assert "# Dataset Exploration Report" in markdown
