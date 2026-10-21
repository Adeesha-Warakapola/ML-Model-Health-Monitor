"""Tests for loading and cleaning the Adult dataset.

These run against the cached copy in data/, so they do not need the network.
"""

import pandas as pd
import pytest

from monitor import config
from monitor.data import clean_data, load_adult_data


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_adult_data()


def test_dataset_has_documented_dimensions(df):
    # 48,842 rows is the combined train + test size documented by UCI.
    assert df.shape == (48842, 15)


def test_columns_match_config_and_use_underscores(df):
    assert list(df.columns) == config.ALL_COLUMNS
    assert not any("-" in column for column in df.columns)


def test_target_has_exactly_two_labels(df):
    # The raw data contains four spellings because the test-split rows keep a
    # trailing period ("<=50K."). Cleaning must collapse these to two.
    assert sorted(df[config.TARGET_COLUMN].unique()) == sorted(config.TARGET_LABELS)


def test_question_marks_are_converted_to_missing_values(df):
    # The training split encodes missing values as "?" while the test split
    # uses real empty values. Every "?" must become NaN.
    assert (df == config.MISSING_VALUE_SYMBOL).sum().sum() == 0
    assert df["workclass"].isna().sum() > 0


def test_category_counts_match_uci_documentation(df):
    # If "?" were still present each of these would be one higher, so this
    # doubles as a regression test for the missing-value cleaning.
    assert df["workclass"].nunique() == 8
    assert df["occupation"].nunique() == 14
    assert df["native_country"].nunique() == 41


def test_only_the_three_documented_columns_have_missing_values(df):
    columns_with_missing = set(df.columns[df.isna().any()])
    assert columns_with_missing == {"workclass", "occupation", "native_country"}


def test_clean_data_rejects_a_dataframe_with_missing_columns():
    incomplete = pd.DataFrame({"age": [1], "income": ["<=50K"]})
    with pytest.raises(ValueError, match="missing expected columns"):
        clean_data(incomplete)


def test_clean_data_strips_trailing_period_and_question_marks():
    raw = pd.DataFrame(
        [
            {col: "?" if col in config.CATEGORICAL_FEATURES else 1 for col in config.ALL_COLUMNS}
            | {"income": "<=50K."},
            {col: "x" if col in config.CATEGORICAL_FEATURES else 2 for col in config.ALL_COLUMNS}
            | {"income": ">50K"},
        ]
    )
    cleaned = clean_data(raw)

    assert list(cleaned["income"]) == ["<=50K", ">50K"]
    assert cleaned.loc[0, "workclass"] is pd.NA or pd.isna(cleaned.loc[0, "workclass"])
    assert cleaned.loc[1, "workclass"] == "x"
