"""Tests for batch validation.

Most tests build a small DataFrame with one deliberate defect, so it is
obvious which check is being exercised.
"""

import pandas as pd
import pytest

from monitor import schema
from monitor.data import load_adult_data
from monitor.validation import ERROR, WARNING, validate_batch


def make_valid_batch(n_rows: int = 3) -> pd.DataFrame:
    """A small batch that satisfies every rule in the schema."""
    row = {
        "age": 39,
        "fnlwgt": 77516,
        "education_num": 13,
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
    }
    return pd.DataFrame([row] * n_rows)


def checks_in(result) -> set[str]:
    return {issue.check for issue in result.issues}


# --- valid data ------------------------------------------------------------


def test_clean_batch_passes_with_no_issues():
    result = validate_batch(make_valid_batch())

    assert result.is_valid
    assert result.issues == []
    assert result.n_rows == 3
    assert "PASSED" in result.summary()


def test_the_real_dataset_passes_validation():
    # The reference data must satisfy its own schema, otherwise the schema
    # is wrong. Missing values are expected, so warnings are allowed here.
    df = load_adult_data()
    result = validate_batch(df[schema.FEATURE_COLUMNS])

    assert result.is_valid, result.summary()
    assert checks_in(result) == {"missing_values"}


def test_schema_matches_the_real_dataset():
    # Verifies the hard-coded allowed_values against the actual data, so the
    # schema cannot silently drift away from what it claims to describe.
    df = load_adult_data()
    derived = schema.derive_categories(df)

    for column, categories in derived.items():
        assert categories == schema.FEATURE_SPECS[column].allowed_values, column


# --- missing values --------------------------------------------------------


def test_missing_values_are_reported_as_warnings_not_errors():
    batch = make_valid_batch()
    batch.loc[0, "occupation"] = None

    result = validate_batch(batch)

    assert result.is_valid  # imputer handles it, so the batch is still usable
    assert checks_in(result) == {"missing_values"}
    issue = result.warnings[0]
    assert issue.column == "occupation"
    assert issue.affected_rows == 1


def test_missing_value_percentage_is_reported():
    batch = make_valid_batch(n_rows=4)
    batch.loc[0, "workclass"] = None

    result = validate_batch(batch)

    assert "25.00%" in result.warnings[0].message


# --- unknown categories ----------------------------------------------------


def test_unknown_category_is_flagged_but_does_not_fail_the_batch():
    batch = make_valid_batch()
    batch.loc[0, "native_country"] = "Atlantis"

    result = validate_batch(batch)

    assert result.is_valid
    assert "unknown_categories" in checks_in(result)
    issue = next(i for i in result.issues if i.check == "unknown_categories")
    assert "Atlantis" in issue.message
    assert issue.affected_rows == 1


def test_multiple_unknown_categories_are_listed():
    batch = make_valid_batch(n_rows=2)
    batch.loc[0, "workclass"] = "Freelance"
    batch.loc[1, "workclass"] = "Gig-economy"

    result = validate_batch(batch)
    issue = next(i for i in result.issues if i.check == "unknown_categories")

    assert "Freelance" in issue.message
    assert "Gig-economy" in issue.message
    assert issue.affected_rows == 2


# --- invalid schemas -------------------------------------------------------


def test_missing_required_column_is_an_error():
    batch = make_valid_batch().drop(columns=["age"])

    result = validate_batch(batch)

    assert not result.is_valid
    assert result.errors[0].check == "required_columns"
    assert "age" in result.errors[0].message


def test_unexpected_column_is_only_a_warning():
    batch = make_valid_batch()
    batch["customer_id"] = [1, 2, 3]

    result = validate_batch(batch)

    assert result.is_valid
    issue = next(i for i in result.issues if i.check == "unexpected_columns")
    assert issue.severity == WARNING
    assert "customer_id" in issue.message


def test_numeric_column_holding_text_is_an_error():
    batch = make_valid_batch()
    batch["age"] = ["thirty", "forty", "fifty"]

    result = validate_batch(batch)

    assert not result.is_valid
    issue = next(i for i in result.errors if i.column == "age")
    assert issue.check == "dtype"


def test_categorical_column_holding_numbers_is_an_error():
    batch = make_valid_batch()
    batch["workclass"] = [1, 2, 3]

    result = validate_batch(batch)

    assert not result.is_valid
    assert any(i.column == "workclass" and i.check == "dtype" for i in result.errors)


def test_empty_batch_is_an_error():
    result = validate_batch(pd.DataFrame())

    assert not result.is_valid
    assert result.errors[0].check == "empty_batch"
    assert result.n_rows == 0


# --- value ranges ----------------------------------------------------------


def test_value_below_minimum_is_flagged():
    batch = make_valid_batch()
    batch.loc[0, "age"] = 5  # below the schema minimum of 17

    result = validate_batch(batch)
    issue = next(i for i in result.issues if i.check == "value_range")

    assert issue.column == "age"
    assert issue.affected_rows == 1
    assert "minimum" in issue.message


def test_value_above_maximum_is_flagged():
    batch = make_valid_batch()
    batch.loc[0, "hours_per_week"] = 200  # a week has only 168 hours

    result = validate_batch(batch)
    issue = next(i for i in result.issues if i.check == "value_range")

    assert issue.column == "hours_per_week"
    assert "maximum" in issue.message


def test_unbounded_column_accepts_large_values():
    # capital_loss has no documented upper bound, so a large value is not an
    # error - it is drift, which a later phase handles.
    batch = make_valid_batch()
    batch.loc[0, "capital_loss"] = 50_000

    result = validate_batch(batch)

    assert "value_range" not in checks_in(result)


# --- result object behaviour -----------------------------------------------


def test_raise_if_invalid_raises_only_on_errors():
    batch = make_valid_batch()
    batch.loc[0, "occupation"] = None
    validate_batch(batch).raise_if_invalid()  # warnings must not raise

    broken = make_valid_batch().drop(columns=["sex"])
    with pytest.raises(ValueError, match="failed validation"):
        validate_batch(broken).raise_if_invalid()


def test_summary_lists_every_issue():
    batch = make_valid_batch()
    batch.loc[0, "occupation"] = None
    batch["extra"] = 1

    summary = validate_batch(batch).summary()

    assert "occupation" in summary
    assert "extra" in summary
    assert summary.count("\n") >= 2
