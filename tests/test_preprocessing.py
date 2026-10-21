"""Tests for the preprocessing pipeline.

The important behaviours here are that fitting happens on training data only,
and that a later batch containing something unexpected is still transformed
into the same shaped output.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from monitor import config, schema
from monitor.data import load_adult_data
from monitor.preprocessing import (
    MISSING_CATEGORY,
    build_preprocessor,
    fit_preprocessor,
    get_feature_names,
    load_preprocessor,
    save_preprocessor,
    split_features_target,
    split_train_test,
    transform_batch,
)
from tests.test_validation import make_valid_batch


@pytest.fixture(scope="module")
def adult():
    return load_adult_data()


@pytest.fixture(scope="module")
def fitted(adult):
    """A preprocessor fitted on the training split only."""
    train_df, _ = split_train_test(adult)
    X_train, _ = split_features_target(train_df)
    return fit_preprocessor(X_train)


# --- valid data ------------------------------------------------------------


def test_transform_produces_one_row_per_input_row(fitted):
    batch = make_valid_batch(n_rows=5)
    result = transform_batch(fitted, batch)

    assert len(result) == 5
    assert result.notna().all().all()


def test_output_columns_are_named_and_stable(fitted):
    first = transform_batch(fitted, make_valid_batch(2))
    second = transform_batch(fitted, make_valid_batch(7))

    assert list(first.columns) == list(second.columns)
    assert list(first.columns) == get_feature_names(fitted)
    assert any(name.startswith("numeric__") for name in first.columns)
    assert any(name.startswith("categorical__") for name in first.columns)


def test_numeric_columns_are_scaled(adult, fitted):
    # Fitted on the training split, so the training data itself should come
    # out roughly centred on zero.
    train_df, _ = split_train_test(adult)
    X_train, _ = split_features_target(train_df)
    result = transform_batch(fitted, X_train)

    age = result["numeric__age"]
    assert abs(age.mean()) < 0.01
    assert abs(age.std() - 1.0) < 0.01


def test_transformed_output_keeps_the_original_index(fitted):
    batch = make_valid_batch(3)
    batch.index = [100, 200, 300]

    result = transform_batch(fitted, batch)

    assert list(result.index) == [100, 200, 300]


# --- no data leakage -------------------------------------------------------


def test_unfitted_preprocessor_refuses_to_transform():
    with pytest.raises(NotFittedError):
        transform_batch(build_preprocessor(), make_valid_batch())


def test_transforming_a_batch_does_not_refit_the_preprocessor(fitted):
    # The scaler's learned mean must be identical before and after a batch
    # with wildly different values passes through it.
    scaler = fitted.named_transformers_["numeric"].named_steps["scale"]
    before = scaler.mean_.copy()

    extreme = make_valid_batch(5)
    extreme["age"] = 90
    extreme["hours_per_week"] = 99
    transform_batch(fitted, extreme)

    np.testing.assert_array_equal(before, scaler.mean_)


def test_train_test_split_preserves_the_class_distribution(adult):
    train_df, test_df = split_train_test(adult)
    full_rate = (adult[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean()

    train_rate = (train_df[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean()
    test_rate = (test_df[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean()

    assert len(train_df) + len(test_df) == len(adult)
    assert abs(train_rate - full_rate) < 0.005
    assert abs(test_rate - full_rate) < 0.005


def test_split_is_reproducible(adult):
    first, _ = split_train_test(adult)
    second, _ = split_train_test(adult)
    assert list(first.index) == list(second.index)


# --- missing values --------------------------------------------------------


def test_missing_categorical_becomes_its_own_category(fitted):
    batch = make_valid_batch(2)
    batch.loc[0, "occupation"] = None

    result = transform_batch(fitted, batch)
    missing_column = f"categorical__occupation_{MISSING_CATEGORY}"

    assert missing_column in result.columns
    assert result.loc[0, missing_column] == 1.0
    assert result.loc[1, missing_column] == 0.0


def test_missing_value_in_a_never_missing_column_degrades_safely(fitted):
    # Only workclass, occupation and native_country contain missing values in
    # the training data, so only they learn a "Missing" category. A missing
    # `sex` is filled with "Missing", which the encoder has never seen for
    # that column, so handle_unknown="ignore" emits all-zeros instead of
    # crashing or silently picking a real category.
    batch = make_valid_batch(1)
    batch.loc[0, "sex"] = None

    result = transform_batch(fitted, batch)
    sex_columns = [c for c in result.columns if c.startswith("categorical__sex_")]

    assert sex_columns == ["categorical__sex_Female", "categorical__sex_Male"]
    assert result[sex_columns].sum(axis=1).iloc[0] == 0.0


def test_missing_numeric_is_imputed_not_dropped(fitted):
    batch = make_valid_batch(2)
    batch.loc[0, "age"] = np.nan

    result = transform_batch(fitted, batch)

    assert len(result) == 2
    assert result["numeric__age"].notna().all()


# --- unknown categories ----------------------------------------------------


def test_unknown_category_encodes_as_all_zeros(fitted):
    batch = make_valid_batch(1)
    batch.loc[0, "native_country"] = "Atlantis"

    result = transform_batch(fitted, batch)
    country_columns = [
        c for c in result.columns if c.startswith("categorical__native_country_")
    ]

    assert result[country_columns].sum(axis=1).iloc[0] == 0.0


def test_unknown_category_does_not_change_the_output_shape(fitted):
    known = transform_batch(fitted, make_valid_batch(1))

    unknown_batch = make_valid_batch(1)
    unknown_batch.loc[0, "workclass"] = "Gig-economy"
    unknown = transform_batch(fitted, unknown_batch)

    assert known.shape == unknown.shape


# --- invalid input ---------------------------------------------------------


def test_fit_rejects_a_frame_with_missing_columns():
    incomplete = make_valid_batch().drop(columns=["age", "sex"])
    with pytest.raises(ValueError, match="columns are missing"):
        fit_preprocessor(incomplete)


def test_transform_rejects_a_batch_with_missing_columns(fitted):
    incomplete = make_valid_batch().drop(columns=["race"])
    with pytest.raises(ValueError, match="columns are missing"):
        transform_batch(fitted, incomplete)


def test_extra_columns_are_dropped_not_passed_through(fitted):
    batch = make_valid_batch(2)
    batch["customer_id"] = [1, 2]

    result = transform_batch(fitted, batch)

    assert not any("customer_id" in name for name in result.columns)


def test_split_features_target_requires_the_target_column():
    with pytest.raises(ValueError, match="income"):
        split_features_target(make_valid_batch())


# --- persistence -----------------------------------------------------------


def test_saved_and_reloaded_preprocessor_gives_identical_output(fitted, tmp_path):
    path = tmp_path / "preprocessor.joblib"
    save_preprocessor(fitted, path)
    reloaded = load_preprocessor(path)

    batch = make_valid_batch(4)
    np.testing.assert_array_almost_equal(
        transform_batch(fitted, batch).to_numpy(),
        transform_batch(reloaded, batch).to_numpy(),
    )


def test_loading_a_missing_preprocessor_explains_how_to_create_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="validate_and_preprocess"):
        load_preprocessor(tmp_path / "does_not_exist.joblib")


def test_every_schema_feature_appears_in_the_output(fitted):
    names = get_feature_names(fitted)
    for column in schema.FEATURE_COLUMNS:
        assert any(column in name for name in names), column
