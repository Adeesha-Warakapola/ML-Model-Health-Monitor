"""Tests for building, training, persisting and reproducing the model.

Training on all 39,000 rows in every test would make the suite slow, so most
tests use a stratified sample of a few thousand rows. The sample keeps the
real class balance, so behaviour matches the full run.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from monitor import config, schema
from monitor.data import load_adult_data
from monitor.evaluation import evaluate_classifier
from monitor.model import (
    BASELINE_MODEL,
    MODEL_BUILDERS,
    apply_threshold,
    build_metadata,
    build_model_pipeline,
    extract_preprocessor,
    get_threshold,
    load_metadata,
    load_model,
    save_json,
    save_model,
    train_baseline,
    train_model,
    unwrap_pipeline,
)
from monitor.preprocessing import split_features_target
from tests.test_validation import make_valid_batch


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    """A stratified 3,000-row sample that keeps the natural class balance."""
    df = load_adult_data()
    small, _ = train_test_split(
        df,
        train_size=3000,
        random_state=config.RANDOM_SEED,
        stratify=df[config.TARGET_COLUMN],
    )
    return small


@pytest.fixture(scope="module")
def trained(sample):
    """A fitted baseline pipeline plus the splits it was trained on."""
    pipeline, train_df, test_df, messages = train_baseline(sample)
    return pipeline, train_df, test_df, messages


# --- building ---------------------------------------------------------------


def test_pipeline_bundles_preprocessing_and_classifier():
    pipeline = build_model_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocess", "classify"]
    assert isinstance(pipeline.named_steps["classify"], LogisticRegression)


def test_baseline_does_not_reweight_the_classes():
    # Requirement: the natural 76/24 balance must be preserved, so the model
    # must not quietly compensate with class weights.
    classifier = build_model_pipeline().named_steps["classify"]
    assert classifier.class_weight is None


def test_unknown_model_name_is_rejected_with_the_valid_options():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model_pipeline("magic_crystal_ball")


def test_every_registered_model_builds():
    for name in MODEL_BUILDERS:
        assert isinstance(build_model_pipeline(name), Pipeline)


def test_unfitted_pipeline_cannot_predict():
    with pytest.raises(NotFittedError):
        build_model_pipeline().predict(make_valid_batch())


# --- training ---------------------------------------------------------------


def test_training_produces_a_model_that_predicts_known_labels(trained):
    pipeline, _, test_df, _ = trained
    X_test, _ = split_features_target(test_df)

    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset(set(config.TARGET_LABELS))


def test_training_preserves_the_class_distribution_in_both_splits(sample, trained):
    _, train_df, test_df, _ = trained
    full_rate = (sample[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean()

    for split in (train_df, test_df):
        rate = (split[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean()
        assert abs(rate - full_rate) < 0.01


def test_training_rejects_data_with_a_broken_schema():
    broken = make_valid_batch(10).drop(columns=["age"])
    y = pd.Series(["<=50K", ">50K"] * 5)

    with pytest.raises(ValueError, match="failed validation"):
        train_model(broken, y)


def test_training_rejects_single_class_labels():
    X = make_valid_batch(10)
    y = pd.Series(["<=50K"] * 10)

    with pytest.raises(ValueError, match="both income classes"):
        train_model(X, y)


def test_preprocessor_is_fitted_on_training_rows_only(trained):
    # The scaler's mean must match the training split, not the whole sample.
    pipeline, train_df, test_df, _ = trained
    X_train, _ = split_features_target(train_df)
    scaler = extract_preprocessor(pipeline).named_transformers_["numeric"].named_steps["scale"]

    age_index = schema.NUMERIC_COLUMNS.index("age")
    assert scaler.mean_[age_index] == pytest.approx(X_train["age"].mean())

    combined_mean = pd.concat([train_df, test_df])["age"].mean()
    assert scaler.mean_[age_index] != pytest.approx(combined_mean)


def test_extract_preprocessor_returns_the_fitted_transformer(trained):
    pipeline, train_df, _, _ = trained
    preprocessor = extract_preprocessor(pipeline)

    X_train, _ = split_features_target(train_df)
    transformed = preprocessor.transform(X_train.head(5))

    assert transformed.shape[0] == 5
    assert len(preprocessor.get_feature_names_out()) == transformed.shape[1]


# --- reproducibility --------------------------------------------------------


def test_training_twice_with_the_same_seed_gives_identical_predictions(sample):
    first, _, test_df, _ = train_baseline(sample)
    second, _, _, _ = train_baseline(sample)
    X_test, _ = split_features_target(test_df)

    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))


def test_training_twice_gives_identical_probabilities(sample):
    first, _, test_df, _ = train_baseline(sample)
    second, _, _, _ = train_baseline(sample)
    X_test, _ = split_features_target(test_df)

    np.testing.assert_allclose(
        first.predict_proba(X_test), second.predict_proba(X_test)
    )


def test_the_split_itself_is_reproducible(sample):
    _, train_a, test_a, _ = train_baseline(sample)
    _, train_b, test_b, _ = train_baseline(sample)

    assert list(train_a.index) == list(train_b.index)
    assert list(test_a.index) == list(test_b.index)


def test_train_and_test_splits_do_not_overlap(trained):
    _, train_df, test_df, _ = trained
    assert set(train_df.index).isdisjoint(set(test_df.index))


# --- saving and loading -----------------------------------------------------


def test_saved_model_reloads_and_predicts_identically(trained, tmp_path):
    pipeline, _, test_df, _ = trained
    X_test, _ = split_features_target(test_df)
    path = tmp_path / "model.joblib"

    save_model(pipeline, path)
    reloaded = load_model(path)

    np.testing.assert_array_equal(pipeline.predict(X_test), reloaded.predict(X_test))
    np.testing.assert_allclose(
        pipeline.predict_proba(X_test), reloaded.predict_proba(X_test)
    )


def test_saved_model_includes_preprocessing_so_raw_rows_work(trained, tmp_path):
    # The whole point of saving the bundled pipeline: a reloaded model must
    # accept raw, unprocessed Adult-shaped rows.
    pipeline, _, _, _ = trained
    path = tmp_path / "model.joblib"
    save_model(pipeline, path)

    raw_batch = make_valid_batch(3)
    predictions = load_model(path).predict(raw_batch)

    assert len(predictions) == 3


def test_loading_a_missing_model_explains_how_to_create_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_baseline.py"):
        load_model(tmp_path / "nope.joblib")


def test_loading_missing_metadata_explains_how_to_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_baseline.py"):
        load_metadata(tmp_path / "nope.json")


def test_saved_model_handles_unknown_categories_after_reload(trained, tmp_path):
    pipeline, _, _, _ = trained
    path = tmp_path / "model.joblib"
    save_model(pipeline, path)

    batch = make_valid_batch(1)
    batch.loc[0, "native_country"] = "Atlantis"

    assert len(load_model(path).predict(batch)) == 1


# --- decision threshold -----------------------------------------------------


def test_plain_pipeline_uses_the_default_threshold(trained):
    pipeline, _, _, _ = trained
    assert get_threshold(pipeline) == 0.5


def test_applied_threshold_is_reported_and_used(sample):
    model, _, test_df, _ = train_baseline(sample, threshold=0.3)
    X_test, _ = split_features_target(test_df)

    assert get_threshold(model) == 0.3

    # predict must agree with applying the threshold to the scores by hand.
    positive_column = list(model.classes_).index(config.POSITIVE_LABEL)
    scores = model.predict_proba(X_test)[:, positive_column]
    expected = np.where(scores >= 0.3, config.POSITIVE_LABEL, "<=50K")
    np.testing.assert_array_equal(model.predict(X_test), expected)


def test_lower_threshold_predicts_the_positive_class_more_often(sample):
    _, _, test_df, _ = train_baseline(sample)
    X_test, _ = split_features_target(test_df)

    cautious, _, _, _ = train_baseline(sample, threshold=0.6)
    eager, _, _, _ = train_baseline(sample, threshold=0.25)

    cautious_rate = (cautious.predict(X_test) == config.POSITIVE_LABEL).mean()
    eager_rate = (eager.predict(X_test) == config.POSITIVE_LABEL).mean()
    assert eager_rate > cautious_rate


def test_threshold_outside_zero_to_one_is_rejected():
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            apply_threshold(build_model_pipeline(), bad)


def test_threshold_does_not_change_the_ranking(sample):
    # Only the cut-off moves, so ROC-AUC - which ignores the cut-off - must
    # be identical. This is what makes threshold tuning a safe change.
    plain, _, test_df, _ = train_baseline(sample)
    tuned, _, _, _ = train_baseline(sample, threshold=0.3)
    X_test, y_test = split_features_target(test_df)

    assert evaluate_classifier(plain, X_test, y_test)["roc_auc"] == pytest.approx(
        evaluate_classifier(tuned, X_test, y_test)["roc_auc"]
    )


def test_preprocessor_is_reachable_through_the_threshold_wrapper(sample):
    model, _, _, _ = train_baseline(sample, threshold=0.3)
    inner = unwrap_pipeline(model)
    preprocessor = extract_preprocessor(model)

    assert isinstance(inner, Pipeline)
    # Must be the very same fitted object the inner pipeline uses, not a copy.
    assert preprocessor is inner.named_steps["preprocess"]
    # One-hot encoding expands the raw columns, so there are always more.
    assert len(preprocessor.get_feature_names_out()) > len(schema.FEATURE_COLUMNS)


def test_thresholded_model_survives_save_and_load(sample, tmp_path):
    model, _, test_df, _ = train_baseline(sample, threshold=0.3)
    X_test, _ = split_features_target(test_df)
    path = tmp_path / "thresholded.joblib"

    save_model(model, path)
    reloaded = load_model(path)

    assert get_threshold(reloaded) == 0.3
    np.testing.assert_array_equal(model.predict(X_test), reloaded.predict(X_test))


# --- metadata ---------------------------------------------------------------


def test_metadata_records_everything_needed_to_understand_the_artifact(
    sample, trained
):
    pipeline, train_df, test_df, _ = trained
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    metadata = build_metadata(
        pipeline,
        BASELINE_MODEL,
        evaluate_classifier(pipeline, X_train, y_train),
        evaluate_classifier(pipeline, X_test, y_test),
        n_rows_total=len(sample),
    )

    assert metadata["model_type"] == "LogisticRegression"
    assert metadata["random_seed"] == config.RANDOM_SEED
    assert metadata["target_column"] == config.TARGET_COLUMN
    assert metadata["input_feature_names"] == schema.FEATURE_COLUMNS
    assert metadata["dataset"]["n_rows_total"] == len(sample)
    assert metadata["trained_on"]  # ISO date string
    assert metadata["versions"]["scikit_learn"]
    assert 0 < metadata["test_metrics"]["accuracy"] <= 1


def test_metadata_lists_the_encoded_feature_names(trained):
    pipeline, _, _, _ = trained
    metadata_names = build_metadata(
        pipeline, BASELINE_MODEL, {}, {}, n_rows_total=100
    )["encoded_feature_names"]

    assert metadata_names == list(extract_preprocessor(pipeline).get_feature_names_out())
    assert len(metadata_names) > len(schema.FEATURE_COLUMNS)  # one-hot expands them


def test_metadata_is_json_serialisable(trained, tmp_path):
    pipeline, train_df, test_df, _ = trained
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    metadata = build_metadata(
        pipeline,
        BASELINE_MODEL,
        evaluate_classifier(pipeline, X_train, y_train),
        evaluate_classifier(pipeline, X_test, y_test),
        n_rows_total=len(train_df) + len(test_df),
    )

    path = tmp_path / "metadata.json"
    save_json(metadata, path)

    assert json.loads(path.read_text())["model_type"] == "LogisticRegression"
