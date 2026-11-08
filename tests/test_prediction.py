"""Tests for the prediction service.

The most important of these is label separation: the service must refuse a
batch that still carries the true income column, because at real prediction
time that column does not exist yet.
"""

import pandas as pd
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from monitor import config, schema
from monitor.data import load_adult_data
from monitor.prediction import (
    PredictionService,
    make_batch_id,
    utc_now,
)
from monitor.preprocessing import build_preprocessor, split_features_target
from tests.test_validation import make_valid_batch


@pytest.fixture(scope="module")
def sample():
    df = load_adult_data()
    small, _ = train_test_split(
        df, train_size=2000, random_state=config.RANDOM_SEED,
        stratify=df[config.TARGET_COLUMN],
    )
    return small


@pytest.fixture(scope="module")
def service(sample):
    """A service wrapping a model trained on the sample, so tests stay fast."""
    from monitor.model import train_model

    X, y = split_features_target(sample)
    model, _ = train_model(X, y, threshold=0.29)
    return PredictionService(model=model)


# --- batch prediction -------------------------------------------------------


def test_predicts_one_label_per_row(service):
    batch = make_valid_batch(6)
    result = service.predict_batch(batch)

    assert result.n_rows == 6
    assert len(result.predictions) == 6
    assert set(result.predictions).issubset(set(config.TARGET_LABELS))


def test_returns_probabilities_for_the_positive_class(service):
    result = service.predict_batch(make_valid_batch(5))

    assert result.probabilities is not None
    assert len(result.probabilities) == 5
    assert ((result.probabilities >= 0) & (result.probabilities <= 1)).all()


def test_predictions_agree_with_the_probability_and_threshold(service):
    # The service reports a threshold; predictions must be consistent with it.
    real = load_adult_data().head(200)[schema.FEATURE_COLUMNS]
    result = service.predict_batch(real)

    predicted_positive = result.predictions == config.POSITIVE_LABEL
    above_threshold = result.probabilities >= result.threshold
    pd.testing.assert_series_equal(
        predicted_positive, above_threshold, check_names=False
    )


def test_original_features_are_preserved_alongside_predictions(service):
    batch = make_valid_batch(4)
    result = service.predict_batch(batch)

    # Needed for monitoring: the inputs must still be inspectable afterwards.
    pd.testing.assert_frame_equal(result.features, batch)
    assert list(result.features.columns) == list(batch.columns)


def test_to_frame_joins_features_predictions_and_batch_identity(service):
    result = service.predict_batch(make_valid_batch(3))
    frame = result.to_frame()

    for column in schema.FEATURE_COLUMNS:
        assert column in frame.columns
    assert "prediction" in frame.columns
    assert f"probability_{config.POSITIVE_LABEL}" in frame.columns
    assert (frame["batch_id"] == result.batch_id).all()
    assert (frame["predicted_at"] == result.predicted_at).all()
    assert len(frame) == 3


def test_row_index_is_preserved_so_rows_stay_traceable(service):
    batch = make_valid_batch(3)
    batch.index = [7, 8, 9]

    result = service.predict_batch(batch)

    assert list(result.predictions.index) == [7, 8, 9]
    assert list(result.probabilities.index) == [7, 8, 9]
    assert list(result.to_frame().index) == [7, 8, 9]


def test_prediction_distribution_percentages_add_up(service):
    result = service.predict_batch(load_adult_data().head(300)[schema.FEATURE_COLUMNS])
    distribution = result.prediction_distribution()

    assert sum(stats["count"] for stats in distribution.values()) == 300
    assert abs(sum(stats["percent"] for stats in distribution.values()) - 100) < 0.05


def test_summary_mentions_the_batch_and_the_model(service):
    result = service.predict_batch(make_valid_batch(2))
    text = result.summary()

    assert result.batch_id in text
    assert "LogisticRegression" in text


# --- label separation -------------------------------------------------------


def test_service_refuses_a_batch_containing_the_label_column(service):
    batch = make_valid_batch(3)
    batch[config.TARGET_COLUMN] = ["<=50K", ">50K", "<=50K"]

    with pytest.raises(ValueError, match="never receive true labels"):
        service.predict_batch(batch)


def test_predict_simulated_passes_only_features(service, sample):
    from monitor.simulation import BatchSimulator

    simulator = BatchSimulator(sample, batch_size=50, seed=1)
    batch = simulator.next_batch()
    result = service.predict_simulated(batch)

    # Labels stay on the batch and never reach the prediction record.
    assert config.TARGET_COLUMN not in result.features.columns
    assert config.TARGET_COLUMN not in result.to_frame().columns
    assert len(batch.labels) == result.n_rows


def test_prediction_record_never_contains_the_true_label(service):
    real = load_adult_data().head(50)
    result = service.predict_batch(real[schema.FEATURE_COLUMNS])

    assert config.TARGET_COLUMN not in result.to_frame().columns


# --- repeatability ----------------------------------------------------------


def test_scoring_the_same_batch_twice_gives_identical_results(service):
    batch = load_adult_data().head(200)[schema.FEATURE_COLUMNS]

    first = service.predict_batch(batch)
    second = service.predict_batch(batch)

    pd.testing.assert_series_equal(first.predictions, second.predictions)
    pd.testing.assert_series_equal(first.probabilities, second.probabilities)


def test_a_reloaded_service_predicts_the_same_way(service, tmp_path):
    from monitor.model import save_model

    path = tmp_path / "model.joblib"
    save_model(service.model, path)
    reloaded = PredictionService(model_path=path)

    batch = make_valid_batch(10)
    pd.testing.assert_series_equal(
        service.predict_batch(batch).predictions,
        reloaded.predict_batch(batch).predictions,
    )


def test_row_order_does_not_change_an_individual_prediction(service):
    batch = load_adult_data().head(100)[schema.FEATURE_COLUMNS]
    forward = service.predict_batch(batch).predictions
    reversed_batch = batch.iloc[::-1]
    backward = service.predict_batch(reversed_batch).predictions

    # Same rows, opposite order: each row must keep its own prediction.
    pd.testing.assert_series_equal(forward, backward.sort_index())


# --- batch metadata ---------------------------------------------------------


def test_each_batch_gets_a_unique_id(service):
    batch = make_valid_batch(2)
    first = service.predict_batch(batch).batch_id
    second = service.predict_batch(batch).batch_id

    assert first != second


def test_batch_id_is_prefixed_and_timestamped():
    batch_id = make_batch_id()
    prefix, stamp, suffix = batch_id.split("_")

    assert prefix == "batch"
    assert stamp.endswith("Z") and len(stamp) == 16  # 20261108T211530Z
    assert len(suffix) == 8


def test_batch_id_can_be_supplied_for_traceability(service):
    result = service.predict_batch(make_valid_batch(2), batch_id="batch_fixed_001")
    assert result.batch_id == "batch_fixed_001"


def test_timestamp_is_iso_8601_and_parseable(service):
    from datetime import datetime

    result = service.predict_batch(make_valid_batch(2))
    parsed = datetime.fromisoformat(result.predicted_at)

    assert parsed.tzinfo is not None  # timezone-aware, so it is unambiguous
    assert datetime.fromisoformat(utc_now()).tzinfo is not None


def test_metadata_records_the_model_and_threshold(service):
    result = service.predict_batch(make_valid_batch(2))

    assert result.model_type == "LogisticRegression"
    assert result.threshold == 0.29


# --- unknown categories -----------------------------------------------------


def test_unknown_category_is_scored_without_failing(service):
    batch = make_valid_batch(3)
    batch.loc[0, "occupation"] = "Prompt-Engineer"

    result = service.predict_batch(batch)

    assert result.n_rows == 3
    assert result.validation.is_valid  # a warning, not an error


def test_unknown_category_is_reported_in_the_validation_record(service):
    batch = make_valid_batch(3)
    batch.loc[0, "native_country"] = "Atlantis"

    result = service.predict_batch(batch)
    issues = [i for i in result.validation.issues if i.check == "unknown_categories"]

    assert len(issues) == 1
    assert "Atlantis" in issues[0].message


def test_missing_values_in_a_batch_are_scored(service):
    batch = make_valid_batch(3)
    batch.loc[0, "workclass"] = None

    result = service.predict_batch(batch)

    assert result.n_rows == 3
    assert result.predictions.notna().all()


# --- error handling ---------------------------------------------------------


def test_missing_model_file_explains_how_to_create_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_baseline.py"):
        PredictionService(model_path=tmp_path / "absent.joblib")


def test_empty_batch_is_rejected(service):
    with pytest.raises(ValueError, match="empty batch"):
        service.predict_batch(pd.DataFrame(columns=schema.FEATURE_COLUMNS))


def test_batch_missing_a_required_column_is_rejected(service):
    with pytest.raises(ValueError, match="failed validation"):
        service.predict_batch(make_valid_batch(3).drop(columns=["age"]))


def test_batch_with_a_wrongly_typed_column_is_rejected(service):
    batch = make_valid_batch(3)
    batch["age"] = ["old", "older", "oldest"]

    with pytest.raises(ValueError, match="failed validation"):
        service.predict_batch(batch)


def test_extra_columns_are_tolerated(service):
    batch = make_valid_batch(3)
    batch["internal_id"] = [1, 2, 3]

    result = service.predict_batch(batch)

    assert result.n_rows == 3
    assert any(i.check == "unexpected_columns" for i in result.validation.issues)


# --- models without probabilities -------------------------------------------


def test_model_without_predict_proba_still_predicts(sample):
    # RidgeClassifier has no predict_proba, so the service must degrade to
    # labels only instead of raising.
    X, y = split_features_target(sample)
    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("classify", RidgeClassifier())]
    )
    pipeline.fit(X, y)
    service = PredictionService(model=pipeline)

    result = service.predict_batch(make_valid_batch(4))

    assert service.supports_probabilities is False
    assert result.probabilities is None
    assert result.n_rows == 4
    assert any("predict_proba" in w for w in result.warnings)


def test_to_frame_omits_the_probability_column_when_unavailable(sample):
    X, y = split_features_target(sample)
    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("classify", RidgeClassifier())]
    )
    pipeline.fit(X, y)

    frame = PredictionService(model=pipeline).predict_batch(make_valid_batch(2)).to_frame()

    assert f"probability_{config.POSITIVE_LABEL}" not in frame.columns
    assert "prediction" in frame.columns
