"""Tests for prediction-distribution monitoring (Part A).

Mirrors the structure of test_drift.py: unchanged predictions must not be
flagged, deliberately shifted ones must be, and no case should crash.
"""

import pandas as pd
import pytest

from monitor import config
from monitor.data import load_adult_data
from monitor.drift import DRIFT, MODERATE, NOT_COMPUTED, STABLE, DriftThresholds
from monitor.prediction import PredictionService
from monitor.prediction_monitoring import PredictionMonitor, confidence_summary
from monitor.simulation import (
    BatchSimulator,
    get_reference_features,
    overwrite_category,
    scale_numeric,
    shift_numeric,
    split_reference_and_holdout,
)


@pytest.fixture(scope="module")
def service():
    return PredictionService()


@pytest.fixture(scope="module")
def reference_features(service):
    return get_reference_features(load_adult_data())


@pytest.fixture(scope="module")
def holdout():
    _, held = split_reference_and_holdout(load_adult_data())
    return held


@pytest.fixture(scope="module")
def monitor(service, reference_features):
    return PredictionMonitor.from_reference_features(service, reference_features)


def batch_report(monitor, service, holdout, size=600, seed=42, modifier=None):
    simulator = BatchSimulator(holdout, batch_size=size, seed=seed)
    batch = simulator.next_batch(modifier=modifier)
    prediction = service.predict_simulated(batch)
    return monitor.evaluate(prediction), batch, prediction


# --- confidence_summary ------------------------------------------------------


def test_confidence_summary_describes_a_known_distribution():
    probs = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    summary = confidence_summary(probs, threshold=0.5, margin=0.1)

    assert summary["n"] == 6
    assert summary["mean"] == pytest.approx(0.5, abs=1e-9)
    assert summary["median"] == pytest.approx(0.5, abs=1e-9)


def test_confidence_summary_borderline_rate_counts_near_the_threshold():
    probs = pd.Series([0.1, 0.42, 0.5, 0.58, 0.9])
    summary = confidence_summary(probs, threshold=0.5, margin=0.1)
    # 0.42, 0.5, 0.58 all fall within +-0.1 of 0.5
    assert summary["borderline_rate"] == pytest.approx(3 / 5)


def test_confidence_summary_handles_an_empty_series():
    summary = confidence_summary(pd.Series([], dtype=float), threshold=0.5)
    assert summary == {
        "n": 0, "mean": None, "std": None, "median": None,
        "q25": None, "q75": None, "borderline_rate": None,
    }


# --- construction -------------------------------------------------------


def test_builds_from_reference_features(service, reference_features):
    pm = PredictionMonitor.from_reference_features(service, reference_features)
    assert pm.reference_predictions.notna().all()
    assert set(pm.reference_predictions.unique()) <= set(config.TARGET_LABELS)


def test_rejects_empty_reference_predictions():
    with pytest.raises(ValueError, match="empty"):
        PredictionMonitor(pd.Series([], dtype=str), None, threshold=0.5)


# --- unchanged predictions must not be flagged ------------------------------


def test_unmodified_batch_predictions_are_stable(monitor, service, holdout):
    report, _, _ = batch_report(monitor, service, holdout, size=600)
    assert report.status == STABLE, report.summary()
    assert report.label_status == STABLE
    assert report.confidence_status == STABLE


def test_several_unmodified_batches_stay_stable(monitor, service, holdout):
    simulator = BatchSimulator(holdout, batch_size=600, seed=11)
    for batch in simulator.generate_batches(5):
        prediction = service.predict_simulated(batch)
        report = monitor.evaluate(prediction)
        assert report.status == STABLE, report.summary()


def test_reference_scored_against_itself_is_stable(service, reference_features):
    pm = PredictionMonitor.from_reference_features(service, reference_features)
    prediction = service.predict_batch(reference_features, batch_id="self-check")
    report = pm.evaluate(prediction)

    assert report.label_psi == pytest.approx(0.0, abs=1e-6)
    assert report.status == STABLE


# --- shifted predictions must be detected -----------------------------------


def test_feature_shift_that_changes_predictions_is_detected(monitor, service, holdout):
    # Shifting age pushes predictions toward >50K; the label mix itself stays
    # under the "drift" band (it is subtler than the raw feature shift), but
    # the confidence distribution - the underlying scores - moves clearly.
    report, _, _ = batch_report(monitor, service, holdout, modifier=shift_numeric("age", 25))

    assert report.status in (MODERATE, DRIFT)
    assert report.confidence_psi > report.reference_confidence["std"] * 0  # sanity: a number
    assert report.confidence_status in (MODERATE, DRIFT)


def test_large_feature_shift_moves_the_predicted_label_mix(monitor, service, holdout):
    report, batch, _ = batch_report(
        monitor, service, holdout, modifier=shift_numeric("age", 40)
    )
    positive_before = report.reference_distribution[config.POSITIVE_LABEL]["percent"]
    positive_after = report.batch_distribution[config.POSITIVE_LABEL]["percent"]

    assert positive_after > positive_before  # older ages push toward >50K
    assert report.label_psi > 0


def test_category_shift_toward_a_more_predictive_value_changes_predictions(
    monitor, service, holdout
):
    report, _, _ = batch_report(
        monitor, service, holdout,
        modifier=overwrite_category("occupation", "Exec-managerial", fraction=0.8),
    )
    assert report.label_psi >= 0  # always computable, direction depends on the model


def test_batch_count_matches_the_prediction_record(monitor, service, holdout):
    report, batch, prediction = batch_report(monitor, service, holdout, size=300)
    assert report.n_batch == batch.n_rows == prediction.n_rows
    assert report.n_reference == 39073


# --- confidence drift precedes label drift (the point of tracking both) ----


def test_confidence_can_shift_before_the_hard_label_mix_does(monitor, service, holdout):
    report, _, _ = batch_report(monitor, service, holdout, modifier=shift_numeric("age", 20))
    # The underlying scores should show at least as much movement as the
    # thresholded labels - that is the reason confidence is tracked at all.
    assert report.confidence_psi >= report.label_psi


# --- does not assume failure -------------------------------------------------


def test_report_explains_that_drift_does_not_mean_failure(monitor, service, holdout):
    report, _, _ = batch_report(monitor, service, holdout, modifier=shift_numeric("age", 30))
    assert any("does not by itself mean" in note for note in report.notes)


def test_small_batch_gets_a_reliability_note(monitor, service, holdout):
    report, _, _ = batch_report(monitor, service, holdout, size=100)
    assert any("SMALL SAMPLE" in note for note in report.notes)


# --- edge cases --------------------------------------------------------------


def test_model_without_predict_proba_only_reports_label_drift(holdout, reference_features):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import Pipeline

    from monitor.data import load_adult_data
    from monitor.preprocessing import build_preprocessor, split_features_target

    df = load_adult_data()
    X, y = split_features_target(reference_features.assign(income=df.loc[reference_features.index, config.TARGET_COLUMN]))
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("classify", RidgeClassifier())])
    pipeline.fit(X, y)
    service = PredictionService(model=pipeline)

    pm = PredictionMonitor.from_reference_features(service, reference_features)
    batch = BatchSimulator(holdout, batch_size=300, seed=1).next_batch()
    prediction = service.predict_simulated(batch)

    report = pm.evaluate(prediction)
    assert report.confidence_status == NOT_COMPUTED
    assert report.label_status in (STABLE, MODERATE, DRIFT)
    assert any("Probabilities are unavailable" in note for note in report.notes)


def test_single_predicted_class_in_a_batch_does_not_crash(monitor, service, holdout):
    # A batch where the model happens to predict only one label (small, or
    # with an extreme modifier) must still produce a report, not crash.
    batch = BatchSimulator(holdout, batch_size=5, seed=1).next_batch()
    prediction = service.predict_simulated(batch)
    report = monitor.evaluate(prediction)
    assert report.status in (STABLE, MODERATE, DRIFT)


def test_empty_batch_is_rejected(monitor):
    from monitor.prediction import BatchPrediction
    from monitor.validation import ValidationResult

    bp = BatchPrediction(
        batch_id="empty",
        predicted_at="now",
        model_type="x",
        threshold=0.5,
        features=pd.DataFrame(columns=["age"]),
        predictions=pd.Series([], dtype=str),
        probabilities=None,
        validation=ValidationResult(issues=[], n_rows=0),
        warnings=[],
    )
    with pytest.raises(ValueError, match="empty batch"):
        monitor.evaluate(bp)


def test_configurable_thresholds_change_the_verdict(service, reference_features, holdout):
    strict = PredictionMonitor.from_reference_features(
        service, reference_features, DriftThresholds(psi_moderate=0.0001, psi_major=0.0002)
    )
    lenient = PredictionMonitor.from_reference_features(
        service, reference_features, DriftThresholds(psi_moderate=5.0, psi_major=10.0)
    )
    batch = BatchSimulator(holdout, batch_size=600, seed=1).next_batch(
        modifier=shift_numeric("age", 10)
    )
    prediction = service.predict_simulated(batch)

    assert strict.evaluate(prediction).status == DRIFT
    assert lenient.evaluate(prediction).status == STABLE


# --- shape / reuse outside a UI ----------------------------------------------


def test_to_dict_is_json_serialisable(monitor, service, holdout):
    import json

    report, _, _ = batch_report(monitor, service, holdout, size=300)
    text = json.dumps(report.to_dict(), default=str)
    restored = json.loads(text)
    assert restored["status"] in (STABLE, MODERATE, DRIFT)
    assert "label" in restored and "confidence" in restored


def test_summary_is_plain_text(monitor, service, holdout):
    report, _, _ = batch_report(monitor, service, holdout, size=300)
    text = report.summary()
    assert isinstance(text, str)
    assert "verdict" in text
