"""Tests for performance monitoring (Part B).

The central concerns, in order: labels can arrive late and partially,
predicted labels are never substituted for actual ones, no record is ever
duplicated across repeated registrations or label submissions, and metrics
match hand-checkable values.
"""

import numpy as np
import pandas as pd
import pytest

from monitor import config
from monitor.data import load_adult_data
from monitor.performance_monitoring import (
    COMPUTED,
    INSUFFICIENT,
    NO_LABELS,
    PerformanceMonitor,
    compare_to_baseline,
    load_baseline_test_metrics,
    make_record_id,
)
from monitor.prediction import PredictionService
from monitor.simulation import BatchSimulator, shift_numeric, split_reference_and_holdout


@pytest.fixture(scope="module")
def service():
    return PredictionService()


@pytest.fixture(scope="module")
def holdout():
    _, held = split_reference_and_holdout(load_adult_data())
    return held


def make_batch_and_prediction(service, holdout, size=350, seed=42, modifier=None):
    simulator = BatchSimulator(holdout, batch_size=size, seed=seed)
    batch = simulator.next_batch(modifier=modifier)
    return batch, service.predict_simulated(batch)


# --- stable identifiers ------------------------------------------------------


def test_record_id_combines_batch_and_row_index():
    assert make_record_id("batch_abc", 42) == "batch_abc:42"


def test_record_id_is_stable_across_repeated_calls():
    assert make_record_id("b1", 5) == make_record_id("b1", 5)


def test_different_batches_never_collide_on_record_id():
    assert make_record_id("batch_1", 5) != make_record_id("batch_2", 5)


# --- registering predictions (before any label exists) -----------------------


def test_register_batch_creates_one_record_per_row(service, holdout):
    _, prediction = make_batch_and_prediction(service, holdout, size=100)
    monitor = PerformanceMonitor()

    created = monitor.register_batch(prediction)

    assert created == 100
    frame = monitor.records_frame(prediction.batch_id)
    assert len(frame) == 100
    assert frame["actual_label"].isna().all()  # no labels yet


def test_registered_records_store_the_prediction_not_a_label(service, holdout):
    _, prediction = make_batch_and_prediction(service, holdout, size=20)
    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)

    frame = monitor.records_frame(prediction.batch_id)
    assert set(frame["predicted_label"]) <= set(config.TARGET_LABELS)
    assert frame["actual_label"].isna().all()


def test_reregistering_the_same_batch_creates_no_duplicates(service, holdout):
    _, prediction = make_batch_and_prediction(service, holdout, size=50)
    monitor = PerformanceMonitor()

    first = monitor.register_batch(prediction)
    second = monitor.register_batch(prediction)

    assert first == 50
    assert second == 0
    assert len(monitor.records_frame(prediction.batch_id)) == 50


# --- delayed and partial labels ----------------------------------------------


def test_labels_can_be_submitted_after_registration(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=50)
    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)

    before = monitor.label_coverage(batch.batch_id)
    monitor.submit_labels(batch.batch_id, batch.labels)
    after = monitor.label_coverage(batch.batch_id)

    assert before["n_labeled"] == 0
    assert after["n_labeled"] == 50


def test_partial_labels_are_reflected_in_coverage(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=100)
    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)

    half = batch.labels.index[:40]
    monitor.submit_labels(batch.batch_id, batch.labels.loc[half])
    coverage = monitor.label_coverage(batch.batch_id)

    assert coverage == {
        "n_total": 100, "n_labeled": 40, "n_unlabeled": 60, "labeled_percent": 40.0,
    }


def test_labels_arriving_in_two_waves_do_not_duplicate_records(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=100)
    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)

    first_wave, second_wave = batch.labels.index[:60], batch.labels.index[60:]
    monitor.submit_labels(batch.batch_id, batch.labels.loc[first_wave])
    monitor.submit_labels(batch.batch_id, batch.labels.loc[second_wave])

    frame = monitor.records_frame(batch.batch_id)
    assert len(frame) == 100  # still exactly one row per original record
    assert frame["actual_label"].notna().all()
    assert frame["record_id"].nunique() == 100


def test_unmatched_labels_are_reported_not_silently_dropped(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=30)
    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)

    stray = pd.Series(["<=50K"], index=[999_999_999])  # an index never registered
    result = monitor.submit_labels(batch.batch_id, stray)

    assert result.n_matched == 0
    assert result.n_unmatched == 1
    assert result.unmatched_record_ids == [make_record_id(batch.batch_id, 999_999_999)]


def test_resubmitting_a_label_updates_in_place_and_is_counted(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=20)
    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)

    idx = batch.labels.index[:5]
    first = monitor.submit_labels(batch.batch_id, batch.labels.loc[idx])
    second = monitor.submit_labels(batch.batch_id, batch.labels.loc[idx])

    assert first.n_relabeled == 0
    assert second.n_relabeled == 5
    assert len(monitor.records_frame(batch.batch_id)) == 20  # no duplicates


# --- never use predicted labels as actual ------------------------------------


def test_unlabeled_record_actual_is_none_not_the_prediction(service, holdout):
    _, prediction = make_batch_and_prediction(service, holdout, size=10)
    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)

    frame = monitor.records_frame(prediction.batch_id)
    assert frame["actual_label"].isna().all()
    # Specifically: not silently equal to the prediction as a fallback value.
    assert not (frame["actual_label"] == frame["predicted_label"]).any()


def test_zero_labels_never_fabricates_metrics(service, holdout):
    _, prediction = make_batch_and_prediction(service, holdout, size=40)
    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)

    report = monitor.compute_performance(prediction.batch_id)

    assert report.status == NO_LABELS
    assert report.metrics is None


def test_metrics_only_ever_use_the_actual_label_column(service, holdout):
    # If predictions were a perfect model of themselves, accuracy on the
    # *predicted* labels would trivially be 1.0. Confirm real labels are used
    # by checking the computed accuracy differs from that degenerate value
    # whenever the model is imperfect (which it is).
    batch, prediction = make_batch_and_prediction(service, holdout, size=400)
    monitor = PerformanceMonitor(min_labeled_sample=300)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)

    report = monitor.compute_performance(batch.batch_id)
    assert 0.0 < report.metrics["accuracy"] < 1.0


# --- insufficient sample sizes ------------------------------------------------


def test_small_labeled_sample_is_flagged_but_still_computed(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=100)
    monitor = PerformanceMonitor(min_labeled_sample=300)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)

    report = monitor.compute_performance(batch.batch_id)

    assert report.status == INSUFFICIENT
    assert report.metrics is not None  # computed, not hidden
    assert any("SMALL LABELLED SAMPLE" in note for note in report.notes)


def test_sample_at_or_above_the_floor_is_fully_computed(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=350)
    monitor = PerformanceMonitor(min_labeled_sample=300)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)

    report = monitor.compute_performance(batch.batch_id)
    assert report.status == COMPUTED


def test_min_labeled_sample_is_configurable():
    assert PerformanceMonitor(min_labeled_sample=50).min_labeled_sample == 50
    with pytest.raises(ValueError, match="at least 1"):
        PerformanceMonitor(min_labeled_sample=0)


def test_roc_auc_is_none_when_labeled_subset_has_one_class(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=350)
    monitor = PerformanceMonitor(min_labeled_sample=1)
    monitor.register_batch(prediction)

    one_class_idx = batch.labels[batch.labels == "<=50K"].index[:10]
    monitor.submit_labels(batch.batch_id, batch.labels.loc[one_class_idx])

    report = monitor.compute_performance(batch.batch_id)
    assert report.metrics["roc_auc"] is None


# --- metrics match known expected values -------------------------------------


def test_metrics_match_a_hand_checkable_confusion_matrix(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=350)
    monitor = PerformanceMonitor(min_labeled_sample=1)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)

    report = monitor.compute_performance(batch.batch_id)
    frame = monitor.records_frame(batch.batch_id)

    tp = int(((frame.predicted_label == ">50K") & (frame.actual_label == ">50K")).sum())
    fp = int(((frame.predicted_label == ">50K") & (frame.actual_label == "<=50K")).sum())
    fn = int(((frame.predicted_label == "<=50K") & (frame.actual_label == ">50K")).sum())
    tn = int(((frame.predicted_label == "<=50K") & (frame.actual_label == "<=50K")).sum())

    assert report.metrics["accuracy"] == pytest.approx((tp + tn) / (tp + fp + fn + tn))
    assert report.metrics["precision"] == pytest.approx(tp / (tp + fp) if (tp + fp) else 0.0)
    assert report.metrics["recall"] == pytest.approx(tp / (tp + fn) if (tp + fn) else 0.0)


def test_compute_performance_matches_direct_evaluate_predictions(service, holdout):
    from monitor.evaluation import evaluate_predictions

    batch, prediction = make_batch_and_prediction(service, holdout, size=350)
    monitor = PerformanceMonitor(min_labeled_sample=1)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)

    report = monitor.compute_performance(batch.batch_id)
    frame = monitor.records_frame(batch.batch_id)
    direct = evaluate_predictions(
        frame["actual_label"], frame["predicted_label"], frame["probability"]
    )

    assert report.metrics["accuracy"] == direct["accuracy"]
    assert report.metrics["roc_auc"] == direct["roc_auc"]


# --- comparison against the Phase 3 baseline ---------------------------------


def test_load_baseline_test_metrics_reads_the_real_file():
    baseline = load_baseline_test_metrics()
    assert "accuracy" in baseline
    assert 0 < baseline["accuracy"] < 1


def test_load_baseline_rejects_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_baseline.py"):
        load_baseline_test_metrics(tmp_path / "missing.json")


def test_compare_to_baseline_flags_a_real_drop():
    baseline = {"accuracy": 0.85, "precision": 0.70, "recall": 0.80, "f1_score": 0.75, "roc_auc": 0.90}
    current = {"accuracy": 0.70, "precision": 0.70, "recall": 0.80, "f1_score": 0.75, "roc_auc": 0.90}

    comparison = compare_to_baseline(current, baseline, margin=0.05)

    assert comparison["accuracy"]["degraded"] is True
    assert comparison["accuracy"]["delta"] == pytest.approx(-0.15)
    assert comparison["precision"]["degraded"] is False


def test_compare_to_baseline_ignores_noise_within_the_margin():
    baseline = {"accuracy": 0.85, "precision": 0.70, "recall": 0.80, "f1_score": 0.75, "roc_auc": 0.90}
    current = {"accuracy": 0.83, "precision": 0.70, "recall": 0.80, "f1_score": 0.75, "roc_auc": 0.90}

    comparison = compare_to_baseline(current, baseline, margin=0.05)
    assert comparison["accuracy"]["degraded"] is False


def test_compare_to_baseline_handles_an_improvement():
    baseline = {"accuracy": 0.80}
    current = {"accuracy": 0.90}
    comparison = compare_to_baseline(current, baseline, margin=0.05)
    assert comparison["accuracy"]["delta"] == pytest.approx(0.10)
    assert comparison["accuracy"]["degraded"] is False


def test_compare_to_baseline_skips_metrics_missing_from_either_side():
    comparison = compare_to_baseline({"accuracy": 0.8}, {"accuracy": 0.8, "recall": 0.7})
    assert "recall" not in comparison  # current never reported recall
    assert "accuracy" in comparison


def test_feature_shift_produces_real_measured_degradation(service, holdout):
    # Features drift while true labels are untouched: the model's accuracy on
    # these now-unfamiliar inputs really does fall, independent of any
    # threshold - this is not a hypothetical, it is measured from the model's
    # actual predictions against real labels.
    simulator = BatchSimulator(holdout, batch_size=600, seed=42)
    batch = simulator.next_batch(modifier=shift_numeric("age", 25))
    prediction = service.predict_simulated(batch)

    monitor = PerformanceMonitor(min_labeled_sample=300)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)
    report = monitor.compute_performance(
        batch.batch_id, baseline=load_baseline_test_metrics()
    )

    assert report.comparison["accuracy"]["degraded"] is True
    assert report.comparison["accuracy"]["delta"] < 0


def test_report_notes_the_threshold_comparison_is_not_a_statistical_test(service, holdout):
    simulator = BatchSimulator(holdout, batch_size=600, seed=42)
    batch = simulator.next_batch(modifier=shift_numeric("age", 25))
    prediction = service.predict_simulated(batch)

    monitor = PerformanceMonitor(min_labeled_sample=300)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)
    report = monitor.compute_performance(
        batch.batch_id, baseline=load_baseline_test_metrics()
    )

    assert any("not a statistical test" in note for note in report.notes)


# --- aggregating across batches ----------------------------------------------


def test_records_frame_with_no_batch_id_covers_every_batch(service, holdout):
    simulator = BatchSimulator(holdout, batch_size=100, seed=1)
    monitor = PerformanceMonitor()
    for batch in simulator.generate_batches(3):
        prediction = service.predict_simulated(batch)
        monitor.register_batch(prediction)
        monitor.submit_labels(batch.batch_id, batch.labels)

    all_records = monitor.records_frame()
    assert len(all_records) == 300
    assert all_records["batch_id"].nunique() == 3


def test_compute_performance_without_batch_id_pools_all_batches(service, holdout):
    simulator = BatchSimulator(holdout, batch_size=200, seed=1)
    monitor = PerformanceMonitor(min_labeled_sample=1)
    for batch in simulator.generate_batches(2):
        prediction = service.predict_simulated(batch)
        monitor.register_batch(prediction)
        monitor.submit_labels(batch.batch_id, batch.labels)

    report = monitor.compute_performance(batch_id=None)
    assert report.n_total == 400
    assert report.status == COMPUTED


# --- error handling -----------------------------------------------------------


def test_empty_registered_set_reports_zero_rows():
    monitor = PerformanceMonitor()
    report = monitor.compute_performance("nonexistent-batch")
    assert report.n_total == 0
    assert report.status == NO_LABELS


def test_records_frame_for_an_unknown_batch_is_empty():
    monitor = PerformanceMonitor()
    frame = monitor.records_frame("nonexistent-batch")
    assert len(frame) == 0
    assert list(frame.columns)  # still has the expected columns, just no rows


# --- shape / reuse outside a UI ------------------------------------------------


def test_report_to_dict_is_json_serialisable(service, holdout):
    import json

    batch, prediction = make_batch_and_prediction(service, holdout, size=350)
    monitor = PerformanceMonitor(min_labeled_sample=300)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)
    report = monitor.compute_performance(batch.batch_id, baseline=load_baseline_test_metrics())

    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["status"] == COMPUTED
    assert "comparison" in restored


def test_summary_is_plain_text(service, holdout):
    batch, prediction = make_batch_and_prediction(service, holdout, size=350)
    monitor = PerformanceMonitor(min_labeled_sample=300)
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)
    text = monitor.compute_performance(batch.batch_id).summary()
    assert isinstance(text, str)
    assert "status" in text.lower()
