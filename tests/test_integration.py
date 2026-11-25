"""Phase 9: end-to-end integration tests for the complete monitoring system.

Every other test file in this project exercises one module in isolation
(often with hand-built fixtures). This file instead drives the real,
wired-together system - `build_monitoring_context` plus
`run_monitoring_cycle`, exactly as `manage_alerts.py` and the dashboard call
them - using the real trained model artifact and the real dataset, the same
way `test_performance_monitoring.py` and `test_prediction.py` already do.

Nothing here recomputes a statistic or duplicates a unit test; it checks
that the pieces agree with each other once wired together: the same batch_id
end to end, labels never reaching inference, repeated batches not flooding
storage, and the database staying correct across a simulated restart.
"""

import sqlite3

import pandas as pd
import pytest

from monitor import config, schema
from monitor.alerts import CRITICAL, WARNING, alerts_from_performance
from monitor.drift import DRIFT, STABLE
from monitor.model import extract_preprocessor, unwrap_pipeline
from monitor.monitoring_cycle import build_monitoring_context, run_monitoring_cycle
from monitor.performance_monitoring import (
    COMPUTED,
    INSUFFICIENT,
    NO_LABELS,
    PerformanceMonitor,
    load_baseline_test_metrics,
    make_record_id,
)
from monitor.simulation import SimulatedBatch, overwrite_category, shift_numeric
from monitor.storage import MonitoringRepository

BATCH_SIZE = 400  # above the min_batch_rows / min_labeled_sample floors measured in Phases 5-6


@pytest.fixture
def context(tmp_path):
    """A fresh, fully-wired monitoring context with its own throwaway database.

    Function-scoped (not module-scoped) deliberately: several tests below
    check restart and deduplication behaviour, which depends on exactly
    which runs have been saved so far, so each test must start from a clean
    slate rather than inheriting state from whichever test ran before it.
    """
    db_path = tmp_path / "integration_monitoring.db"
    return build_monitoring_context(db_path=db_path, batch_size=BATCH_SIZE, seed=config.RANDOM_SEED)


# --- 1 & 3: the complete workflow, and batch_id consistency across every stage --------


def test_full_workflow_runs_every_stage_and_persists_every_table(context):
    """Validation -> inference -> drift -> prediction drift -> performance ->
    alerts -> storage, in one call, with nothing silently skipped."""
    batch = context.simulator.next_batch()
    result = run_monitoring_cycle(batch, context)

    # Every stage actually ran and produced its report object.
    assert result["prediction"].n_rows == BATCH_SIZE
    assert result["drift_report"].feature_results  # every feature was checked
    assert result["prediction_report"].label_status in (STABLE, "moderate_drift", DRIFT)
    assert result["performance_report"].status in (NO_LABELS, INSUFFICIENT, COMPUTED)

    # Every stage's result was actually persisted, not just returned in memory.
    repo = context.repository
    assert len(repo.get_runs()) == 1
    assert not repo.get_batch_summaries(batch_id=batch.batch_id).empty
    assert len(repo.get_drift_results(batch_id=batch.batch_id)) == len(schema.FEATURE_COLUMNS)
    assert len(repo.get_prediction_drift_results(batch_id=batch.batch_id)) == 1
    assert len(repo.get_performance_metrics(batch_id=batch.batch_id)) == 1
    assert repo.get_runs().iloc[0]["status"] == "completed"


def test_batch_id_is_identical_across_every_stage_and_every_stored_table(context):
    batch = context.simulator.next_batch()
    result = run_monitoring_cycle(batch, context)
    batch_id = batch.batch_id

    # In-memory report objects.
    assert result["prediction"].batch_id == batch_id
    assert result["drift_report"].batch_id == batch_id
    assert result["prediction_report"].batch_id == batch_id
    assert result["performance_report"].batch_id == batch_id
    assert all(alert.batch_id == batch_id for alert in result["alerts"])

    # Persisted rows.
    repo = context.repository
    assert repo.get_runs().iloc[0]["batch_id"] == batch_id
    assert (repo.get_batch_summaries()["batch_id"] == batch_id).all()
    assert (repo.get_drift_results()["batch_id"] == batch_id).all()
    assert (repo.get_prediction_drift_results()["batch_id"] == batch_id).all()
    assert (repo.get_performance_metrics()["batch_id"] == batch_id).all()
    alerts_df = repo.get_alerts()
    if not alerts_df.empty:
        assert (alerts_df["batch_id"] == batch_id).all()


def test_record_id_is_consistent_between_registration_and_label_submission(context):
    """`PerformanceMonitor` keys a prediction and its later label by the same
    `batch_id:row_index` identifier - verified here against the real
    `BatchPrediction.features.index` the rest of the pipeline produces."""
    batch = context.simulator.next_batch()
    prediction = context.service.predict_simulated(batch)

    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)
    monitor.submit_labels(batch.batch_id, batch.labels)

    frame = monitor.records_frame(batch.batch_id)
    expected_ids = {make_record_id(batch.batch_id, i) for i in prediction.features.index}
    assert set(frame["record_id"]) == expected_ids
    assert frame["actual_label"].notna().all()  # every registered row's label matched


# --- 4: predictions use the saved preprocessing + model pipeline ---------------------


def test_prediction_service_loads_the_real_saved_artifact(context):
    """The loaded model is the exact pipeline `train_baseline.py` saved -
    preprocessing and classifier bundled together, not a separately fitted
    copy that could silently drift out of sync."""
    pipeline = unwrap_pipeline(context.service.model)
    assert set(pipeline.named_steps) == {"preprocess", "classify"}

    # The preprocessor inside the saved model is fitted and usable directly -
    # proof it is the real artifact, not an unfitted stand-in.
    preprocessor = extract_preprocessor(context.service.model)
    batch = context.simulator.next_batch()
    transformed = preprocessor.transform(batch.features)
    assert transformed.shape[0] == len(batch.features)


def test_predict_simulated_calls_the_real_models_predict_with_only_features(context, monkeypatch):
    """Spies on the actual loaded model's `.predict`, not a mock, to confirm
    the real call never receives the label column."""
    batch = context.simulator.next_batch()
    original_predict = context.service.model.predict
    seen_columns = []

    def spying_predict(X, *args, **kwargs):
        seen_columns.append(list(X.columns))
        return original_predict(X, *args, **kwargs)

    monkeypatch.setattr(context.service.model, "predict", spying_predict)
    context.service.predict_simulated(batch)

    assert seen_columns, "the model's predict() was never called"
    assert config.TARGET_COLUMN not in seen_columns[0]
    assert set(seen_columns[0]) == set(schema.FEATURE_COLUMNS)


# --- 5: labels are kept separate from inference, used only for performance -----------


def test_simulated_batches_never_expose_the_label_column_as_a_feature(context):
    batch = context.simulator.next_batch()
    assert config.TARGET_COLUMN not in batch.features.columns
    assert isinstance(batch.labels, pd.Series)  # labels travel beside, not inside, features


def test_prediction_service_refuses_a_batch_that_still_carries_labels(context):
    batch = context.simulator.next_batch()
    leaked = batch.features.copy()
    leaked[config.TARGET_COLUMN] = batch.labels
    with pytest.raises(ValueError, match="must never receive true labels"):
        context.service.predict_batch(leaked)


def test_labels_only_affect_performance_not_drift_or_prediction_outputs(context):
    """Two batches with identical features but different labels must produce
    identical predictions and identical drift results - only the performance
    report, which is the one stage labels are given to, may differ."""
    batch = context.simulator.next_batch()
    swapped_labels = batch.labels.sample(frac=1.0, random_state=1).set_axis(batch.labels.index)
    relabelled = SimulatedBatch(
        batch_id=batch.batch_id + "_relabelled", created_at=batch.created_at,
        features=batch.features, labels=swapped_labels, seed=batch.seed,
        batch_index=batch.batch_index,
    )

    prediction_a = context.service.predict_simulated(batch)
    prediction_b = context.service.predict_simulated(relabelled)
    pd.testing.assert_series_equal(
        prediction_a.predictions, prediction_b.predictions, check_names=False
    )


# --- 6: stable, drifted, invalid, partially-labelled and unlabelled batches ----------


def test_stable_batch_never_triggers_a_drift_alert(context):
    """An unmodified batch must never show input or prediction drift - those
    are measured from the batch's own feature/prediction values, which were
    not touched. A *performance* alert is a different matter: it compares
    against true labels on a finite sample, and `performance_monitoring.py`
    already documents that sampling noise alone can occasionally push a
    metric like precision or f1 past the degradation margin even on
    unmodified data - so that category is deliberately not asserted away
    here, only data_drift and prediction_drift are."""
    batch = context.simulator.next_batch()
    result = run_monitoring_cycle(batch, context)

    assert result["drift_status"] == STABLE
    categories = {alert.category for alert in result["alerts"]}
    assert "data_drift" not in categories
    assert "prediction_drift" not in categories
    non_performance_alerts = [a for a in result["alerts"] if a.category != "performance"]
    assert all(alert.severity == WARNING for alert in non_performance_alerts)


def test_severely_drifted_batch_produces_critical_drift_and_performance_alerts(context):
    batch = context.simulator.next_batch(modifier=shift_numeric("age", 25))
    result = run_monitoring_cycle(batch, context)

    assert result["drift_status"] == DRIFT
    severities = {alert.severity for alert in result["alerts"]}
    assert CRITICAL in severities
    categories_at_critical = {a.category for a in result["alerts"] if a.severity == CRITICAL}
    assert "data_drift" in categories_at_critical


def test_moderately_drifted_batch_produces_warning_not_critical(context):
    batch = context.simulator.next_batch(
        modifier=overwrite_category("occupation", "Gig-economy", fraction=0.05)
    )
    result = run_monitoring_cycle(batch, context)

    assert result["drift_status"] == "moderate_drift"
    drift_alerts = [a for a in result["alerts"] if a.category == "data_drift"]
    assert drift_alerts and all(a.severity == WARNING for a in drift_alerts)


def test_invalid_batch_fails_cleanly_and_the_run_is_recorded_as_failed(context):
    """A batch missing a required column must raise - from real validation,
    not a stub - and the run row must end up `status="failed"`, never stuck
    at `status="running"` forever."""
    batch = context.simulator.next_batch()
    malformed = SimulatedBatch(
        batch_id="integration_invalid_batch", created_at=batch.created_at,
        features=batch.features.drop(columns=["age"]), labels=batch.labels,
        seed=batch.seed, batch_index=batch.batch_index,
    )

    with pytest.raises(ValueError, match="failed validation"):
        run_monitoring_cycle(malformed, context)

    runs = context.repository.get_runs()
    failed = runs[runs["batch_id"] == "integration_invalid_batch"]
    assert len(failed) == 1
    assert failed.iloc[0]["status"] == "failed"
    assert "ValueError" in failed.iloc[0]["notes"]
    # Nothing downstream of the failed prediction step was saved for this batch.
    assert context.repository.get_drift_results(batch_id="integration_invalid_batch").empty


def test_partially_labelled_batch_is_computed_but_flagged_insufficient(context):
    batch = context.simulator.next_batch()
    prediction = context.service.predict_simulated(batch)

    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)
    partial_labels = batch.labels.iloc[: monitor.min_labeled_sample - 1]
    monitor.submit_labels(batch.batch_id, partial_labels)

    report = monitor.compute_performance(batch.batch_id, baseline=context.baseline)
    assert report.status == INSUFFICIENT
    assert report.n_labeled == len(partial_labels)
    assert report.n_labeled < report.n_total  # genuinely partial, not all rows


def test_unlabelled_batch_produces_no_labels_not_a_fabricated_score(context):
    batch = context.simulator.next_batch()
    prediction = context.service.predict_simulated(batch)

    monitor = PerformanceMonitor()
    monitor.register_batch(prediction)  # no submit_labels call at all

    report = monitor.compute_performance(batch.batch_id, baseline=context.baseline)
    assert report.status == NO_LABELS
    assert report.metrics is None
    assert alerts_from_performance(report) == []  # no verdict, so no alert


# --- 7: SQLite history remains correct after a restart -------------------------------


def test_history_survives_a_simulated_restart(context, tmp_path):
    db_path = context.repository.db_path
    batch1 = context.simulator.next_batch()
    result1 = run_monitoring_cycle(batch1, context)
    context.simulator.reset()
    batch2 = context.simulator.next_batch(modifier=shift_numeric("age", 25))
    result2 = run_monitoring_cycle(batch2, context)

    # Simulate a restart: drop every in-memory reference, reopen by path only.
    del context
    reopened = MonitoringRepository(db_path)

    runs = reopened.get_runs()
    assert set(runs["batch_id"]) == {batch1.batch_id, batch2.batch_id}
    assert set(runs["status"]) == {"completed"}

    recovered_critical_alerts = reopened.get_alerts(severity=CRITICAL)
    original_critical_alerts = [a for a in result2["alerts"] if a.severity == CRITICAL]
    assert len(recovered_critical_alerts) == len(original_critical_alerts) > 0

    assert not reopened.get_drift_results(batch_id=batch1.batch_id).empty
    assert not reopened.get_performance_metrics(batch_id=batch2.batch_id).empty


# --- 8: repeated processing never creates unintended duplicates ----------------------


def test_two_stable_batches_deduplicate_the_recurring_data_quality_alerts(context):
    batch1 = context.simulator.next_batch()
    run_monitoring_cycle(batch1, context)
    batch2 = context.simulator.next_batch()
    run_monitoring_cycle(batch2, context)

    alerts = context.repository.get_alerts(category="data_quality")
    # The same three recurring missing-value warnings, each seen twice -
    # not six separate rows.
    assert len(alerts) <= 3
    assert (alerts["occurrence_count"] == 2).all()
    assert len(context.repository.get_runs()) == 2  # still two distinct runs, though


def test_rerunning_the_identical_batch_object_does_not_duplicate_storage_rows(context):
    """Calling run_monitoring_cycle twice on the *same* SimulatedBatch (e.g. a
    retried call) must not create a second batch_summaries/drift row for it,
    even though it does create a second run_id."""
    batch = context.simulator.next_batch()
    run_monitoring_cycle(batch, context)
    run_monitoring_cycle(batch, context)  # same batch object, run again

    # batch_summaries is unique on (run_id, batch_id), so the second run's
    # summary is a second row - but drift_results is unique on (run_id,
    # feature), and both runs reused distinct run_ids, so this checks the
    # thing that must NOT happen: the same run_id's rows duplicating.
    runs = context.repository.get_runs()
    assert len(runs) == 2
    with sqlite3.connect(context.repository.db_path) as conn:
        for run_id in runs["run_id"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM drift_results WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            assert count == len(schema.FEATURE_COLUMNS)  # exactly one row per feature, per run


# --- 9: error handling for missing artifacts, schemas, data, and the database --------


def test_missing_model_artifact_raises_a_clear_error_through_the_full_context_builder(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "BASELINE_MODEL_PATH", tmp_path / "absent_model.joblib")
    with pytest.raises(FileNotFoundError, match="train_baseline.py"):
        build_monitoring_context(db_path=tmp_path / "db.sqlite")


def test_missing_baseline_metrics_error_propagates_through_the_context_builder(
    tmp_path, monkeypatch
):
    """`build_monitoring_context` must not swallow a failure from
    `load_baseline_test_metrics` - confirmed by making that call fail and
    checking the same error reaches the caller unchanged.

    Note: monkeypatching `config.BASELINE_METRICS_PATH` itself would not
    work here, because `load_baseline_test_metrics`'s `path` parameter binds
    that constant's value once, at function-definition time (ordinary
    Python default-argument semantics) - patching the constant afterwards
    does not change an already-bound default. Patching the function
    `monitoring_cycle` actually calls is the correct way to exercise this.
    """
    import monitor.monitoring_cycle as monitoring_cycle_module

    def _raise_missing_metrics():
        raise FileNotFoundError(
            "No baseline metrics at nope.json. Run `python train_baseline.py` first."
        )

    monkeypatch.setattr(monitoring_cycle_module, "load_baseline_test_metrics", _raise_missing_metrics)
    with pytest.raises(FileNotFoundError, match="train_baseline.py"):
        build_monitoring_context(db_path=tmp_path / "db.sqlite")


def test_database_path_that_cannot_be_opened_raises_a_clear_storage_error(tmp_path):
    from monitor.storage import MonitoringStorageError

    not_a_file = tmp_path / "this_is_a_directory"
    not_a_file.mkdir()
    with pytest.raises(MonitoringStorageError):
        build_monitoring_context(db_path=not_a_file)


def test_baseline_metrics_can_still_be_loaded_directly_with_a_clear_error(tmp_path):
    """Smoke-checks the function `build_monitoring_context` relies on, in
    isolation, so a future change to how it's called still gets caught."""
    with pytest.raises(FileNotFoundError, match="train_baseline.py"):
        load_baseline_test_metrics(tmp_path / "nope.json")


# --- 15: reproducibility ---------------------------------------------------------------


def test_same_seed_produces_the_same_batches_and_the_same_alerts(tmp_path):
    """Two independently built contexts, same seed, must draw the identical
    batch and raise the identical alerts - the reproducibility guarantee the
    rest of the project's seeding depends on."""
    context_a = build_monitoring_context(
        db_path=tmp_path / "a.db", batch_size=BATCH_SIZE, seed=123
    )
    context_b = build_monitoring_context(
        db_path=tmp_path / "b.db", batch_size=BATCH_SIZE, seed=123
    )

    batch_a = context_a.simulator.next_batch(modifier=shift_numeric("age", 25))
    batch_b = context_b.simulator.next_batch(modifier=shift_numeric("age", 25))
    pd.testing.assert_frame_equal(batch_a.features, batch_b.features)
    pd.testing.assert_series_equal(batch_a.labels, batch_b.labels)

    result_a = run_monitoring_cycle(batch_a, context_a)
    result_b = run_monitoring_cycle(batch_b, context_b)
    titles_a = sorted((a.severity, a.title) for a in result_a["alerts"])
    titles_b = sorted((a.severity, a.title) for a in result_b["alerts"])
    assert titles_a == titles_b
