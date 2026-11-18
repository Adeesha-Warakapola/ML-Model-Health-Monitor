"""Tests for the SQLite monitoring repository.

Every test uses a throwaway database file under pytest's `tmp_path`, so
nothing here touches the real monitoring/ directory, and each test starts
from a genuinely empty database - including the ones that test what happens
when the file does not exist yet.
"""

import sqlite3

import pandas as pd
import pytest

from monitor.alerts import CRITICAL, WARNING, Alert
from monitor.drift import DRIFT, STABLE, DriftReport, DriftThresholds, FeatureDriftResult
from monitor.performance_monitoring import COMPUTED, PerformanceReport
from monitor.prediction import BatchPrediction
from monitor.prediction_monitoring import PredictionDriftReport
from monitor.storage import MonitoringRepository, MonitoringStorageError, make_run_id
from monitor.validation import ValidationResult


@pytest.fixture
def repo(tmp_path):
    return MonitoringRepository(tmp_path / "test_monitoring.db")


def make_prediction(batch_id="batch_1", n=5) -> BatchPrediction:
    index = list(range(n))
    return BatchPrediction(
        batch_id=batch_id, predicted_at="now", model_type="LogisticRegression",
        threshold=0.29, features=pd.DataFrame({"age": [30] * n}, index=index),
        predictions=pd.Series(["<=50K"] * (n - 1) + [">50K"], index=index),
        probabilities=pd.Series([0.1] * (n - 1) + [0.9], index=index),
        validation=ValidationResult(issues=[], n_rows=n),
    )


def make_drift_report(batch_id="batch_1") -> DriftReport:
    results = [
        FeatureDriftResult(feature="age", feature_type="numeric", method="ks+psi",
                            status=DRIFT, psi=2.5, statistic=0.4, p_value=0.001,
                            n_reference=1000, n_batch=200),
        FeatureDriftResult(feature="sex", feature_type="categorical", method="chi2+psi",
                            status=STABLE, psi=0.01, n_reference=1000, n_batch=200),
    ]
    return DriftReport(
        batch_id=batch_id, evaluated_at="now", feature_results=results,
        validation=ValidationResult(issues=[], n_rows=200),
        thresholds=DriftThresholds(), n_reference=1000, n_batch=200,
    )


def make_prediction_drift_report(batch_id="batch_1") -> PredictionDriftReport:
    return PredictionDriftReport(
        batch_id=batch_id, evaluated_at="now", n_reference=1000, n_batch=200,
        threshold=0.29, reference_distribution={}, batch_distribution={},
        label_status=STABLE, label_psi=0.02, label_statistic=None, label_p_value=None,
        label_effect_size=None, reference_confidence={"n": 0}, batch_confidence={"n": 0},
        confidence_status=STABLE, confidence_psi=0.02, confidence_statistic=None,
        confidence_p_value=None, thresholds=DriftThresholds(),
    )


def make_performance_report(batch_id="batch_1") -> PerformanceReport:
    return PerformanceReport(
        status=COMPUTED, evaluated_at="now", batch_id=batch_id, n_total=200,
        n_labeled=200, min_labeled_sample=300,
        metrics={"accuracy": 0.8, "precision": 0.6, "recall": 0.7, "f1_score": 0.65, "roc_auc": 0.9},
        comparison={"accuracy": {"baseline": 0.85, "current": 0.8, "delta": -0.05, "degraded": False}},
    )


# --- initialization and missing database files -------------------------------


def test_creates_the_database_file_if_it_does_not_exist(tmp_path):
    db_path = tmp_path / "subdir" / "new.db"
    assert not db_path.exists()

    MonitoringRepository(db_path)

    assert db_path.exists()


def test_creates_parent_directories_that_do_not_exist(tmp_path):
    db_path = tmp_path / "a" / "b" / "c" / "monitoring.db"
    MonitoringRepository(db_path)
    assert db_path.exists()


def test_reinitializing_an_existing_database_does_not_lose_data(tmp_path):
    db_path = tmp_path / "monitoring.db"
    first = MonitoringRepository(db_path)
    first.save_run(make_run_id(), "batch_1", "now")

    second = MonitoringRepository(db_path)  # same file, fresh instance
    assert len(second.get_runs()) == 1


def test_all_tables_exist_after_initialization(repo):
    with sqlite3.connect(repo.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {
        "monitoring_runs", "batch_summaries", "drift_results",
        "prediction_drift_results", "performance_metrics", "alerts",
    } <= tables


def test_opening_a_pre_existing_database_without_the_newer_columns_adds_them(tmp_path):
    """A database created before notes_json/metrics_json existed must still open.

    Simulates upgrading: a hand-built database using the *original* Phase 7
    schema (no notes_json, no metrics_json, no batch_*_json columns) must be
    usable by the current code without deleting any data, rather than
    failing the first time a dashboard page tries to read a new column.
    """
    db_path = tmp_path / "old_schema.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE monitoring_runs (run_id TEXT PRIMARY KEY, batch_id TEXT, "
            "started_at TEXT, completed_at TEXT, status TEXT, notes TEXT)"
        )
        conn.execute(
            "CREATE TABLE drift_results (id INTEGER PRIMARY KEY, run_id TEXT, "
            "batch_id TEXT, feature TEXT, feature_type TEXT, method TEXT, "
            "status TEXT, psi REAL, statistic REAL, p_value REAL, "
            "effect_size REAL, n_reference INTEGER, n_batch INTEGER, recorded_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE prediction_drift_results (id INTEGER PRIMARY KEY, "
            "run_id TEXT, batch_id TEXT, label_status TEXT, label_psi REAL, "
            "confidence_status TEXT, confidence_psi REAL, recorded_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE performance_metrics (id INTEGER PRIMARY KEY, run_id TEXT, "
            "batch_id TEXT, status TEXT, n_total INTEGER, n_labeled INTEGER, "
            "accuracy REAL, precision_score REAL, recall REAL, f1_score REAL, "
            "roc_auc REAL, comparison_json TEXT, recorded_at TEXT)"
        )

    repo = MonitoringRepository(db_path)  # must not raise

    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_drift_report(run_id, make_drift_report("batch_1"))
    repo.save_prediction_drift_report(run_id, make_prediction_drift_report("batch_1"))
    repo.save_performance_report(run_id, make_performance_report("batch_1"))

    assert len(repo.get_drift_results(batch_id="batch_1")) == 2
    assert len(repo.get_prediction_drift_results(batch_id="batch_1")) == 1
    assert len(repo.get_performance_metrics(batch_id="batch_1")) == 1


# --- runs ----------------------------------------------------------------------


def test_save_and_retrieve_a_run(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "2026-01-01T00:00:00+00:00")

    runs = repo.get_runs()
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == run_id
    assert runs.iloc[0]["batch_id"] == "batch_1"


def test_complete_run_sets_completed_at_and_status(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.complete_run(run_id, status="completed")

    row = repo.get_runs().iloc[0]
    assert row["status"] == "completed"
    assert row["completed_at"] is not None


def test_saving_the_same_run_id_twice_does_not_duplicate(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_run(run_id, "batch_1", "now")  # e.g. a retried call

    assert len(repo.get_runs()) == 1


def test_runs_can_be_filtered_by_date(repo):
    repo.save_run(make_run_id(), "batch_1", "2026-01-01T00:00:00+00:00")
    repo.save_run(make_run_id(), "batch_2", "2026-06-01T00:00:00+00:00")

    recent = repo.get_runs(start_date="2026-03-01T00:00:00+00:00")
    assert len(recent) == 1
    assert recent.iloc[0]["batch_id"] == "batch_2"


# --- drift results ---------------------------------------------------------------


def test_save_and_retrieve_drift_results(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_drift_report(run_id, make_drift_report("batch_1"))

    results = repo.get_drift_results(batch_id="batch_1")
    assert len(results) == 2
    assert set(results["feature"]) == {"age", "sex"}
    age_row = results[results["feature"] == "age"].iloc[0]
    assert age_row["status"] == DRIFT
    assert age_row["psi"] == pytest.approx(2.5)


def test_drift_results_can_be_filtered_by_feature(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_drift_report(run_id, make_drift_report("batch_1"))

    results = repo.get_drift_results(feature="age")
    assert len(results) == 1
    assert results.iloc[0]["feature"] == "age"


def test_resaving_drift_for_the_same_run_replaces_not_duplicates(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_drift_report(run_id, make_drift_report("batch_1"))
    repo.save_drift_report(run_id, make_drift_report("batch_1"))  # saved twice

    assert len(repo.get_drift_results(batch_id="batch_1")) == 2  # still 2, not 4


# --- prediction drift and performance --------------------------------------------


def test_save_and_retrieve_prediction_drift(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_prediction_drift_report(run_id, make_prediction_drift_report("batch_1"))

    with sqlite3.connect(repo.db_path) as conn:
        row = conn.execute(
            "SELECT label_status, confidence_psi FROM prediction_drift_results "
            "WHERE batch_id = ?",
            ("batch_1",),
        ).fetchone()
    assert row[0] == STABLE
    assert row[1] == pytest.approx(0.02)


def test_save_and_retrieve_performance_metrics(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_performance_report(run_id, make_performance_report("batch_1"))

    results = repo.get_performance_metrics(batch_id="batch_1")
    assert len(results) == 1
    assert results.iloc[0]["accuracy"] == pytest.approx(0.8)
    assert results.iloc[0]["status"] == COMPUTED


def test_get_prediction_drift_results_returns_saved_rows(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_prediction_drift_report(run_id, make_prediction_drift_report("batch_1"))

    results = repo.get_prediction_drift_results(batch_id="batch_1")
    assert len(results) == 1
    assert results.iloc[0]["label_status"] == STABLE


def test_get_prediction_drift_results_with_no_matching_batch_is_empty(repo):
    assert repo.get_prediction_drift_results(batch_id="nonexistent").empty


def test_performance_metrics_json_round_trips_the_full_metrics_dict(repo):
    import json

    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_performance_report(run_id, make_performance_report("batch_1"))

    results = repo.get_performance_metrics(batch_id="batch_1")
    saved_metrics = json.loads(results.iloc[0]["metrics_json"])
    assert saved_metrics["accuracy"] == pytest.approx(0.8)
    assert saved_metrics["f1_score"] == pytest.approx(0.65)


def test_drift_results_notes_json_round_trips_feature_notes(repo):
    import json

    report = make_drift_report("batch_1")
    report.feature_results[0].notes.append("SMALL SAMPLE: just a test note")
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_drift_report(run_id, report)

    results = repo.get_drift_results(batch_id="batch_1", feature="age")
    notes = json.loads(results.iloc[0]["notes_json"])
    assert notes == ["SMALL SAMPLE: just a test note"]


def test_batch_summary_is_saved_and_retrievable(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_batch_summary(run_id, make_prediction("batch_1", n=10), actual_positive_rate=0.3, n_labeled=10)

    summaries = repo.get_batch_summaries(batch_id="batch_1")
    assert len(summaries) == 1
    assert summaries.iloc[0]["n_rows"] == 10
    assert summaries.iloc[0]["actual_positive_rate"] == pytest.approx(0.3)


# --- alert deduplication (the flooding-prevention requirement) ------------------


def test_first_occurrence_of_an_alert_is_created(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    alert = Alert(category="data_drift", severity=WARNING, title="Feature drift: age",
                   description="d", reason="psi=0.15", batch_id="batch_1", feature="age")

    outcome = repo.save_alert(alert, run_id)

    assert outcome.action == "created"
    assert outcome.occurrence_count == 1
    assert len(repo.get_alerts()) == 1


def test_repeated_identical_alert_updates_the_same_row_instead_of_adding_one(repo):
    alert = Alert(category="data_drift", severity=WARNING, title="Feature drift: age",
                   description="d", reason="r", batch_id="batch_1", feature="age")

    for batch_id in ("batch_1", "batch_2", "batch_3"):
        run_id = make_run_id()
        repo.save_run(run_id, batch_id, "now")
        repeated = Alert(category="data_drift", severity=WARNING, title="Feature drift: age",
                          description="d", reason="r", batch_id=batch_id, feature="age")
        repo.save_alert(repeated, run_id)

    alerts = repo.get_alerts()
    assert len(alerts) == 1  # not 3 rows
    assert alerts.iloc[0]["occurrence_count"] == 3
    assert alerts.iloc[0]["batch_id"] == "batch_3"  # moved to the most recent batch


def test_deduplication_never_downgrades_severity(repo):
    run_id_1 = make_run_id()
    repo.save_run(run_id_1, "batch_1", "now")
    repo.save_alert(
        Alert(category="data_drift", severity=CRITICAL, title="t", description="d",
              reason="r", batch_id="batch_1", feature="age"),
        run_id_1,
    )

    run_id_2 = make_run_id()
    repo.save_run(run_id_2, "batch_2", "now")
    outcome = repo.save_alert(
        Alert(category="data_drift", severity=WARNING, title="t", description="d",
              reason="r", batch_id="batch_2", feature="age"),
        run_id_2,
    )

    # A milder recurrence must not erase the fact that it was once CRITICAL.
    assert outcome.severity == CRITICAL
    assert repo.get_alerts().iloc[0]["severity"] == CRITICAL


def test_different_features_never_deduplicate_together(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_alert(
        Alert(category="data_drift", severity=WARNING, title="t1", description="d",
              reason="r", batch_id="batch_1", feature="age"),
        run_id,
    )
    repo.save_alert(
        Alert(category="data_drift", severity=WARNING, title="t2", description="d",
              reason="r", batch_id="batch_1", feature="hours_per_week"),
        run_id,
    )

    assert len(repo.get_alerts()) == 2


def test_save_alerts_saves_a_whole_list_and_returns_one_outcome_each(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    alerts = [
        Alert(category="data_drift", severity=WARNING, title="t1", description="d",
              reason="r", batch_id="batch_1", feature="age"),
        Alert(category="performance", severity=CRITICAL, title="t2", description="d",
              reason="r", batch_id="batch_1", metric_name="accuracy"),
    ]
    outcomes = repo.save_alerts(alerts, run_id)

    assert len(outcomes) == 2
    assert all(o.action == "created" for o in outcomes)
    assert len(repo.get_alerts()) == 2


# --- retrieval filters: date, batch ID, category, severity ----------------------


def test_alerts_filtered_by_severity(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_alert(Alert(category="data_drift", severity=WARNING, title="w", description="d",
                           reason="r", batch_id="batch_1", feature="age"), run_id)
    repo.save_alert(Alert(category="performance", severity=CRITICAL, title="c", description="d",
                           reason="r", batch_id="batch_1", metric_name="accuracy"), run_id)

    critical_only = repo.get_alerts(severity=CRITICAL)
    assert len(critical_only) == 1
    assert critical_only.iloc[0]["title"] == "c"


def test_alerts_filtered_by_category(repo):
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_alert(Alert(category="data_drift", severity=WARNING, title="w", description="d",
                           reason="r", batch_id="batch_1", feature="age"), run_id)
    repo.save_alert(Alert(category="performance", severity=CRITICAL, title="c", description="d",
                           reason="r", batch_id="batch_1", metric_name="accuracy"), run_id)

    performance_only = repo.get_alerts(category="performance")
    assert len(performance_only) == 1
    assert performance_only.iloc[0]["category"] == "performance"


def test_alerts_filtered_by_batch_id(repo):
    for batch_id in ("batch_1", "batch_2"):
        run_id = make_run_id()
        repo.save_run(run_id, batch_id, "now")
        repo.save_alert(
            Alert(category="data_drift", severity=WARNING, title=f"t_{batch_id}",
                  description="d", reason="r", batch_id=batch_id, feature=batch_id),
            run_id,
        )

    assert len(repo.get_alerts(batch_id="batch_1")) == 1
    assert repo.get_alerts(batch_id="batch_1").iloc[0]["batch_id"] == "batch_1"


def test_alerts_filtered_by_date_range(repo):
    import sqlite3 as sql

    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_alert(
        Alert(category="data_drift", severity=WARNING, title="t", description="d",
              reason="r", batch_id="batch_1", feature="age"),
        run_id,
    )
    # Backdate it directly, since save_alert always stamps "now".
    with sql.connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE alerts SET first_seen_at = ?, last_seen_at = ?",
            ("2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00"),
        )

    assert len(repo.get_alerts(start_date="2025-01-01T00:00:00+00:00")) == 0
    assert len(repo.get_alerts(end_date="2025-01-01T00:00:00+00:00")) == 1


def test_empty_database_returns_empty_dataframes_not_errors(repo):
    assert len(repo.get_alerts()) == 0
    assert len(repo.get_runs()) == 0
    assert len(repo.get_drift_results()) == 0
    assert len(repo.get_performance_metrics()) == 0


# --- error handling -----------------------------------------------------------


def test_unwritable_database_path_raises_a_clear_storage_error(tmp_path):
    # A path that is a directory, not a file, cannot be opened as a database.
    directory_as_db = tmp_path / "not_a_file"
    directory_as_db.mkdir()

    with pytest.raises(MonitoringStorageError):
        MonitoringRepository(directory_as_db)


def test_saving_an_alert_for_a_nonexistent_run_is_rejected(repo):
    # The foreign key on alerts.run_id must be enforced, not silently ignored.
    alert = Alert(category="data_drift", severity=WARNING, title="t", description="d",
                  reason="r", batch_id="batch_1", feature="age")
    with pytest.raises(MonitoringStorageError):
        repo.save_alert(alert, run_id="run_that_does_not_exist")


# --- independence from any dashboard ---------------------------------------------


def test_storage_module_does_not_import_streamlit():
    import sys

    import monitor.storage  # noqa: F401

    assert "streamlit" not in sys.modules


def test_repository_is_usable_with_only_plain_data_no_live_services(tmp_path):
    # Every save_* call above already uses hand-built dataclasses rather than
    # a live model or dataset - this test just states that guarantee plainly.
    repo = MonitoringRepository(tmp_path / "standalone.db")
    run_id = make_run_id()
    repo.save_run(run_id, "batch_1", "now")
    repo.save_drift_report(run_id, make_drift_report("batch_1"))
    assert len(repo.get_drift_results()) == 2
