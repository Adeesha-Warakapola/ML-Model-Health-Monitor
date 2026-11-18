"""Tests for the shared monitoring-cycle orchestration used by both
`manage_alerts.py` and the dashboard.

Uses the real trained artifact (as `test_performance_monitoring.py` and
`test_prediction.py` already do) rather than a stub model, because the point
of this module is wiring already-tested services together correctly - a
fake model would not exercise that wiring.
"""

import pytest

from monitor.monitoring_cycle import build_monitoring_context, run_monitoring_cycle
from monitor.simulation import shift_numeric


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("monitoring_cycle") / "test_monitoring.db"
    return build_monitoring_context(db_path=db_path, batch_size=350, seed=42)


def test_build_monitoring_context_wires_every_service(context):
    assert context.service is not None
    assert context.drift_detector is not None
    assert context.prediction_monitor is not None
    assert context.baseline
    assert context.repository is not None
    assert context.simulator.rows_remaining > 0


def test_run_monitoring_cycle_returns_every_expected_key(context):
    batch = context.simulator.next_batch()
    result = run_monitoring_cycle(batch, context)

    for key in (
        "run_id", "batch_id", "alerts", "outcomes", "drift_status",
        "prediction_status", "performance_status", "prediction",
        "drift_report", "prediction_report", "performance_report", "batch",
    ):
        assert key in result

    assert result["batch_id"] == batch.batch_id
    assert result["drift_report"].batch_id == batch.batch_id


def test_run_monitoring_cycle_persists_to_the_repository(context):
    batch = context.simulator.next_batch()
    result = run_monitoring_cycle(batch, context)

    runs = context.repository.get_runs()
    assert result["run_id"] in runs["run_id"].tolist()

    drift_rows = context.repository.get_drift_results(batch_id=batch.batch_id)
    assert not drift_rows.empty

    performance_rows = context.repository.get_performance_metrics(batch_id=batch.batch_id)
    assert len(performance_rows) == 1

    prediction_drift_rows = context.repository.get_prediction_drift_results(batch_id=batch.batch_id)
    assert len(prediction_drift_rows) == 1


def test_a_severe_shift_produces_a_critical_drift_alert(context):
    context.simulator.reset()
    batch = context.simulator.next_batch(modifier=shift_numeric("age", 25))
    result = run_monitoring_cycle(batch, context)

    assert result["drift_status"] == "drift"
    assert any(alert.severity == "CRITICAL" for alert in result["alerts"])
