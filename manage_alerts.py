"""Entry point: run monitoring cycles, generate alerts, and persist history.

Run with:
    python manage_alerts.py              # three scenarios, then a simulated restart
    python manage_alerts.py --fresh       # delete monitoring/monitoring_history.db first

Ties together every existing service from Phases 2-6 - validation, drift
detection, prediction monitoring, performance monitoring - and the two new
Phase 7 modules, `alerts.py` and `storage.py`. Nothing here recomputes a
statistic; it only calls the existing modules once each and decides what to
alert on and what to save.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from monitor import config
from monitor.alerts import generate_alerts
from monitor.data import load_adult_data
from monitor.drift import DriftDetector
from monitor.performance_monitoring import PerformanceMonitor, load_baseline_test_metrics
from monitor.prediction import PredictionService, utc_now
from monitor.prediction_monitoring import PredictionMonitor
from monitor.simulation import (
    BatchSimulator,
    get_reference_features,
    overwrite_category,
    shift_numeric,
    split_reference_and_holdout,
)
from monitor.storage import MonitoringRepository, make_run_id

SEPARATOR = "=" * 72


def run_monitoring_cycle(
    batch,
    service: PredictionService,
    drift_detector: DriftDetector,
    prediction_monitor: PredictionMonitor,
    baseline: dict,
    repository: MonitoringRepository,
) -> dict:
    """One full cycle: predict, check drift, check performance, alert, persist.

    Every piece of monitoring logic is called exactly once here and nowhere
    recomputed - this function's job is sequencing and persistence, not
    calculation.
    """
    run_id = make_run_id()
    repository.save_run(run_id, batch.batch_id, started_at=utc_now())

    prediction = service.predict_simulated(batch)
    drift_report = drift_detector.detect(batch.features, batch_id=batch.batch_id)
    prediction_report = prediction_monitor.evaluate(prediction)

    performance_monitor = PerformanceMonitor(min_labeled_sample=300)
    performance_monitor.register_batch(prediction)
    performance_monitor.submit_labels(batch.batch_id, batch.labels)
    performance_report = performance_monitor.compute_performance(
        batch.batch_id, baseline=baseline
    )

    alerts = generate_alerts(
        batch_id=batch.batch_id,
        validation=prediction.validation,
        drift=drift_report,
        prediction_drift=prediction_report,
        performance=performance_report,
    )

    actual_positive_rate = (batch.labels == config.POSITIVE_LABEL).mean()
    repository.save_batch_summary(
        run_id, prediction, actual_positive_rate=actual_positive_rate,
        n_labeled=len(batch.labels),
    )
    repository.save_drift_report(run_id, drift_report)
    repository.save_prediction_drift_report(run_id, prediction_report)
    repository.save_performance_report(run_id, performance_report)
    outcomes = repository.save_alerts(alerts, run_id)
    repository.complete_run(run_id)

    return {
        "run_id": run_id, "batch_id": batch.batch_id, "alerts": alerts,
        "outcomes": outcomes, "drift_status": drift_report.status,
        "prediction_status": prediction_report.status,
        "performance_status": performance_report.status,
    }


def describe_cycle(title: str, result: dict) -> None:
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print(f"run_id: {result['run_id']}   batch_id: {result['batch_id']}")
    print(
        f"data drift: {result['drift_status']}   "
        f"prediction drift: {result['prediction_status']}   "
        f"performance: {result['performance_status']}"
    )
    if not result["alerts"]:
        print("No alerts raised.")
    for alert, outcome in zip(result["alerts"], result["outcomes"]):
        print(f"  [{outcome.action:12s}] {alert}  (occurrence #{outcome.occurrence_count})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alerts and persist monitoring history.")
    parser.add_argument(
        "--fresh", action="store_true", help="Delete the existing history database first."
    )
    args = parser.parse_args()

    if args.fresh and config.MONITORING_DB_PATH.exists():
        config.MONITORING_DB_PATH.unlink()
        print(f"Deleted {config.MONITORING_DB_PATH}\n")

    df = load_adult_data()
    reference_features = get_reference_features(df)
    _, holdout = split_reference_and_holdout(df)

    service = PredictionService()
    drift_detector = DriftDetector(reference_features)
    prediction_monitor = PredictionMonitor.from_reference_features(service, reference_features)
    baseline = load_baseline_test_metrics()

    repository = MonitoringRepository(config.MONITORING_DB_PATH)
    print(f"Monitoring database: {config.MONITORING_DB_PATH}\n")

    simulator = BatchSimulator(holdout, batch_size=600, seed=config.RANDOM_SEED)

    # Scenario 1: stable batch. Expect only the dataset's known, recurring
    # missing-value warnings (Phase 1/2 found workclass/occupation/
    # native_country run ~2-6% missing) - nothing data-drift, prediction-
    # drift, or performance related.
    result1 = run_monitoring_cycle(
        simulator.next_batch(), service, drift_detector, prediction_monitor,
        baseline, repository,
    )
    describe_cycle("SCENARIO 1 - normal batch (expect WARNING at most)", result1)

    # Scenario 2: the same kind of batch again. The three recurring
    # data-quality warnings must deduplicate into the same three alert rows
    # with a higher occurrence_count, not create three new ones.
    result2 = run_monitoring_cycle(
        simulator.next_batch(), service, drift_detector, prediction_monitor,
        baseline, repository,
    )
    describe_cycle(
        "SCENARIO 2 - another normal batch (watch occurrence counts go up, not rows)",
        result2,
    )

    # Scenario 3: a batch with an unseen category affecting 5% of rows.
    # Measured directly (see docs/alert_management.md): 5% gives PSI 0.1587,
    # which lands in the moderate band (0.10-0.25) and so should alert at
    # WARNING, not CRITICAL. A larger fraction (8%+) crosses into drift.
    simulator.reset()
    warning_batch = simulator.next_batch(
        modifier=overwrite_category("occupation", "Gig-economy", fraction=0.05)
    )
    result3 = run_monitoring_cycle(
        warning_batch, service, drift_detector, prediction_monitor, baseline, repository,
    )
    describe_cycle("SCENARIO 3 - moderate problem (expect WARNING)", result3)

    # Scenario 4: a severe, deliberate shift. Expect CRITICAL alerts for
    # data drift and for confirmed performance degradation.
    simulator.reset()
    critical_batch = simulator.next_batch(modifier=shift_numeric("age", 25))
    result4 = run_monitoring_cycle(
        critical_batch, service, drift_detector, prediction_monitor, baseline, repository,
    )
    describe_cycle("SCENARIO 4 - severe problem (expect CRITICAL)", result4)

    # --- Simulated restart -------------------------------------------------
    print(SEPARATOR)
    print("SIMULATED RESTART - new MonitoringRepository instance, same file")
    print(SEPARATOR)
    del repository
    reopened = MonitoringRepository(config.MONITORING_DB_PATH)

    runs = reopened.get_runs()
    print(f"\n{len(runs)} runs recovered from disk:")
    print(runs[["run_id", "batch_id", "status"]].to_string(index=False))

    print(f"\nAll alerts after 4 cycles (deduplication keeps this list short):")
    alerts_df = reopened.get_alerts()
    print(
        alerts_df[["title", "severity", "occurrence_count", "batch_id"]]
        .to_string(index=False)
    )

    print(f"\nCRITICAL only (python manage_alerts.py persisted these across the 'restart'):")
    critical_df = reopened.get_alerts(severity="CRITICAL")
    print(critical_df[["title", "severity", "occurrence_count"]].to_string(index=False))

    print(f"\nBy category=data_quality (shows deduplication: occurrence_count > 1):")
    quality_df = reopened.get_alerts(category="data_quality")
    print(quality_df[["title", "occurrence_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
