"""One monitoring cycle: predict, check drift, check performance, alert, persist.

This is the sequencing that both `manage_alerts.py` (the CLI demo) and the
Streamlit dashboard need, pulled into one place so the two never run the
pipeline differently. Nothing here is a new calculation - it only calls
`PredictionService`, `DriftDetector`, `PredictionMonitor`,
`PerformanceMonitor`, `generate_alerts` and `MonitoringRepository`, each
exactly once, in the same order Phase 7 established.
"""

from dataclasses import dataclass
from pathlib import Path

from . import config
from .alerts import generate_alerts
from .data import load_adult_data
from .drift import DriftDetector, DriftReport
from .performance_monitoring import PerformanceMonitor, PerformanceReport, load_baseline_test_metrics
from .prediction import BatchPrediction, PredictionService, utc_now
from .prediction_monitoring import PredictionDriftReport, PredictionMonitor
from .simulation import BatchSimulator, SimulatedBatch, get_reference_features, split_reference_and_holdout
from .storage import MonitoringRepository, make_run_id


@dataclass
class MonitoringContext:
    """Everything one monitoring cycle needs, built once and reused.

    Loading the dataset and scoring the reference batch (to build
    `prediction_monitor`'s baseline) is a fixed cost that every cycle would
    otherwise repeat. Both the CLI script and the dashboard build exactly one
    of these per process.
    """

    service: PredictionService
    drift_detector: DriftDetector
    prediction_monitor: PredictionMonitor
    baseline: dict
    repository: MonitoringRepository
    simulator: BatchSimulator


def build_monitoring_context(
    db_path: Path | None = None,
    batch_size: int = 600,
    seed: int = config.RANDOM_SEED,
) -> MonitoringContext:
    """Load the dataset once and build every service a monitoring cycle needs.

    Does not train or retrain anything: `PredictionService` loads the
    already-trained artifact saved by `train_baseline.py`. If that artifact
    is missing, `PredictionService` raises `FileNotFoundError` with
    instructions, which is left to propagate rather than being caught here.
    """
    df = load_adult_data()
    reference_features = get_reference_features(df)
    _, holdout = split_reference_and_holdout(df)

    service = PredictionService()
    drift_detector = DriftDetector(reference_features)
    prediction_monitor = PredictionMonitor.from_reference_features(service, reference_features)
    baseline = load_baseline_test_metrics()
    repository = MonitoringRepository(db_path or config.MONITORING_DB_PATH)
    simulator = BatchSimulator(holdout, batch_size=batch_size, seed=seed)

    return MonitoringContext(
        service=service,
        drift_detector=drift_detector,
        prediction_monitor=prediction_monitor,
        baseline=baseline,
        repository=repository,
        simulator=simulator,
    )


def run_monitoring_cycle(batch: SimulatedBatch, context: MonitoringContext) -> dict:
    """One full cycle: predict, check drift, check performance, alert, persist.

    A fresh `PerformanceMonitor` is used for each cycle, so the performance
    report reflects this batch's own labels rather than accumulating across
    every batch ever run in the process - the same per-batch behaviour Phase
    7 demonstrated. Returns a dict with both the summary fields the CLI
    script prints and the full report objects a richer UI needs.
    """
    run_id = make_run_id()
    context.repository.save_run(run_id, batch.batch_id, started_at=utc_now())

    prediction: BatchPrediction = context.service.predict_simulated(batch)
    drift_report: DriftReport = context.drift_detector.detect(batch.features, batch_id=batch.batch_id)
    prediction_report: PredictionDriftReport = context.prediction_monitor.evaluate(prediction)

    performance_monitor = PerformanceMonitor(min_labeled_sample=300)
    performance_monitor.register_batch(prediction)
    performance_monitor.submit_labels(batch.batch_id, batch.labels)
    performance_report: PerformanceReport = performance_monitor.compute_performance(
        batch.batch_id, baseline=context.baseline
    )

    alerts = generate_alerts(
        batch_id=batch.batch_id,
        validation=prediction.validation,
        drift=drift_report,
        prediction_drift=prediction_report,
        performance=performance_report,
    )

    actual_positive_rate = (batch.labels == config.POSITIVE_LABEL).mean()
    context.repository.save_batch_summary(
        run_id, prediction, actual_positive_rate=actual_positive_rate,
        n_labeled=len(batch.labels),
    )
    context.repository.save_drift_report(run_id, drift_report)
    context.repository.save_prediction_drift_report(run_id, prediction_report)
    context.repository.save_performance_report(run_id, performance_report)
    outcomes = context.repository.save_alerts(alerts, run_id)
    context.repository.complete_run(run_id)

    return {
        "run_id": run_id,
        "batch_id": batch.batch_id,
        "alerts": alerts,
        "outcomes": outcomes,
        "drift_status": drift_report.status,
        "prediction_status": prediction_report.status,
        "performance_status": performance_report.status,
        "prediction": prediction,
        "drift_report": drift_report,
        "prediction_report": prediction_report,
        "performance_report": performance_report,
        "batch": batch,
    }
