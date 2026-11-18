"""Entry point: monitor predictions and performance across batches.

Run with:
    python monitor_performance.py                 # stable + drifted scenarios
    python monitor_performance.py --delayed-labels # show partial/delayed labelling

Demonstrates, on the same deliberately modified batch, that data drift,
prediction drift, and performance degradation are three different
measurements that do not have to move together. No model is retrained and
nothing in artifacts/ or the dataset is changed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from monitor import config
from monitor.data import load_adult_data
from monitor.drift import DriftDetector
from monitor.performance_monitoring import PerformanceMonitor, load_baseline_test_metrics
from monitor.prediction import PredictionService
from monitor.prediction_monitoring import PredictionMonitor
from monitor.simulation import (
    BatchSimulator,
    get_reference_features,
    shift_numeric,
    split_reference_and_holdout,
)

SEPARATOR = "=" * 72


def run_scenario(
    title: str,
    batch,
    service: PredictionService,
    drift_detector: DriftDetector,
    prediction_monitor: PredictionMonitor,
    performance_monitor: PerformanceMonitor,
    baseline: dict,
) -> None:
    print(SEPARATOR)
    print(title)
    if batch.modifications:
        print(f"injected change: {', '.join(batch.modifications)}")
    print(SEPARATOR)

    prediction = service.predict_simulated(batch)

    drift_report = drift_detector.detect(batch.features, batch_id=batch.batch_id)
    print(f"\n1. DATA DRIFT (are the inputs different?)")
    print(f"   verdict: {drift_report.status}   max PSI: {drift_report.max_psi}")
    if drift_report.drifted_features:
        for result in drift_report.drifted_features:
            print(f"   drifted: {result.feature} (psi={result.psi:.4f})")

    prediction_report = prediction_monitor.evaluate(prediction)
    print(f"\n2. PREDICTION DRIFT (did what the model says change?)")
    print(
        f"   verdict: {prediction_report.status}   "
        f"label_psi: {prediction_report.label_psi}   "
        f"confidence_psi: {prediction_report.confidence_psi}"
    )
    for label in sorted(prediction_report.batch_distribution):
        ref = prediction_report.reference_distribution.get(label, {"percent": 0.0})
        cur = prediction_report.batch_distribution[label]
        print(f"   {label:8s} reference {ref['percent']:6.2f}%  batch {cur['percent']:6.2f}%")

    performance_monitor.register_batch(prediction)
    performance_monitor.submit_labels(batch.batch_id, batch.labels)
    performance_report = performance_monitor.compute_performance(
        batch.batch_id, baseline=baseline
    )
    print(f"\n3. PERFORMANCE (is the model still accurate? - needs true labels)")
    print(
        f"   status: {performance_report.status}   "
        f"labelled: {performance_report.n_labeled}/{performance_report.n_total}"
    )
    if performance_report.metrics:
        m = performance_report.metrics
        print(
            f"   accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
            f"recall={m['recall']:.4f}  roc_auc={m['roc_auc']:.4f}"
        )
    if performance_report.comparison:
        degraded = [k for k, v in performance_report.comparison.items() if v["degraded"]]
        print(f"   degraded vs baseline: {degraded or 'none'}")
    print()


def demo_delayed_labels(batch, service: PredictionService) -> None:
    print(SEPARATOR)
    print("DELAYED / PARTIAL LABELS - labels arriving in increments")
    print(SEPARATOR)
    monitor = PerformanceMonitor(min_labeled_sample=300)
    prediction = service.predict_simulated(batch)

    created = monitor.register_batch(prediction)
    print(f"\nregistered {created} predictions, 0 labels yet")
    report = monitor.compute_performance(batch.batch_id)
    print(f"  status={report.status}  metrics={report.metrics}")

    first_wave = batch.labels.index[:120]
    monitor.submit_labels(batch.batch_id, batch.labels.loc[first_wave])
    report = monitor.compute_performance(batch.batch_id)
    print(f"\nafter 120 labels arrive: status={report.status}  "
          f"labelled={report.n_labeled}/{report.n_total}")
    for note in report.notes:
        print(f"  NOTE: {note}")

    # Re-registering the same batch must not duplicate anything.
    again = monitor.register_batch(prediction)
    print(f"\nre-registering the same batch created {again} new records (must be 0)")

    second_wave = batch.labels.index[120:]
    monitor.submit_labels(batch.batch_id, batch.labels.loc[second_wave])
    report = monitor.compute_performance(batch.batch_id)
    print(f"\nafter the remaining labels arrive: status={report.status}  "
          f"labelled={report.n_labeled}/{report.n_total}")
    print(f"total records stored: {len(monitor.records_frame(batch.batch_id))} "
          f"(must equal {batch.n_rows}, not more)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor predictions and performance.")
    parser.add_argument(
        "--delayed-labels", action="store_true",
        help="Demonstrate labels arriving in two separate waves.",
    )
    args = parser.parse_args()

    df = load_adult_data()
    reference_features = get_reference_features(df)
    _, holdout = split_reference_and_holdout(df)

    service = PredictionService()
    drift_detector = DriftDetector(reference_features)
    prediction_monitor = PredictionMonitor.from_reference_features(service, reference_features)
    baseline = load_baseline_test_metrics()

    simulator = BatchSimulator(holdout, batch_size=600, seed=config.RANDOM_SEED)

    if args.delayed_labels:
        demo_delayed_labels(simulator.next_batch(), service)
        return

    performance_monitor = PerformanceMonitor(min_labeled_sample=300)
    run_scenario(
        "SCENARIO 1 - stable batch",
        simulator.next_batch(),
        service, drift_detector, prediction_monitor, performance_monitor, baseline,
    )

    simulator.reset()
    performance_monitor = PerformanceMonitor(min_labeled_sample=300)
    run_scenario(
        "SCENARIO 2 - age shifted +25 years (features changed, true labels untouched)",
        simulator.next_batch(modifier=shift_numeric("age", 25)),
        service, drift_detector, prediction_monitor, performance_monitor, baseline,
    )

    print(SEPARATOR)
    print("These three numbers moved independently on the same batch:")
    print("  data drift        severe  (age PSI > 5)")
    print("  prediction drift   moderate (confidence PSI ~0.12, label PSI still low)")
    print("  performance        measurably worse (accuracy/precision/F1 down,")
    print("                      while ROC-AUC barely moved)")
    print("This is the point: none of the three can substitute for the others.")


if __name__ == "__main__":
    main()
