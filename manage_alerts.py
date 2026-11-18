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
from monitor.monitoring_cycle import build_monitoring_context, run_monitoring_cycle
from monitor.simulation import overwrite_category, shift_numeric
from monitor.storage import MonitoringRepository

SEPARATOR = "=" * 72


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

    context = build_monitoring_context(batch_size=600)
    print(f"Monitoring database: {config.MONITORING_DB_PATH}\n")

    # Scenario 1: stable batch. Expect only the dataset's known, recurring
    # missing-value warnings (Phase 1/2 found workclass/occupation/
    # native_country run ~2-6% missing) - nothing data-drift, prediction-
    # drift, or performance related.
    result1 = run_monitoring_cycle(context.simulator.next_batch(), context)
    describe_cycle("SCENARIO 1 - normal batch (expect WARNING at most)", result1)

    # Scenario 2: the same kind of batch again. The three recurring
    # data-quality warnings must deduplicate into the same three alert rows
    # with a higher occurrence_count, not create three new ones.
    result2 = run_monitoring_cycle(context.simulator.next_batch(), context)
    describe_cycle(
        "SCENARIO 2 - another normal batch (watch occurrence counts go up, not rows)",
        result2,
    )

    # Scenario 3: a batch with an unseen category affecting 5% of rows.
    # Measured directly (see docs/alert_management.md): 5% gives PSI 0.1587,
    # which lands in the moderate band (0.10-0.25) and so should alert at
    # WARNING, not CRITICAL. A larger fraction (8%+) crosses into drift.
    context.simulator.reset()
    warning_batch = context.simulator.next_batch(
        modifier=overwrite_category("occupation", "Gig-economy", fraction=0.05)
    )
    result3 = run_monitoring_cycle(warning_batch, context)
    describe_cycle("SCENARIO 3 - moderate problem (expect WARNING)", result3)

    # Scenario 4: a severe, deliberate shift. Expect CRITICAL alerts for
    # data drift and for confirmed performance degradation.
    context.simulator.reset()
    critical_batch = context.simulator.next_batch(modifier=shift_numeric("age", 25))
    result4 = run_monitoring_cycle(critical_batch, context)
    describe_cycle("SCENARIO 4 - severe problem (expect CRITICAL)", result4)

    # --- Simulated restart -------------------------------------------------
    print(SEPARATOR)
    print("SIMULATED RESTART - new MonitoringRepository instance, same file")
    print(SEPARATOR)
    del context
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
