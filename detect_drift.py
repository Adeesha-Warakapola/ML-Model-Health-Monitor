"""Entry point: compare incoming batches against the training distribution.

Run with:
    python detect_drift.py                      # stable batch, then drifted ones
    python detect_drift.py --size 500
    python detect_drift.py --small-sample-demo  # show why small batches mislead
    python detect_drift.py --json               # machine-readable output

Nothing is retrained and no model is needed: drift compares input data
against the reference distribution, so this runs on the dataset alone.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from monitor import config
from monitor.data import load_adult_data
from monitor.drift import DriftDetector, DriftThresholds
from monitor.simulation import (
    BatchSimulator,
    combine,
    get_reference_features,
    overwrite_category,
    scale_numeric,
    shift_numeric,
    split_reference_and_holdout,
)

SEPARATOR = "=" * 72


def show(detector: DriftDetector, batch, title: str) -> None:
    report = detector.detect(batch.features, batch_id=batch.batch_id)
    print(SEPARATOR)
    print(title)
    if batch.modifications:
        print(f"injected change: {', '.join(batch.modifications)}")
    print(SEPARATOR)
    print(report.summary())
    print()
    print("  Per-feature detail (features of interest first):")
    print(report.explain())
    print()

    actual = batch.actual_class_distribution().get(config.POSITIVE_LABEL, {})
    print(
        f"  Label balance untouched: {config.POSITIVE_LABEL} is "
        f"{actual.get('percent')}% of this batch "
        f"(no resampling applied anywhere)."
    )
    print()


def run(size: int, small_demo: bool, as_json: bool) -> None:
    df = load_adult_data()
    reference = get_reference_features(df)
    _, holdout = split_reference_and_holdout(df)

    detector = DriftDetector(reference)
    simulator = BatchSimulator(holdout, batch_size=size, seed=config.RANDOM_SEED)

    if as_json:
        batch = simulator.next_batch()
        report = detector.detect(batch.features, batch_id=batch.batch_id)
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return

    print(f"Reference: {len(reference):,} rows (the training distribution)")
    print(f"Holdout:   {len(holdout):,} rows (never seen in training)")
    print(f"Batch size: {size}\n")

    # 1. A stable batch: unmodified holdout rows.
    show(detector, simulator.next_batch(), "SCENARIO 1 - stable batch, nothing changed")

    # 2. Numerical drift.
    simulator.reset()
    show(
        detector,
        simulator.next_batch(
            modifier=shift_numeric("age", 15), modification_label="age + 15 years"
        ),
        "SCENARIO 2 - numerical drift (shifted age)",
    )

    # 3. Categorical drift, including a category never seen in training.
    simulator.reset()
    show(
        detector,
        simulator.next_batch(
            modifier=overwrite_category("occupation", "Prompt-Engineer", fraction=0.4),
            modification_label="40% of occupation set to an unseen category",
        ),
        "SCENARIO 3 - categorical drift (new unseen category)",
    )

    # 4. Several features at once.
    simulator.reset()
    show(
        detector,
        simulator.next_batch(
            modifier=combine(
                shift_numeric("age", 10),
                scale_numeric("hours_per_week", 1.4),
                overwrite_category("education", "Bachelors", fraction=0.6),
            ),
            modification_label="age +10, hours x1.4, 60% education=Bachelors",
        ),
        "SCENARIO 4 - multiple features drifting together",
    )

    # 5. Data quality versus drift: a broken batch, not a shifted one.
    simulator.reset()
    broken = simulator.next_batch()
    broken.features.loc[broken.features.index[:50], "age"] = -5
    broken.features.loc[broken.features.index[:30], "workclass"] = None
    report = detector.detect(broken.features, batch_id=broken.batch_id)
    print(SEPARATOR)
    print("SCENARIO 5 - data-quality failure, which is NOT the same as drift")
    print("injected change: 50 impossible ages (-5), 30 missing workclass values")
    print(SEPARATOR)
    print(report.summary())
    print()
    print("  Data-quality issues found (reported separately from drift):")
    for issue in report.validation.issues:
        print(f"    {issue}")
    print()
    print(
        "  Note the distinction: impossible values are a pipeline bug to fix,\n"
        "  while a distribution shift is a reason to consider retraining."
    )
    print()

    if small_demo:
        print(SEPARATOR)
        print("WHY BATCH SIZE MATTERS - unmodified batches at several sizes")
        print(SEPARATOR)
        print("  Every batch below is untouched holdout data, so any flag is a")
        print("  false alarm caused purely by sampling noise.\n")
        rows = []
        for small_size in (100, 200, 300, 400, 750):
            sub = BatchSimulator(holdout, batch_size=small_size, seed=11)
            n = min(10, sub.batches_remaining)
            flagged = 0
            peaks = []
            for candidate in sub.generate_batches(n):
                result = detector.detect(candidate.features)
                peaks.append(result.max_psi)
                flagged += result.status != "stable"
            rows.append(
                {
                    "batch_size": small_size,
                    "batches": n,
                    "mean max PSI": round(sum(peaks) / len(peaks), 4),
                    "falsely flagged": f"{flagged}/{n}",
                }
            )
        table = pd.DataFrame(rows)
        print("  " + table.to_string(index=False).replace("\n", "\n  "))
        print()
        print(
            f"  This is why min_batch_rows defaults to "
            f"{DriftThresholds().min_batch_rows}: PSI needs enough rows per bin.\n"
            "  A p-value would not have saved us here - the problem is variance,\n"
            "  not significance."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect data drift in batches.")
    parser.add_argument("--size", type=int, default=500, help="Rows per batch.")
    parser.add_argument(
        "--small-sample-demo",
        action="store_true",
        help="Show measured false-alarm rates at small batch sizes.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print one report as JSON and exit."
    )
    args = parser.parse_args()
    if args.size < 1:
        parser.error("--size must be at least 1")
    run(args.size, args.small_sample_demo, args.json)


if __name__ == "__main__":
    main()
