"""Entry point: simulate incoming batches and score them with the saved model.

Run with:
    python simulate_batches.py                        # 3 batches of 500
    python simulate_batches.py --batches 5 --size 200
    python simulate_batches.py --seed 7
    python simulate_batches.py --demo-modifier        # show the drift hook

Nothing is retrained and nothing is overwritten. The model is loaded from
artifacts/ and used only to predict.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from monitor import config
from monitor.data import load_adult_data
from monitor.prediction import PredictionService
from monitor.simulation import (
    BatchSimulator,
    combine,
    overwrite_category,
    shift_numeric,
    split_reference_and_holdout,
)


def show_batch(batch, result) -> None:
    print(batch.summary())
    print(result.summary())

    # Features and predictions side by side - the record a monitor would keep.
    frame = result.to_frame()
    columns = ["age", "education", "occupation", "hours_per_week", "prediction"]
    probability_column = f"probability_{config.POSITIVE_LABEL}"
    if probability_column in frame.columns:
        columns.append(probability_column)
    print("  first 5 rows of the stored record:")
    preview = frame[columns].head(5).to_string(index=False)
    print("    " + preview.replace("\n", "\n    "))
    print()


def run(n_batches: int, size: int, seed: int, demo_modifier: bool) -> None:
    df = load_adult_data()
    reference, holdout = split_reference_and_holdout(df)

    print(f"Dataset           {len(df):,} rows")
    print(f"Reference (train) {len(reference):,} rows  - the model learned from these")
    print(f"Holdout (unseen)  {len(holdout):,} rows  - batches are drawn from these")
    overlap = set(reference.index) & set(holdout.index)
    print(f"Overlap           {len(overlap)} rows  - must be 0\n")

    service = PredictionService()
    print(
        f"Loaded {service.model_type} from {config.BASELINE_MODEL_PATH.name} "
        f"(threshold {service.threshold}, "
        f"probabilities {'yes' if service.supports_probabilities else 'no'})\n"
    )

    simulator = BatchSimulator(holdout, batch_size=size, seed=seed)
    print(
        f"Simulator ready: {simulator.rows_remaining:,} rows available, "
        f"{simulator.batches_remaining} batches of {size} possible\n"
    )

    rates = []
    for _ in range(n_batches):
        batch = simulator.next_batch()
        result = service.predict_simulated(batch)
        show_batch(batch, result)

        actual = batch.actual_class_distribution().get(config.POSITIVE_LABEL, {})
        predicted = result.prediction_distribution().get(config.POSITIVE_LABEL, {})
        rates.append(
            {
                "batch": batch.batch_index,
                "rows": batch.n_rows,
                f"actual {config.POSITIVE_LABEL} %": actual.get("percent", 0.0),
                f"predicted {config.POSITIVE_LABEL} %": predicted.get("percent", 0.0),
            }
        )

    print("Natural variation across batches (no balancing applied)")
    table = pd.DataFrame(rates)
    print("  " + table.to_string(index=False).replace("\n", "\n  "))
    full_rate = (holdout[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean() * 100
    print(f"  whole holdout set: {full_rate:.2f}% {config.POSITIVE_LABEL}\n")

    if demo_modifier:
        print("=" * 70)
        print("Drift hook demonstration (Phase 5 will detect this; here it is")
        print("only injected to show the mechanism works)\n")
        simulator.reset()
        plain = simulator.next_batch()
        plain_result = service.predict_simulated(plain)

        simulator.reset()
        modified = simulator.next_batch(
            modifier=combine(
                shift_numeric("age", 15),
                overwrite_category("occupation", "Prompt-Engineer", fraction=0.5),
            ),
            modification_label="age +15, half of occupation set to an unseen value",
        )
        modified_result = service.predict_simulated(modified)

        print(modified.summary())
        print()
        print("  Same rows, before and after the modification:")
        comparison = pd.DataFrame(
            {
                "metric": ["mean age", f"predicted {config.POSITIVE_LABEL} %", f"mean P({config.POSITIVE_LABEL})"],
                "original": [
                    round(plain.features["age"].mean(), 2),
                    plain_result.prediction_distribution()
                    .get(config.POSITIVE_LABEL, {})
                    .get("percent", 0.0),
                    round(plain_result.probabilities.mean(), 4),
                ],
                "modified": [
                    round(modified.features["age"].mean(), 2),
                    modified_result.prediction_distribution()
                    .get(config.POSITIVE_LABEL, {})
                    .get("percent", 0.0),
                    round(modified_result.probabilities.mean(), 4),
                ],
            }
        )
        print("    " + comparison.to_string(index=False).replace("\n", "\n    "))
        print()
        print("  The unseen 'Prompt-Engineer' category was reported by validation")
        print("  and encoded as all-zeros, so scoring continued without error:")
        for issue in modified_result.validation.issues:
            if issue.check == "unknown_categories":
                print(f"    {issue}")
        print()
        print("  Labels were untouched by the modification:")
        print(
            f"    actual {config.POSITIVE_LABEL}: "
            f"{plain.actual_class_distribution().get(config.POSITIVE_LABEL, {}).get('percent')}% "
            f"-> {modified.actual_class_distribution().get(config.POSITIVE_LABEL, {}).get('percent')}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate incoming batches and score them."
    )
    parser.add_argument("--batches", type=int, default=3, help="How many batches.")
    parser.add_argument("--size", type=int, default=500, help="Rows per batch.")
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED, help="Sampling seed."
    )
    parser.add_argument(
        "--demo-modifier",
        action="store_true",
        help="Also show the controlled-change hook that Phase 5 will use.",
    )
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be at least 1")
    if args.size < 1:
        parser.error("--size must be at least 1")
    run(args.batches, args.size, args.seed, args.demo_modifier)


if __name__ == "__main__":
    main()
