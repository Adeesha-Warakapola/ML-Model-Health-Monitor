"""Entry point: inspect the Adult dataset and write the exploration report.

Run with:  python explore_dataset.py
Add --refresh to re-download from UCI instead of using the local cache.

This script adds `src/` to the import path so it can be run directly without
installing the project as a package.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from monitor import config, explore
from monitor.data import load_adult_data
from monitor.report import build_markdown_report, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the UCI Adult dataset.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the dataset from UCI instead of using the local cache.",
    )
    args = parser.parse_args()

    df = load_adult_data(refresh=args.refresh)

    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns\n")
    print("Schema")
    print(explore.schema_summary(df).to_string(index=False))
    print("\nMissing values")
    print(explore.missing_value_summary(df).to_string(index=False))
    print("\nDuplicates")
    print(explore.duplicate_summary(df))
    print("\nTarget distribution")
    print(explore.target_distribution(df).to_string(index=False))
    print(f"Imbalance ratio: {explore.class_imbalance_ratio(df)}:1")

    save_report(build_markdown_report(df))
    print(f"\nReport written to {config.EXPLORATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
