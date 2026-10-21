"""Entry point: validate the dataset, fit the preprocessor, and save it.

Run with:  python validate_and_preprocess.py

This demonstrates the exact flow that monitoring batches will follow later:
validate first, then transform with a preprocessor that was fitted only on
training data. The fitted preprocessor is saved to artifacts/ so later
phases reuse this one rather than fitting a new one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from monitor import config
from monitor.data import load_adult_data
from monitor.preprocessing import (
    fit_preprocessor,
    get_feature_names,
    save_preprocessor,
    split_features_target,
    split_train_test,
    transform_batch,
)
from monitor.validation import validate_batch


def main() -> None:
    df = load_adult_data()
    print(f"Loaded {len(df):,} rows\n")

    train_df, test_df = split_train_test(df)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    positive = config.POSITIVE_LABEL
    print(f"Train: {len(X_train):,} rows  ({(y_train == positive).mean() * 100:.2f}% {positive})")
    print(f"Test:  {len(X_test):,} rows  ({(y_test == positive).mean() * 100:.2f}% {positive})")
    print("The class balance matches the full dataset, so the natural")
    print("imbalance is preserved in both splits.\n")

    print("Validating the training data")
    train_result = validate_batch(X_train)
    print(train_result.summary())
    train_result.raise_if_invalid()

    print("\nFitting the preprocessor on training data only")
    preprocessor = fit_preprocessor(X_train)
    print(f"Produces {len(get_feature_names(preprocessor))} features")

    print("\nValidating and transforming the held-out test data")
    test_result = validate_batch(X_test)
    print(test_result.summary())
    transformed = transform_batch(preprocessor, X_test)
    print(f"Transformed shape: {transformed.shape}")

    save_preprocessor(preprocessor)
    print(f"\nFitted preprocessor saved to {config.PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
