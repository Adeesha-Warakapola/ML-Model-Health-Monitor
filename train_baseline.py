"""Entry point: train the baseline model, evaluate it, and save everything.

Run with:
    python train_baseline.py                   # full run
    python train_baseline.py --compare         # also train Random Forest
    python train_baseline.py --search          # also search hyperparameters
    python train_baseline.py --threshold 0.5   # keep the default cut-off
    python train_baseline.py --model random_forest

What happens, in order:

1. Split 80/20, stratified, fixed seed.
2. Cross-validate on the training half, to get metrics with a spread rather
   than a single number.
3. Choose the decision threshold from out-of-fold predictions on the
   training half only - never from the test set.
4. Train the final model on the whole training half with that threshold
   baked in.
5. Evaluate on the untouched test set, measure per-group behaviour, and save
   the model, its metadata and the metrics.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from monitor import config
from monitor.data import load_adult_data
from monitor.evaluation import (
    evaluate_classifier,
    fairness_summary,
    format_metrics_report,
    majority_class_baseline,
)
from monitor.model import (
    BASELINE_MODEL,
    MODEL_BUILDERS,
    build_metadata,
    build_model_pipeline,
    get_threshold,
    save_json,
    save_model,
    train_baseline,
)
from monitor.preprocessing import split_features_target, split_train_test
from monitor.tuning import cross_validate_model, search_hyperparameters, select_threshold

FAIRNESS_COLUMNS = ["sex", "race"]


def describe_distribution(name: str, y: pd.Series) -> None:
    counts = y.value_counts()
    parts = [
        f"{label} {count:,} ({count / len(y) * 100:.2f}%)"
        for label, count in counts.items()
    ]
    print(f"  {name:<6} {len(y):>6,} rows   " + "   ".join(parts))


def print_cv(cv: dict) -> None:
    print(f"Cross-validation on the training half ({cv['n_splits']} stratified folds)")
    for metric, stats in cv["summary"].items():
        print(f"  {metric:<10} {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print()


def print_threshold(selection: dict) -> None:
    default = selection["default_threshold_scores"]
    chosen = selection["selected_scores"]
    print(
        f"Decision threshold chosen from out-of-fold predictions "
        f"(maximising {selection['objective']})"
    )
    print(f"  selected threshold: {selection['selected_threshold']:.2f}")
    print(f"  {'':<12}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}")
    for label, scores in (("at 0.50", default), ("at chosen", chosen)):
        print(
            f"  {label:<12}{scores['accuracy']:>10.4f}{scores['precision']:>11.4f}"
            f"{scores['recall']:>9.4f}{scores['f1_score']:>9.4f}"
        )
    if not selection["constraint_satisfied"]:
        print("  NOTE: no threshold met the requested precision floor; it was ignored.")
    print()


def print_fairness(fairness: dict) -> None:
    print("Per-group behaviour on the test set (measured, not corrected)")
    for column, result in fairness.items():
        print(f"  by {column}:")
        table = pd.DataFrame(result["groups"])
        print("    " + table.to_string(index=False).replace("\n", "\n    "))
        print(
            f"    selection_rate_ratio {result['selection_rate_ratio']}   "
            f"recall_ratio {result['recall_ratio']}   "
            f"(groups with n>={result['min_group_size_for_ratios']}: "
            f"{result['n_groups_compared']})"
        )
    print()


def run(model_name: str, compare: bool, search: bool, fixed_threshold: float | None) -> None:
    df = load_adult_data()
    print(f"Loaded {len(df):,} rows\n")

    train_df, test_df = split_train_test(df, test_size=config.TEST_SIZE)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    print("Class distribution (natural balance preserved, no resampling)")
    describe_distribution("train", y_train)
    describe_distribution("test", y_test)
    print()

    cv = cross_validate_model(lambda: build_model_pipeline(model_name), X_train, y_train)
    print_cv(cv)

    if fixed_threshold is None:
        selection = select_threshold(y_train, cv["out_of_fold_probabilities"])
        print_threshold(selection)
        threshold = selection["selected_threshold"]
    else:
        selection = None
        threshold = fixed_threshold
        print(f"Using the threshold given on the command line: {threshold}\n")

    hyperparameters = None
    if search:
        print("Searching hyperparameters (ROC-AUC, threshold-independent)")
        hyperparameters = search_hyperparameters(
            lambda: build_model_pipeline(model_name), X_train, y_train
        )
        for row in hyperparameters["results"]:
            print(f"  C={row['C']:<6} roc_auc {row['mean_roc_auc']:.4f} +/- {row['std_roc_auc']:.4f}")
        print(
            f"  best C={hyperparameters['best_C']} "
            f"({hyperparameters['best_roc_auc']:.4f}), "
            f"default C=1.0 ({hyperparameters['default_C_roc_auc']:.4f}), "
            f"gain {hyperparameters['improvement_over_default']:+.4f}\n"
        )

    model, _, _, messages = train_baseline(df, model_name=model_name, threshold=threshold)
    for message in messages:
        print(f"WARNING: {message}")

    train_metrics = evaluate_classifier(model, X_train, y_train)
    test_metrics = evaluate_classifier(model, X_test, y_test)

    print(
        format_metrics_report(
            test_metrics, title=f"{model_name} - TEST set at threshold {get_threshold(model):.2f}"
        )
    )
    print(
        f"Reference point: always predicting the majority class would score "
        f"{majority_class_baseline(y_test):.4f} accuracy."
    )
    print(
        f"Train accuracy {train_metrics['accuracy']:.4f} vs "
        f"test accuracy {test_metrics['accuracy']:.4f} "
        f"(gap {train_metrics['accuracy'] - test_metrics['accuracy']:+.4f})\n"
    )

    fairness = fairness_summary(model, X_test, y_test, FAIRNESS_COLUMNS)
    print_fairness(fairness)

    if compare:
        comparison = {model_name: test_metrics}
        for other in MODEL_BUILDERS:
            if other == model_name:
                continue
            print(f"Training comparison model: {other} (same threshold)")
            other_model, _, _, other_messages = train_baseline(
                df, model_name=other, threshold=threshold
            )
            for message in other_messages:
                print(f"WARNING: {message}")
            comparison[other] = evaluate_classifier(other_model, X_test, y_test)

        print("\nComparison on the TEST set")
        print(f"  {'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'roc_auc':>9}")
        for name, metrics in comparison.items():
            print(
                f"  {name:<22}{metrics['accuracy']:>10.4f}{metrics['precision']:>11.4f}"
                f"{metrics['recall']:>9.4f}{metrics['f1_score']:>9.4f}{metrics['roc_auc']:>9.4f}"
            )
        print()

    cv_summary = {"n_splits": cv["n_splits"], "summary": cv["summary"], "per_fold": cv["per_fold"]}
    save_model(model)
    metadata = build_metadata(
        model,
        model_name,
        train_metrics,
        test_metrics,
        n_rows_total=len(df),
        cv_summary=cv_summary,
        threshold_selection=selection,
        hyperparameter_search=hyperparameters,
        fairness=fairness,
    )
    save_json(metadata, config.BASELINE_METADATA_PATH)
    save_json(
        {
            "model_name": model_name,
            "model_type": metadata["model_type"],
            "trained_on": metadata["trained_on"],
            "decision_threshold": get_threshold(model),
            "majority_class_accuracy": majority_class_baseline(y_test),
            "cross_validation": cv_summary,
            "threshold_selection": selection,
            "hyperparameter_search": hyperparameters,
            "fairness": fairness,
            "train": train_metrics,
            "test": test_metrics,
        },
        config.BASELINE_METRICS_PATH,
    )

    print(f"Model saved to     {config.BASELINE_MODEL_PATH}")
    print(f"Metadata saved to  {config.BASELINE_METADATA_PATH}")
    print(f"Metrics saved to   {config.BASELINE_METRICS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline classifier.")
    parser.add_argument(
        "--model", default=BASELINE_MODEL, choices=sorted(MODEL_BUILDERS),
        help="Which model to train and save as the baseline.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also train the other available model and print a comparison.",
    )
    parser.add_argument(
        "--search", action="store_true",
        help="Also run the hyperparameter search (slower).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Use this decision threshold instead of selecting one (e.g. 0.5).",
    )
    args = parser.parse_args()
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    run(args.model, args.compare, args.search, args.threshold)


if __name__ == "__main__":
    main()
