"""Tests for the evaluation functions.

Where possible these use a tiny stub model with hand-chosen predictions, so
the expected metric values can be worked out on paper. That makes the tests
check the metric logic rather than just agreeing with whatever the real model
happens to produce.
"""

import numpy as np
import pandas as pd
import pytest

from monitor import config
from monitor.evaluation import (
    class_distribution,
    classification_report_text,
    confusion_matrix_frame,
    evaluate_by_group,
    evaluate_classifier,
    fairness_summary,
    format_metrics_report,
    majority_class_baseline,
    positive_class_scores,
    predict_with_threshold,
)

LOW = "<=50K"
HIGH = ">50K"


class StubModel:
    """Returns predictions handed to it, so metrics can be checked by hand."""

    def __init__(self, predictions, scores=None):
        self.predictions = list(predictions)
        self.classes_ = np.array([LOW, HIGH])
        self._scores = scores

    def predict(self, X):
        return np.array(self.predictions[: len(X)])

    def predict_proba(self, X):
        if self._scores is None:
            # Confident probability in the direction of each prediction.
            high = [0.9 if p == HIGH else 0.1 for p in self.predictions[: len(X)]]
        else:
            high = list(self._scores[: len(X)])
        return np.column_stack([1 - np.array(high), np.array(high)])


def frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"anything": range(n)})


# --- class distribution -----------------------------------------------------


def test_class_distribution_counts_and_percentages():
    y = pd.Series([LOW, LOW, LOW, HIGH])
    result = class_distribution(y)

    assert result[LOW] == {"count": 3, "percent": 75.0}
    assert result[HIGH] == {"count": 1, "percent": 25.0}


def test_majority_class_baseline_is_the_largest_share():
    y = pd.Series([LOW] * 76 + [HIGH] * 24)
    assert majority_class_baseline(y) == 0.76


# --- metric correctness -----------------------------------------------------


def test_metrics_match_hand_calculated_values():
    # 2 true positives, 1 false positive, 1 false negative, 1 true negative.
    y_true = pd.Series([HIGH, HIGH, HIGH, LOW, LOW])
    y_pred = [HIGH, HIGH, LOW, HIGH, LOW]
    metrics = evaluate_classifier(StubModel(y_pred), frame(5), y_true)

    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["precision"] == pytest.approx(2 / 3)  # 2 of 3 predicted HIGH
    assert metrics["recall"] == pytest.approx(2 / 3)  # found 2 of 3 real HIGH
    assert metrics["f1_score"] == pytest.approx(2 / 3)
    assert metrics["n_samples"] == 5


def test_metrics_are_reported_for_the_positive_class_not_the_majority():
    # Everything predicted LOW: recall for the minority HIGH class must be 0,
    # even though overall accuracy looks respectable.
    y_true = pd.Series([LOW] * 8 + [HIGH] * 2)
    metrics = evaluate_classifier(StubModel([LOW] * 10), frame(10), y_true)

    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0  # zero_division guard, not a crash


def test_roc_auc_is_perfect_when_scores_rank_the_classes_correctly():
    y_true = pd.Series([LOW, LOW, HIGH, HIGH])
    model = StubModel([LOW, LOW, HIGH, HIGH], scores=[0.1, 0.2, 0.8, 0.9])

    assert evaluate_classifier(model, frame(4), y_true)["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_half_when_scores_are_uninformative():
    y_true = pd.Series([LOW, HIGH, LOW, HIGH])
    model = StubModel([LOW] * 4, scores=[0.5] * 4)

    assert evaluate_classifier(model, frame(4), y_true)["roc_auc"] == pytest.approx(0.5)


# --- confusion matrix -------------------------------------------------------


def test_confusion_matrix_is_labelled_and_oriented_correctly():
    y_true = pd.Series([HIGH, HIGH, LOW])
    y_pred = [HIGH, LOW, LOW]
    matrix = confusion_matrix_frame(y_true, y_pred)

    assert list(matrix.index) == [f"actual {LOW}", f"actual {HIGH}"]
    assert list(matrix.columns) == [f"predicted {LOW}", f"predicted {HIGH}"]
    assert matrix.loc[f"actual {HIGH}", f"predicted {HIGH}"] == 1  # true positive
    assert matrix.loc[f"actual {HIGH}", f"predicted {LOW}"] == 1  # missed one
    assert matrix.loc[f"actual {LOW}", f"predicted {LOW}"] == 1


def test_confusion_matrix_totals_equal_the_row_count():
    y_true = pd.Series([HIGH, LOW, LOW, HIGH, LOW])
    matrix = confusion_matrix_frame(y_true, [HIGH, HIGH, LOW, LOW, LOW])
    assert matrix.to_numpy().sum() == 5


# --- output shape -----------------------------------------------------------


def test_evaluate_returns_every_expected_key():
    y_true = pd.Series([LOW, HIGH])
    metrics = evaluate_classifier(StubModel([LOW, HIGH]), frame(2), y_true)

    expected = {
        "n_samples",
        "class_distribution",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "confusion_matrix_labels",
        "classification_report",
    }
    assert set(metrics) == expected


def test_all_scores_are_plain_floats_so_json_can_store_them():
    y_true = pd.Series([LOW, HIGH])
    metrics = evaluate_classifier(StubModel([LOW, HIGH]), frame(2), y_true)

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert type(metrics[key]) is float


def test_classification_report_covers_both_classes():
    report = classification_report_text(pd.Series([LOW, HIGH]), [LOW, HIGH])
    assert LOW in report
    assert HIGH in report


def test_format_metrics_report_includes_the_headline_numbers():
    y_true = pd.Series([LOW, HIGH])
    metrics = evaluate_classifier(StubModel([LOW, HIGH]), frame(2), y_true)
    text = format_metrics_report(metrics, title="Test run")

    assert "Test run" in text
    assert "accuracy" in text
    assert "Confusion matrix" in text


# --- subgroup / fairness measurement ----------------------------------------


def group_frame() -> pd.DataFrame:
    return pd.DataFrame({"sex": ["Male"] * 4 + ["Female"] * 4})


def test_evaluate_by_group_splits_metrics_per_group():
    X = group_frame()
    # Males: 2 real HIGH, both found. Females: 2 real HIGH, neither found.
    y = pd.Series([HIGH, HIGH, LOW, LOW, HIGH, HIGH, LOW, LOW])
    predictions = [HIGH, HIGH, LOW, LOW, LOW, LOW, LOW, LOW]

    table = evaluate_by_group(StubModel(predictions), X, y, "sex")
    male = table[table["group"] == "Male"].iloc[0]
    female = table[table["group"] == "Female"].iloc[0]

    assert male["n"] == 4 and female["n"] == 4
    assert male["recall"] == 1.0
    assert female["recall"] == 0.0
    assert male["selection_rate"] == 0.5
    assert female["selection_rate"] == 0.0


def test_evaluate_by_group_records_the_real_positive_rate():
    X = group_frame()
    y = pd.Series([HIGH, HIGH, HIGH, LOW, HIGH, LOW, LOW, LOW])
    table = evaluate_by_group(StubModel([LOW] * 8), X, y, "sex")

    male = table[table["group"] == "Male"].iloc[0]
    assert male["actual_positive_rate"] == 0.75


def test_evaluate_by_group_rejects_an_unknown_column():
    with pytest.raises(ValueError, match="not a column"):
        evaluate_by_group(StubModel([LOW] * 8), group_frame(), pd.Series([LOW] * 8), "height")


def test_fairness_summary_reports_disparity_ratios():
    X = group_frame()
    y = pd.Series([HIGH, HIGH, LOW, LOW, HIGH, HIGH, LOW, LOW])
    predictions = [HIGH, HIGH, LOW, LOW, LOW, LOW, LOW, LOW]

    summary = fairness_summary(StubModel(predictions), X, y, ["sex"])["sex"]

    # Groups are below the n>=100 reliability floor, so all are compared.
    assert summary["n_groups_compared"] == 2
    assert summary["recall_ratio"] == 0.0  # 0.0 / 1.0
    assert summary["selection_rate_ratio"] == 0.0
    assert len(summary["groups"]) == 2


def test_fairness_ratio_is_one_when_groups_are_treated_alike():
    X = group_frame()
    y = pd.Series([HIGH, LOW, HIGH, LOW] * 2)
    predictions = [HIGH, LOW, HIGH, LOW] * 2

    summary = fairness_summary(StubModel(predictions), X, y, ["sex"])["sex"]

    assert summary["recall_ratio"] == 1.0
    assert summary["selection_rate_ratio"] == 1.0


def test_fairness_summary_covers_every_requested_column():
    X = group_frame()
    X["race"] = ["White"] * 8
    y = pd.Series([HIGH, LOW] * 4)

    summary = fairness_summary(StubModel([HIGH, LOW] * 4), X, y, ["sex", "race"])

    assert set(summary) == {"sex", "race"}


# --- threshold-based prediction ---------------------------------------------


def test_predict_with_threshold_moves_the_cut_off():
    model = StubModel([LOW] * 4, scores=[0.2, 0.4, 0.6, 0.8])

    eager = predict_with_threshold(model, frame(4), 0.3)
    cautious = predict_with_threshold(model, frame(4), 0.7)

    assert list(eager) == [LOW, HIGH, HIGH, HIGH]
    assert list(cautious) == [LOW, LOW, LOW, HIGH]


def test_predict_with_threshold_rejects_impossible_thresholds():
    model = StubModel([LOW], scores=[0.5])
    for bad in (0.0, 1.0, -1.0, 2.0):
        with pytest.raises(ValueError, match="between 0 and 1"):
            predict_with_threshold(model, frame(1), bad)


# --- error handling ---------------------------------------------------------


def test_evaluating_an_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="empty dataset"):
        evaluate_classifier(StubModel([]), frame(0), pd.Series([], dtype=object))


def test_mismatched_feature_and_label_lengths_are_rejected():
    with pytest.raises(ValueError, match="different lengths"):
        evaluate_classifier(StubModel([LOW] * 3), frame(3), pd.Series([LOW, HIGH]))


def test_model_without_the_positive_label_is_rejected_clearly():
    model = StubModel([LOW])
    model.classes_ = np.array([LOW, "something-else"])

    with pytest.raises(ValueError, match="positive label"):
        positive_class_scores(model, frame(1))


def test_positive_scores_are_taken_from_the_right_column():
    # classes_ reversed, so the positive column is index 0 rather than 1.
    model = StubModel([HIGH])
    model.classes_ = np.array([HIGH, LOW])
    model.predict_proba = lambda X: np.array([[0.7, 0.3]])

    assert positive_class_scores(model, frame(1))[0] == pytest.approx(0.7)
