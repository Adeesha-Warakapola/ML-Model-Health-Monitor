"""Measure how well a fitted classifier performs.

These functions take an already-fitted pipeline and some labelled data, and
return plain dictionaries and DataFrames. Nothing here trains, saves or
prints anything, which means the same functions are reused later to score a
monitoring batch once its true labels arrive.

Why several metrics rather than just accuracy: the income classes are
imbalanced about 3.18 to 1, so a model that predicted `<=50K` for everybody
would score roughly 76% accuracy while being completely useless. Accuracy
alone cannot tell the difference between that and a genuinely good model.
See docs/baseline_model.md for what each metric means.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def class_distribution(y: pd.Series) -> dict:
    """Count and percentage for each label, so the real balance is on record."""
    counts = y.value_counts()
    total = int(counts.sum())
    return {
        label: {
            "count": int(count),
            "percent": round(float(count) / total * 100, 2),
        }
        for label, count in counts.items()
    }


def positive_class_scores(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predicted probability of the positive (`>50K`) class.

    `predict_proba` returns one column per class in `pipeline.classes_` order,
    so the right column has to be looked up by label rather than assumed to
    be the second one.
    """
    classes = list(pipeline.classes_)
    if config.POSITIVE_LABEL not in classes:
        raise ValueError(
            f"Model does not know the positive label '{config.POSITIVE_LABEL}'. "
            f"It was trained on classes: {classes}"
        )
    positive_column = classes.index(config.POSITIVE_LABEL)
    return pipeline.predict_proba(X)[:, positive_column]


def confusion_matrix_frame(y_true: pd.Series, y_pred) -> pd.DataFrame:
    """Confusion matrix as a labelled table rather than a bare array.

    Rows are the true label, columns are what the model predicted, so the
    off-diagonal cells name the two kinds of mistake explicitly.
    """
    labels = config.TARGET_LABELS
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"actual {label}" for label in labels],
        columns=[f"predicted {label}" for label in labels],
    )


def classification_report_text(y_true: pd.Series, y_pred) -> str:
    """scikit-learn's per-class precision/recall/F1 table as text."""
    return classification_report(
        y_true, y_pred, labels=config.TARGET_LABELS, zero_division=0
    )


def evaluate_classifier(pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Score a fitted pipeline on labelled data and return every metric.

    All metrics for the positive class use `pos_label='>50K'` explicitly.
    That is the class worth detecting and the minority class, so leaving the
    default would silently report scores for the wrong class.
    """
    if len(X) == 0:
        raise ValueError("Cannot evaluate on an empty dataset.")
    if len(X) != len(y):
        raise ValueError(
            f"Features and labels have different lengths: {len(X)} vs {len(y)}"
        )

    y_pred = pipeline.predict(X)
    y_score = positive_class_scores(pipeline, X)
    positive = config.POSITIVE_LABEL

    return {
        "n_samples": int(len(X)),
        "class_distribution": class_distribution(y),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, pos_label=positive, zero_division=0)),
        "recall": float(recall_score(y, y_pred, pos_label=positive, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, pos_label=positive, zero_division=0)),
        # ROC-AUC needs 0/1 truth values plus a score, not the predicted labels.
        "roc_auc": float(roc_auc_score((y == positive).astype(int), y_score)),
        "confusion_matrix": confusion_matrix_frame(y, y_pred).values.tolist(),
        "confusion_matrix_labels": list(config.TARGET_LABELS),
        "classification_report": classification_report_text(y, y_pred),
    }


def evaluate_by_group(
    pipeline, X: pd.DataFrame, y: pd.Series, group_column: str
) -> pd.DataFrame:
    """Score the model separately within each value of `group_column`.

    A single overall number can hide very different behaviour between groups,
    so this breaks the same metrics down per subgroup. `selection_rate` is the
    share of each group predicted `>50K`; `recall` is the share of each
    group's genuine `>50K` cases that were found.

    This **measures** disparity. It does not correct it, and a small group
    will produce noisy numbers, so counts are reported alongside the rates.
    """
    if group_column not in X.columns:
        raise ValueError(
            f"Cannot group by '{group_column}': not a column in the data. "
            f"Available: {sorted(X.columns)}"
        )

    y_pred = pd.Series(pipeline.predict(X), index=X.index)
    positive = config.POSITIVE_LABEL
    rows = []

    for group in sorted(X[group_column].dropna().unique()):
        mask = X[group_column] == group
        group_true = y[mask]
        group_pred = y_pred[mask]
        actual_positive = group_true == positive

        rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "actual_positive_rate": round(float(actual_positive.mean()), 4),
                "selection_rate": round(float((group_pred == positive).mean()), 4),
                "accuracy": round(float(accuracy_score(group_true, group_pred)), 4),
                "precision": round(
                    float(precision_score(group_true, group_pred, pos_label=positive, zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(group_true, group_pred, pos_label=positive, zero_division=0)), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def fairness_summary(
    pipeline, X: pd.DataFrame, y: pd.Series, group_columns: list[str]
) -> dict:
    """Per-group metrics plus simple disparity ratios for each column.

    Two standard comparisons are reported:

    * `selection_rate_ratio` - the lowest group's rate of being predicted
      `>50K` divided by the highest. 1.0 would mean every group is selected
      at the same rate (sometimes called demographic parity).
    * `recall_ratio` - the lowest group's recall divided by the highest. 1.0
      would mean the model is equally good at finding genuine `>50K` cases in
      every group (sometimes called equal opportunity).

    Both are descriptive. A ratio below 1 does not by itself prove the model
    is unfair, because groups also differ in their true positive rates in
    this 1994 census data. It does show where the model behaves unevenly.
    """
    summary = {}
    for column in group_columns:
        table = evaluate_by_group(pipeline, X, y, column)
        # Ignore tiny groups when computing ratios: a handful of rows produces
        # a 0.0 or 1.0 rate that says more about the sample than the model.
        reliable = table[table["n"] >= 100]
        basis = reliable if len(reliable) >= 2 else table

        def ratio(name: str) -> float | None:
            highest = basis[name].max()
            if highest == 0:
                return None
            return round(float(basis[name].min() / highest), 4)

        summary[column] = {
            "groups": table.to_dict(orient="records"),
            "n_groups_compared": int(len(basis)),
            "min_group_size_for_ratios": 100,
            "selection_rate_ratio": ratio("selection_rate"),
            "recall_ratio": ratio("recall"),
        }
    return summary


def predict_with_threshold(pipeline, X: pd.DataFrame, threshold: float):
    """Label rows positive when their score reaches `threshold`.

    Useful for exploring a threshold without refitting. The saved model has
    its chosen threshold applied internally, so normal `predict` already
    honours it.
    """
    if not 0 < threshold < 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    negative = [l for l in config.TARGET_LABELS if l != config.POSITIVE_LABEL][0]
    scores = positive_class_scores(pipeline, X)
    return np.where(scores >= threshold, config.POSITIVE_LABEL, negative)


def majority_class_baseline(y: pd.Series) -> float:
    """Accuracy of always predicting the most common label.

    This is the number any real model has to beat. Quoting accuracy without
    it is how an imbalanced-data model gets mistaken for a good one.
    """
    return round(float(y.value_counts(normalize=True).max()), 4)


def format_metrics_report(metrics: dict, title: str = "Evaluation") -> str:
    """Readable summary for printing to a terminal or log."""
    lines = [
        f"{title} on {metrics['n_samples']:,} rows",
        f"  accuracy   {metrics['accuracy']:.4f}",
        f"  precision  {metrics['precision']:.4f}   (of those predicted >50K, how many were)",
        f"  recall     {metrics['recall']:.4f}   (of all real >50K, how many were found)",
        f"  f1_score   {metrics['f1_score']:.4f}",
        f"  roc_auc    {metrics['roc_auc']:.4f}",
        "",
        "Confusion matrix",
    ]
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"actual {label}" for label in metrics["confusion_matrix_labels"]],
        columns=[f"predicted {label}" for label in metrics["confusion_matrix_labels"]],
    )
    lines.append(matrix.to_string())
    lines.extend(["", "Per-class report", metrics["classification_report"]])
    return "\n".join(lines)
