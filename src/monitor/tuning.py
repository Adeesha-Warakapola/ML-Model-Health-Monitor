"""Cross-validation, decision-threshold selection, and a hyperparameter check.

These exist to answer three questions that a single train/test split cannot:

1. **How much do the scores wobble?** One split gives one number with no sense
   of its uncertainty. Cross-validation gives a mean and a spread, so a later
   change can be judged as real improvement or just noise.
2. **Is 0.5 the right cut-off?** A classifier outputs a probability; turning
   that into a label needs a threshold. The default 0.5 is an arbitrary
   inheritance, and on imbalanced data it is usually too conservative about
   the minority class.
3. **Are the default hyperparameters reasonable?** Worth confirming rather
   than assuming.

Everything here runs on **training data only**. The threshold is chosen from
out-of-fold predictions inside the training split, never from the test set -
picking a threshold by looking at test scores would make the test metrics
optimistic and meaningless.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config

N_SPLITS = 5


def _positive_index(classes) -> int:
    classes = list(classes)
    if config.POSITIVE_LABEL not in classes:
        raise ValueError(
            f"Positive label '{config.POSITIVE_LABEL}' not among classes {classes}"
        )
    return classes.index(config.POSITIVE_LABEL)


def _scores_at(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    """Metrics obtained by labelling a row positive when its score >= threshold."""
    positive = config.POSITIVE_LABEL
    negative = [label for label in config.TARGET_LABELS if label != positive][0]
    y_pred = np.where(y_prob >= threshold, positive, negative)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=positive, zero_division=0)),
    }


def cross_validate_model(
    pipeline_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = config.RANDOM_SEED,
) -> dict:
    """Run stratified k-fold cross-validation on the training data.

    `pipeline_factory` is a zero-argument callable returning a fresh unfitted
    pipeline. A new one is built per fold so that no fold ever reuses
    parameters learned on another fold.

    Returns the per-fold metrics, their mean and standard deviation, and the
    pooled out-of-fold probabilities, which the threshold search then reuses
    rather than refitting anything.
    """
    if n_splits < 2:
        raise ValueError(f"Cross-validation needs at least 2 folds, got {n_splits}")

    folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    oof_probabilities = np.zeros(len(X), dtype=float)
    per_fold = []

    for fold_number, (train_index, validation_index) in enumerate(
        folds.split(X, y), start=1
    ):
        X_fold_train = X.iloc[train_index]
        y_fold_train = y.iloc[train_index]
        X_validation = X.iloc[validation_index]
        y_validation = y.iloc[validation_index]

        pipeline = pipeline_factory()
        pipeline.fit(X_fold_train, y_fold_train)

        positive_column = _positive_index(pipeline.classes_)
        probabilities = pipeline.predict_proba(X_validation)[:, positive_column]
        oof_probabilities[validation_index] = probabilities

        metrics = _scores_at(y_validation, probabilities, 0.5)
        metrics["roc_auc"] = float(
            roc_auc_score((y_validation == config.POSITIVE_LABEL).astype(int), probabilities)
        )
        metrics["fold"] = fold_number
        per_fold.append(metrics)

    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    summary = {
        name: {
            "mean": round(float(np.mean([f[name] for f in per_fold])), 4),
            "std": round(float(np.std([f[name] for f in per_fold])), 4),
        }
        for name in metric_names
    }

    return {
        "n_splits": n_splits,
        "per_fold": per_fold,
        "summary": summary,
        "out_of_fold_probabilities": oof_probabilities,
    }


def threshold_sweep(
    y_true: pd.Series, y_prob: np.ndarray, step: float = 0.01
) -> pd.DataFrame:
    """Metrics at every candidate threshold, for inspecting the trade-off."""
    thresholds = np.arange(0.05, 0.96, step)
    return pd.DataFrame([_scores_at(y_true, y_prob, t) for t in thresholds])


def select_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    objective: str = "f1_score",
    min_precision: float | None = None,
) -> dict:
    """Choose the decision threshold that maximises `objective`.

    `min_precision` optionally constrains the search, for the case where a
    floor on precision matters more than the best possible F1. If no
    threshold satisfies the constraint it is reported rather than silently
    ignored.

    Must be given out-of-fold probabilities from the training split. Passing
    test-set probabilities here would leak the test set into model selection.
    """
    sweep = threshold_sweep(y_true, y_prob)
    if objective not in sweep.columns:
        raise ValueError(
            f"Unknown objective '{objective}'. Choose from: "
            f"{[c for c in sweep.columns if c != 'threshold']}"
        )

    candidates = sweep
    constraint_met = True
    if min_precision is not None:
        candidates = sweep[sweep["precision"] >= min_precision]
        if candidates.empty:
            constraint_met = False
            candidates = sweep

    best = candidates.loc[candidates[objective].idxmax()]
    default = sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]

    return {
        "objective": objective,
        "min_precision": min_precision,
        "constraint_satisfied": constraint_met,
        "selected_threshold": float(best["threshold"]),
        "selected_scores": {
            name: round(float(best[name]), 4)
            for name in ("accuracy", "precision", "recall", "f1_score")
        },
        "default_threshold_scores": {
            name: round(float(default[name]), 4)
            for name in ("accuracy", "precision", "recall", "f1_score")
        },
    }


def search_hyperparameters(
    pipeline_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = config.RANDOM_SEED,
) -> dict:
    """Check whether Logistic Regression's default strength is sensible.

    Only `C` is searched, the inverse regularisation strength. The grid is
    deliberately small: the aim is to confirm the default is not a poor
    choice, not to squeeze out the last fraction of a percent.

    Scored on ROC-AUC because it does not depend on the decision threshold,
    which is tuned separately afterwards.
    """
    probe = pipeline_factory().named_steps["classify"]
    if not isinstance(probe, LogisticRegression):
        raise ValueError(
            "The hyperparameter grid is defined for LogisticRegression, "
            f"but the pipeline uses {type(probe).__name__}."
        )

    grid = {"classify__C": [0.01, 0.1, 1.0, 10.0]}
    search = GridSearchCV(
        pipeline_factory(),
        param_grid=grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        n_jobs=-1,
    )
    search.fit(X, y)

    results = [
        {
            "C": params["classify__C"],
            "mean_roc_auc": round(float(mean), 4),
            "std_roc_auc": round(float(std), 4),
        }
        for params, mean, std in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        )
    ]
    default_score = next(row["mean_roc_auc"] for row in results if row["C"] == 1.0)

    return {
        "searched": "classify__C",
        "grid": grid["classify__C"],
        "results": results,
        "best_C": search.best_params_["classify__C"],
        "best_roc_auc": round(float(search.best_score_), 4),
        "default_C_roc_auc": default_score,
        "improvement_over_default": round(float(search.best_score_) - default_score, 4),
    }
