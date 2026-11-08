"""Tests for cross-validation, threshold selection and the hyperparameter check.

The threshold tests use synthetic probability arrays rather than a trained
model, so the correct answer can be reasoned about directly instead of being
whatever the model happens to produce.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from monitor import config
from monitor.data import load_adult_data
from monitor.model import build_model_pipeline
from monitor.preprocessing import split_features_target
from monitor.tuning import (
    cross_validate_model,
    search_hyperparameters,
    select_threshold,
    threshold_sweep,
)

LOW = "<=50K"
HIGH = ">50K"


@pytest.fixture(scope="module")
def small():
    """A stratified 2,000-row sample, big enough to cross-validate quickly."""
    df = load_adult_data()
    sample, _ = train_test_split(
        df, train_size=2000, random_state=config.RANDOM_SEED,
        stratify=df[config.TARGET_COLUMN],
    )
    return split_features_target(sample)


# --- cross-validation -------------------------------------------------------


def test_cross_validation_reports_every_fold_and_a_summary(small):
    X, y = small
    cv = cross_validate_model(build_model_pipeline, X, y, n_splits=3)

    assert cv["n_splits"] == 3
    assert len(cv["per_fold"]) == 3
    assert set(cv["summary"]) == {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    for stats in cv["summary"].values():
        assert "mean" in stats and "std" in stats


def test_cross_validation_produces_one_out_of_fold_probability_per_row(small):
    X, y = small
    cv = cross_validate_model(build_model_pipeline, X, y, n_splits=3)
    probabilities = cv["out_of_fold_probabilities"]

    # Every row must be predicted exactly once, by a model that did not see
    # it during training. A leftover zero would mean a row was never covered.
    assert len(probabilities) == len(X)
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_cross_validation_summary_is_plausible(small):
    X, y = small
    cv = cross_validate_model(build_model_pipeline, X, y, n_splits=3)

    assert 0.7 < cv["summary"]["accuracy"]["mean"] < 1.0
    assert 0.5 < cv["summary"]["roc_auc"]["mean"] < 1.0
    assert cv["summary"]["accuracy"]["std"] < 0.1  # folds should broadly agree


def test_cross_validation_is_reproducible(small):
    X, y = small
    first = cross_validate_model(build_model_pipeline, X, y, n_splits=3)
    second = cross_validate_model(build_model_pipeline, X, y, n_splits=3)

    np.testing.assert_allclose(
        first["out_of_fold_probabilities"], second["out_of_fold_probabilities"]
    )
    assert first["summary"] == second["summary"]


def test_cross_validation_rejects_too_few_folds(small):
    X, y = small
    with pytest.raises(ValueError, match="at least 2 folds"):
        cross_validate_model(build_model_pipeline, X, y, n_splits=1)


# --- threshold selection ----------------------------------------------------


def make_separable(n: int = 200):
    """Scores that rank perfectly: positives high, negatives low."""
    y = pd.Series([LOW] * n + [HIGH] * n)
    scores = np.concatenate([np.linspace(0.05, 0.45, n), np.linspace(0.55, 0.95, n)])
    return y, scores


def test_threshold_sweep_covers_a_range_and_reports_each_metric():
    y, scores = make_separable()
    sweep = threshold_sweep(y, scores)

    assert set(sweep.columns) == {"threshold", "accuracy", "precision", "recall", "f1_score"}
    assert sweep["threshold"].min() < 0.1
    assert sweep["threshold"].max() > 0.9
    assert sweep["threshold"].is_monotonic_increasing


def test_recall_falls_as_the_threshold_rises():
    y, scores = make_separable()
    sweep = threshold_sweep(y, scores)

    low = sweep[sweep["threshold"] < 0.2]["recall"].mean()
    high = sweep[sweep["threshold"] > 0.8]["recall"].mean()
    assert low > high


def test_perfectly_separable_scores_give_a_perfect_threshold():
    y, scores = make_separable()
    selection = select_threshold(y, scores)

    assert selection["selected_scores"]["f1_score"] == pytest.approx(1.0)
    # Negatives top out at 0.45 and positives start at 0.55, so any threshold
    # in that gap separates them perfectly; the search must land inside it.
    assert 0.45 <= selection["selected_threshold"] <= 0.55


def test_selection_lowers_the_threshold_on_imbalanced_data():
    # 10% positives whose scores sit below 0.5: the default cut-off would find
    # none of them, so a useful selection must move below 0.5.
    y = pd.Series([LOW] * 900 + [HIGH] * 100)
    scores = np.concatenate([
        np.random.default_rng(0).uniform(0.0, 0.25, 900),
        np.random.default_rng(1).uniform(0.25, 0.49, 100),
    ])
    selection = select_threshold(y, scores)

    assert selection["selected_threshold"] < 0.5
    assert selection["selected_scores"]["recall"] > selection["default_threshold_scores"]["recall"]


def test_selection_reports_the_default_for_comparison():
    y, scores = make_separable()
    selection = select_threshold(y, scores)

    assert "default_threshold_scores" in selection
    assert set(selection["default_threshold_scores"]) == {
        "accuracy", "precision", "recall", "f1_score"
    }


def test_selection_can_optimise_recall_instead():
    y, scores = make_separable()
    by_recall = select_threshold(y, scores, objective="recall")

    assert by_recall["objective"] == "recall"
    assert by_recall["selected_scores"]["recall"] == pytest.approx(1.0)


def test_unknown_objective_is_rejected_with_the_valid_options():
    y, scores = make_separable()
    with pytest.raises(ValueError, match="Unknown objective"):
        select_threshold(y, scores, objective="vibes")


def test_precision_floor_is_respected():
    y, scores = make_separable()
    selection = select_threshold(y, scores, min_precision=0.95)

    assert selection["constraint_satisfied"]
    assert selection["selected_scores"]["precision"] >= 0.95


def test_impossible_precision_floor_is_reported_not_hidden():
    # Scores that cannot rank at all, so high precision is unreachable.
    y = pd.Series([LOW, HIGH] * 100)
    scores = np.full(200, 0.5)
    selection = select_threshold(y, scores, min_precision=0.99)

    assert selection["constraint_satisfied"] is False
    assert selection["min_precision"] == 0.99


# --- hyperparameter search --------------------------------------------------


def test_hyperparameter_search_compares_against_the_default(small):
    X, y = small
    result = search_hyperparameters(build_model_pipeline, X, y, n_splits=3)

    assert result["searched"] == "classify__C"
    assert len(result["results"]) == len(result["grid"])
    assert result["best_C"] in result["grid"]
    assert 0.5 < result["best_roc_auc"] < 1.0
    # The best score cannot be worse than the default's, by definition.
    assert result["improvement_over_default"] >= 0


def test_hyperparameter_search_rejects_a_model_it_has_no_grid_for(small):
    X, y = small
    with pytest.raises(ValueError, match="LogisticRegression"):
        search_hyperparameters(
            lambda: build_model_pipeline("random_forest"), X, y, n_splits=3
        )
