"""Build, train, and persist the baseline classifier.

The model is wrapped in a scikit-learn `Pipeline` whose first step is the
preprocessor from Phase 2 and whose second step is the classifier:

    Pipeline([("preprocess", build_preprocessor()), ("classify", ...)])

Bundling them has two benefits that matter for monitoring.

First, it removes any chance of data leakage. Calling `.fit(X_train, y_train)`
on the pipeline fits the preprocessor on the training features *only*, as part
of the same call, so there is no opportunity to accidentally fit it on the
full dataset before splitting.

Second, the saved artifact is self-contained. Loading one file gives something
that takes raw Adult-shaped rows and returns predictions, with the imputing,
scaling and encoding applied exactly as they were during training. A later
monitoring batch cannot be preprocessed differently by mistake.

Note on class balance: no `class_weight` is set and no resampling is done.
The project deliberately keeps the natural 76/24 split, so the baseline
reflects the distribution real incoming data actually has.
"""

import json
import platform
import warnings
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline

from . import config, schema
from .preprocessing import build_preprocessor, split_features_target, split_train_test
from .validation import validate_batch

BASELINE_MODEL = "logistic_regression"


def _logistic_regression() -> LogisticRegression:
    """The baseline. Simple, fast, and its coefficients are interpretable."""
    return LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)


def _random_forest() -> RandomForestClassifier:
    """Optional comparison model.

    Included for one reason only: to show whether the linear baseline is
    leaving obvious performance on the table. A tree ensemble can capture
    interactions that Logistic Regression cannot, so if the two score
    similarly then the simple model is a sound choice, and if the forest is
    far ahead that is worth knowing before settling on a baseline.
    """
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )


MODEL_BUILDERS = {
    BASELINE_MODEL: _logistic_regression,
    "random_forest": _random_forest,
}


def build_model_pipeline(model_name: str = BASELINE_MODEL) -> Pipeline:
    """Create an unfitted preprocessing + classifier pipeline."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {sorted(MODEL_BUILDERS)}"
        )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classify", MODEL_BUILDERS[model_name]()),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = BASELINE_MODEL,
    threshold: float | None = None,
) -> tuple[Pipeline, list[str]]:
    """Fit a pipeline on training data.

    When `threshold` is given the fitted model applies it inside `predict`,
    so the saved artifact and the reported metrics can never disagree about
    the operating point.

    Returns the fitted model plus a list of any warnings worth surfacing, so
    a convergence problem is reported rather than scrolling past unnoticed.
    """
    result = validate_batch(X_train)
    result.raise_if_invalid()

    observed_classes = sorted(y_train.dropna().unique())
    if len(observed_classes) < 2:
        raise ValueError(
            "Training data must contain both income classes, but only found "
            f"{observed_classes}. A classifier cannot be trained on one class."
        )

    model = build_model_pipeline(model_name)
    if threshold is not None:
        model = apply_threshold(model, threshold)

    messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            model.fit(X_train, y_train)
        except Exception as error:
            raise RuntimeError(
                f"Training '{model_name}' failed: {type(error).__name__}: {error}"
            ) from error

    classifier = unwrap_pipeline(model).named_steps["classify"]
    max_iter = getattr(classifier, "max_iter", None)
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            limit = f" in {max_iter} iterations" if max_iter else ""
            messages.append(
                f"{model_name} did not fully converge{limit}; "
                "coefficients may be unstable."
            )
    return model, messages


def apply_threshold(pipeline: Pipeline, threshold: float) -> FixedThresholdClassifier:
    """Wrap a pipeline so that `predict` uses `threshold` instead of 0.5.

    Returned object is unfitted; fit it on the training data as usual. The
    wrapper is used rather than storing the threshold as a loose number so
    that the saved model's own `predict` always agrees with the metrics that
    were reported for it. A threshold kept only in documentation is a trap:
    somebody eventually calls `predict` and silently gets different answers.
    """
    if not 0 < threshold < 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    return FixedThresholdClassifier(
        pipeline, threshold=threshold, pos_label=config.POSITIVE_LABEL
    )


def unwrap_pipeline(model) -> Pipeline:
    """Return the inner `Pipeline`, whether or not a threshold wrapper is present."""
    if isinstance(model, FixedThresholdClassifier):
        # `estimator_` is the fitted inner pipeline; `estimator` the unfitted one.
        return getattr(model, "estimator_", model.estimator)
    return model


def get_threshold(model) -> float:
    """The decision threshold a model actually uses."""
    if isinstance(model, FixedThresholdClassifier):
        return float(model.threshold)
    return 0.5


def extract_preprocessor(model) -> ColumnTransformer:
    """Pull the fitted preprocessor out of a trained model.

    Later monitoring phases need transformed features to compare batches
    against the reference period. Taking the preprocessor from inside the
    model keeps a single source of truth, instead of relying on a separately
    fitted copy that might have been fitted on different rows.
    """
    return unwrap_pipeline(model).named_steps["preprocess"]


def build_metadata(
    pipeline: Pipeline,
    model_name: str,
    train_metrics: dict,
    test_metrics: dict,
    n_rows_total: int,
    cv_summary: dict | None = None,
    threshold_selection: dict | None = None,
    hyperparameter_search: dict | None = None,
    fairness: dict | None = None,
) -> dict:
    """Describe exactly what was trained, on what, and how well it scored.

    Saved alongside the model so a stored artifact is never a mystery: the
    seed, the feature names and the real measured scores all travel with it.
    """
    return {
        "model_type": type(unwrap_pipeline(pipeline).named_steps["classify"]).__name__,
        "model_name": model_name,
        "trained_on": date.today().isoformat(),
        "random_seed": config.RANDOM_SEED,
        "decision_threshold": get_threshold(pipeline),
        "threshold_selection": threshold_selection,
        "cross_validation": cv_summary,
        "hyperparameter_search": hyperparameter_search,
        "fairness": fairness,
        "target_column": config.TARGET_COLUMN,
        "positive_label": config.POSITIVE_LABEL,
        "target_labels": list(config.TARGET_LABELS),
        "input_feature_names": list(schema.FEATURE_COLUMNS),
        "numeric_features": list(schema.NUMERIC_COLUMNS),
        "categorical_features": list(schema.CATEGORICAL_COLUMNS),
        "encoded_feature_names": list(
            extract_preprocessor(pipeline).get_feature_names_out()
        ),
        "dataset": {
            "name": "Adult (Census Income)",
            "source": config.UCI_DATASET_URL,
            "uci_id": config.UCI_DATASET_ID,
            "n_rows_total": int(n_rows_total),
            "test_size": config.TEST_SIZE,
            "split": "stratified on the target, natural class balance preserved",
        },
        "class_balance_handling": (
            "none - no class_weight and no resampling, so the natural "
            "76/24 distribution is preserved"
        ),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "train_metrics": {
            key: value
            for key, value in train_metrics.items()
            if key != "classification_report"
        },
        "test_metrics": {
            key: value
            for key, value in test_metrics.items()
            if key != "classification_report"
        },
    }


def save_model(pipeline: Pipeline, path: Path = config.BASELINE_MODEL_PATH) -> None:
    """Save the complete fitted pipeline (preprocessing included)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path = config.BASELINE_MODEL_PATH) -> Pipeline:
    """Load a saved pipeline, with a message that says how to create one."""
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python train_baseline.py` first."
        )
    return joblib.load(path)


def save_json(payload: dict, path: Path) -> None:
    """Write a dictionary to a JSON file, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metadata(path: Path = config.BASELINE_METADATA_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No model metadata at {path}. Run `python train_baseline.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def train_baseline(
    df: pd.DataFrame,
    model_name: str = BASELINE_MODEL,
    threshold: float | None = None,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, list[str]]:
    """Split, then train on the training half only.

    Returns the fitted model and both splits, so the caller can evaluate
    without re-splitting and risking a different partition.
    """
    train_df, test_df = split_train_test(df, test_size=config.TEST_SIZE)
    X_train, y_train = split_features_target(train_df)
    model, messages = train_model(
        X_train, y_train, model_name=model_name, threshold=threshold
    )
    return model, train_df, test_df, messages
