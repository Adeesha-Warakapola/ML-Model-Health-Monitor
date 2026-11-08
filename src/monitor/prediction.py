"""Score incoming batches with the saved model.

The service loads the trained pipeline once and reuses it for every batch.
Because the saved artifact bundles the preprocessor, the classifier and the
decision threshold together, inference automatically applies exactly the same
imputing, scaling and encoding that were fitted during training. There is no
separate preprocessing step here that could drift out of step with the model.

One rule is enforced rather than merely documented: **the service never sees
the true labels.** `predict_batch` refuses a frame that still contains the
income column. At prediction time in a real system that column would not
exist yet - labels arrive later, if at all - so letting it through here would
make the simulation dishonest and would risk a future change accidentally
using it.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import config
from .model import get_threshold, load_model, unwrap_pipeline
from .validation import ValidationResult, validate_batch


def make_batch_id(prefix: str = "batch") -> str:
    """A unique, sortable identifier such as `batch_20261108T211530Z_4f3a9c21`.

    The timestamp makes IDs sort chronologically and stay readable; the random
    suffix keeps two batches created in the same second distinct.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def utc_now() -> str:
    """Current time as an ISO-8601 string, so it survives JSON round-trips."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class BatchPrediction:
    """Everything produced for one scored batch.

    The original feature values are kept alongside the predictions on purpose.
    Later monitoring phases need both: the features to detect input drift, and
    the predictions to detect a shift in what the model is saying.
    """

    batch_id: str
    predicted_at: str
    model_type: str
    threshold: float
    features: pd.DataFrame
    predictions: pd.Series
    probabilities: pd.Series | None
    validation: ValidationResult
    warnings: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.features)

    def to_frame(self) -> pd.DataFrame:
        """Features plus predictions in one table, ready to store or inspect."""
        frame = self.features.copy()
        frame["prediction"] = self.predictions
        if self.probabilities is not None:
            frame[f"probability_{config.POSITIVE_LABEL}"] = self.probabilities
        frame["batch_id"] = self.batch_id
        frame["predicted_at"] = self.predicted_at
        return frame

    def prediction_distribution(self) -> dict:
        """How many of each label the model predicted, with percentages.

        This is a prediction distribution, not a class distribution: it says
        what the model *claimed*, which is not the same as what was true.
        """
        counts = self.predictions.value_counts()
        total = int(counts.sum())
        return {
            label: {
                "count": int(count),
                "percent": round(float(count) / total * 100, 2),
            }
            for label, count in counts.items()
        }

    def summary(self) -> str:
        lines = [
            f"Batch {self.batch_id}",
            f"  scored at      {self.predicted_at}",
            f"  model          {self.model_type} at threshold {self.threshold}",
            f"  rows           {self.n_rows}",
            f"  predicted      " + ", ".join(
                f"{label} {stats['count']} ({stats['percent']}%)"
                for label, stats in self.prediction_distribution().items()
            ),
        ]
        if self.probabilities is not None:
            lines.append(
                f"  mean P({config.POSITIVE_LABEL})   "
                f"{self.probabilities.mean():.4f}"
            )
        issues = len(self.validation.issues)
        lines.append(
            f"  validation     {'passed' if self.validation.is_valid else 'FAILED'}"
            f" ({len(self.validation.errors)} errors, "
            f"{len(self.validation.warnings)} warnings)"
        )
        for warning in self.warnings:
            lines.append(f"  WARNING        {warning}")
        return "\n".join(lines)


class PredictionService:
    """Loads the trained model once and scores batches with it."""

    def __init__(self, model=None, model_path=None):
        """Load the saved model, or accept one directly (useful in tests).

        A missing model file raises a `FileNotFoundError` explaining how to
        create one, rather than failing later with something cryptic.
        """
        if model is None:
            self.model = load_model(model_path or config.BASELINE_MODEL_PATH)
        else:
            self.model = model

        self.threshold = get_threshold(self.model)
        self.model_type = type(unwrap_pipeline(self.model).named_steps["classify"]).__name__
        # Not every classifier can produce probabilities, so this is checked
        # once here instead of being assumed at every call.
        self.supports_probabilities = hasattr(self.model, "predict_proba")

    def _positive_column(self) -> int:
        classes = list(self.model.classes_)
        if config.POSITIVE_LABEL not in classes:
            raise ValueError(
                f"Model does not know the positive label "
                f"'{config.POSITIVE_LABEL}'; it knows {classes}."
            )
        return classes.index(config.POSITIVE_LABEL)

    def predict_batch(
        self, features: pd.DataFrame, batch_id: str | None = None
    ) -> BatchPrediction:
        """Validate and score one batch of feature rows.

        `features` must not contain the income column: labels are not
        available at prediction time and are deliberately kept out of reach.
        """
        if config.TARGET_COLUMN in features.columns:
            raise ValueError(
                f"Batch contains the label column '{config.TARGET_COLUMN}'. "
                "The prediction service must never receive true labels - pass "
                "only feature columns and keep labels separate."
            )
        if features.empty:
            raise ValueError("Cannot predict on an empty batch.")

        validation = validate_batch(features)
        validation.raise_if_invalid()

        try:
            predictions = self.model.predict(features)
        except Exception as error:
            raise RuntimeError(
                f"Prediction failed on batch of {len(features)} rows: "
                f"{type(error).__name__}: {error}"
            ) from error

        probabilities = None
        warnings = []
        if self.supports_probabilities:
            try:
                scores = self.model.predict_proba(features)[:, self._positive_column()]
                probabilities = pd.Series(scores, index=features.index, name="probability")
            except Exception as error:
                warnings.append(
                    f"Probabilities unavailable: {type(error).__name__}: {error}"
                )
        else:
            warnings.append(
                f"{self.model_type} does not support predict_proba; "
                "probabilities omitted."
            )

        return BatchPrediction(
            batch_id=batch_id or make_batch_id(),
            predicted_at=utc_now(),
            model_type=self.model_type,
            threshold=self.threshold,
            features=features.copy(),
            predictions=pd.Series(predictions, index=features.index, name="prediction"),
            probabilities=probabilities,
            validation=validation,
            warnings=warnings,
        )

    def predict_simulated(self, batch) -> BatchPrediction:
        """Score a `SimulatedBatch`, passing only its features through.

        The batch's labels stay with the batch and are never handed over, so
        they remain available for scoring performance later without ever
        having influenced the prediction.
        """
        return self.predict_batch(batch.features, batch_id=batch.batch_id)
