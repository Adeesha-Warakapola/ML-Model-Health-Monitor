"""Monitor what the model is predicting, as distinct from what is true.

`drift.py` asks "has the input changed?" This module asks a related but
different question: **has what the model says changed?** The two usually move
together, but not always - a model can start predicting more `>50K` either
because more people in the batch genuinely look like `>50K` cases (input
drift, which `drift.py` would also flag), or because the model's own decision
surface is behaving differently on inputs that look unchanged (which `drift.py`
would miss entirely, since it never looks at predictions).

Neither this module nor `drift.py` can say whether the model got *worse* -
that needs true labels and is `performance_monitoring.py`'s job. This module
only describes what the model claimed.

The reference prediction distribution is the model's own predictions on the
training data (Phase 4's `get_reference_features`), scored once through the
same `PredictionService` used for every other batch. That keeps this phase
honest: the "normal" distribution is not invented, it is measured from data
already proven to be in scope.

Reuses the PSI machinery from `drift.py` rather than re-implementing it: the
predicted label is just another categorical column, and the predicted
probability is just another numeric column, so the same zero-inflation-safe
binning and rare-category pooling that were built and measured for input
features apply unchanged here.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from . import config
from .drift import (
    DriftThresholds,
    _describe_significance,
    _status_from_psi,
    categorical_psi,
    cramers_v,
    frequent_categories,
    population_stability_index,
    quantile_bin_edges,
)
from .prediction import BatchPrediction, PredictionService

STABLE = "stable"
MODERATE = "moderate_drift"
DRIFT = "drift"
NOT_COMPUTED = "not_computed"

# Margin either side of the decision threshold counted as "borderline": a
# prediction the model was not confident about either way.
DEFAULT_CONFIDENCE_MARGIN = 0.10


def confidence_summary(
    probabilities: pd.Series, threshold: float, margin: float = DEFAULT_CONFIDENCE_MARGIN
) -> dict:
    """Describe how confident a set of predictions was.

    `borderline_rate` is the share of predictions within `margin` of the
    decision threshold - cases where a small change in the model or the input
    would have flipped the label. A rising borderline rate is an early signal
    worth watching even before the prediction mix itself visibly shifts.
    """
    values = probabilities.dropna()
    if values.empty:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "q25": None, "q75": None, "borderline_rate": None,
        }
    return {
        "n": int(len(values)),
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "median": round(float(values.median()), 4),
        "q25": round(float(values.quantile(0.25)), 4),
        "q75": round(float(values.quantile(0.75)), 4),
        "borderline_rate": round(
            float(((values >= threshold - margin) & (values <= threshold + margin)).mean()), 4
        ),
    }


@dataclass
class PredictionDriftReport:
    """What the model predicted on a batch, compared with its reference behaviour."""

    batch_id: str | None
    evaluated_at: str
    n_reference: int
    n_batch: int
    threshold: float

    reference_distribution: dict
    batch_distribution: dict
    label_status: str
    label_psi: float
    label_statistic: float | None
    label_p_value: float | None
    label_effect_size: float | None

    reference_confidence: dict
    batch_confidence: dict
    confidence_status: str
    confidence_psi: float | None
    confidence_statistic: float | None
    confidence_p_value: float | None

    thresholds: DriftThresholds
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Worse of the label-mix and confidence verdicts.

        One feature's worth of severity logic, reused: this mirrors
        `DriftReport.status` in spirit (the loudest signal decides) using the
        same DRIFT > MODERATE > STABLE ordering.
        """
        order = {STABLE: 0, NOT_COMPUTED: 0, MODERATE: 1, DRIFT: 2}
        worst = max(self.label_status, self.confidence_status, key=lambda s: order[s])
        return worst

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "evaluated_at": self.evaluated_at,
            "status": self.status,
            "n_reference": self.n_reference,
            "n_batch": self.n_batch,
            "threshold": self.threshold,
            "label": {
                "reference_distribution": self.reference_distribution,
                "batch_distribution": self.batch_distribution,
                "status": self.label_status,
                "psi": self.label_psi,
                "chi2_statistic": self.label_statistic,
                "p_value": self.label_p_value,
                "cramers_v": self.label_effect_size,
            },
            "confidence": {
                "reference": self.reference_confidence,
                "batch": self.batch_confidence,
                "status": self.confidence_status,
                "psi": self.confidence_psi,
                "ks_statistic": self.confidence_statistic,
                "p_value": self.confidence_p_value,
            },
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lines = [
            f"Prediction monitoring for batch {self.batch_id or '(unnamed)'}",
            f"  evaluated at   {self.evaluated_at}",
            f"  verdict        {self.status.upper()}",
            "",
            "  Predicted-label mix:",
        ]
        for label in sorted(
            set(self.reference_distribution) | set(self.batch_distribution)
        ):
            ref = self.reference_distribution.get(label, {"count": 0, "percent": 0.0})
            cur = self.batch_distribution.get(label, {"count": 0, "percent": 0.0})
            lines.append(
                f"    {label:8s} reference {ref['percent']:6.2f}%   "
                f"batch {cur['percent']:6.2f}%"
            )
        lines.append(
            f"    status={self.label_status}  psi={self.label_psi:.4f}  "
            f"p={self.label_p_value if self.label_p_value is None else f'{self.label_p_value:.2e}'}"
        )
        if self.batch_confidence["n"]:
            lines.extend(
                [
                    "",
                    "  Confidence (predicted probability of the positive class):",
                    f"    reference mean={self.reference_confidence['mean']}  "
                    f"std={self.reference_confidence['std']}  "
                    f"borderline_rate={self.reference_confidence['borderline_rate']}",
                    f"    batch     mean={self.batch_confidence['mean']}  "
                    f"std={self.batch_confidence['std']}  "
                    f"borderline_rate={self.batch_confidence['borderline_rate']}",
                    f"    status={self.confidence_status}  "
                    f"psi={self.confidence_psi}",
                ]
            )
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


class PredictionMonitor:
    """Compares a batch's predictions against the model's reference behaviour."""

    def __init__(
        self,
        reference_predictions: pd.Series,
        reference_probabilities: pd.Series | None,
        threshold: float,
        thresholds: DriftThresholds | None = None,
    ):
        if reference_predictions.empty:
            raise ValueError("Reference predictions are empty.")
        self.reference_predictions = reference_predictions.astype(str)
        self.reference_probabilities = reference_probabilities
        self.threshold = threshold
        self.thresholds = thresholds or DriftThresholds()
        self._frequent_labels = frequent_categories(
            self.reference_predictions, self.thresholds.min_category_share
        )
        self._reference_confidence = (
            confidence_summary(reference_probabilities, threshold)
            if reference_probabilities is not None
            else confidence_summary(pd.Series([], dtype=float), threshold)
        )
        self._bin_edges = (
            quantile_bin_edges(reference_probabilities, self.thresholds.n_bins)
            if reference_probabilities is not None
            else None
        )

    @classmethod
    def from_reference_features(
        cls,
        service: PredictionService,
        reference_features: pd.DataFrame,
        thresholds: DriftThresholds | None = None,
    ) -> "PredictionMonitor":
        """Build the reference distribution by scoring the training features.

        This is the one call that does the "original reference prediction
        distribution" work: the model predicts on the data it was trained on,
        and whatever it says there becomes the yardstick for every later
        batch. Reuses `PredictionService` exactly as every other batch does,
        so the reference is produced by the identical code path.
        """
        result = service.predict_batch(reference_features, batch_id="reference")
        return cls(
            result.predictions, result.probabilities, result.threshold, thresholds
        )

    def _label_drift(self, batch_predictions: pd.Series) -> dict:
        batch_predictions = batch_predictions.astype(str)
        psi = round(
            categorical_psi(
                self.reference_predictions,
                batch_predictions,
                self.thresholds.psi_epsilon,
                keep=self._frequent_labels,
            ),
            4,
        )
        status = _status_from_psi(psi, self.thresholds)

        categories = sorted(
            set(self.reference_predictions.unique()) | set(batch_predictions.unique())
        )
        table = np.vstack(
            [
                self.reference_predictions.value_counts().reindex(categories, fill_value=0).to_numpy(),
                batch_predictions.value_counts().reindex(categories, fill_value=0).to_numpy(),
            ]
        )
        table = table[:, table.sum(axis=0) > 0]

        statistic = p_value = effect_size = None
        notes = []
        if table.shape[1] >= 2:
            try:
                chi2, p_value, _, _ = chi2_contingency(table)
                statistic = float(chi2)
                p_value = float(p_value)
                effect_size = cramers_v(chi2, int(table.sum()), table.shape)
                notes.extend(
                    _describe_significance(
                        p_value, psi, self.thresholds, len(self.reference_predictions)
                    )
                )
            except ValueError as error:
                notes.append(f"Chi-square test could not be computed: {error}")
        else:
            notes.append(
                "Only one predicted label is present across reference and batch; "
                "a chi-square test is undefined. PSI is still reported."
            )

        return {
            "status": status, "psi": psi, "statistic": statistic,
            "p_value": p_value, "effect_size": effect_size, "notes": notes,
        }

    def _confidence_drift(self, batch_probabilities: pd.Series | None) -> dict:
        if self.reference_probabilities is None or batch_probabilities is None:
            return {
                "status": NOT_COMPUTED, "psi": None, "statistic": None,
                "p_value": None,
                "notes": ["Probabilities are unavailable, so confidence drift "
                          "cannot be assessed; only the predicted-label mix is."],
            }
        values = batch_probabilities.dropna()
        if values.empty or self._bin_edges is None:
            return {
                "status": NOT_COMPUTED, "psi": None, "statistic": None,
                "p_value": None,
                "notes": ["No usable probability values in this batch."],
            }

        reference_values = self.reference_probabilities.dropna()
        ks = ks_2samp(reference_values, values)
        psi = round(
            population_stability_index(
                reference_values, values, self._bin_edges, self.thresholds.psi_epsilon
            ),
            4,
        )
        status = _status_from_psi(psi, self.thresholds)
        notes = _describe_significance(
            float(ks.pvalue), psi, self.thresholds, len(reference_values)
        )
        return {
            "status": status, "psi": psi, "statistic": float(ks.statistic),
            "p_value": float(ks.pvalue), "notes": notes,
        }

    def evaluate(self, batch_prediction: BatchPrediction) -> PredictionDriftReport:
        """Compare one scored batch against the reference prediction behaviour."""
        if batch_prediction.n_rows == 0:
            raise ValueError("Cannot monitor predictions on an empty batch.")

        label_result = self._label_drift(batch_prediction.predictions)
        confidence_result = self._confidence_drift(batch_prediction.probabilities)
        batch_confidence = (
            confidence_summary(batch_prediction.probabilities, self.threshold)
            if batch_prediction.probabilities is not None
            else confidence_summary(pd.Series([], dtype=float), self.threshold)
        )

        notes = list(label_result["notes"]) + list(confidence_result["notes"])
        notes.append(
            "A shift in predicted-class proportions does not by itself mean the "
            "model has gotten worse. It can also reflect a genuine change in "
            "the incoming population - compare against the data-drift report "
            "for the same batch - or simply be sampling noise in a small "
            "batch. Confirming actual performance change needs true labels; "
            "see performance_monitoring.py."
        )
        if batch_prediction.n_rows < self.thresholds.min_batch_rows:
            notes.append(
                f"SMALL SAMPLE: this batch has {batch_prediction.n_rows} rows, "
                f"below the {self.thresholds.min_batch_rows}-row floor measured "
                "for reliable PSI (see drift.py). Treat any verdict here with "
                "the same caution as a small input-drift batch."
            )

        return PredictionDriftReport(
            batch_id=batch_prediction.batch_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_reference=len(self.reference_predictions),
            n_batch=batch_prediction.n_rows,
            threshold=self.threshold,
            reference_distribution=self._reference_distribution(),
            batch_distribution=batch_prediction.prediction_distribution(),
            label_status=label_result["status"],
            label_psi=label_result["psi"],
            label_statistic=label_result["statistic"],
            label_p_value=label_result["p_value"],
            label_effect_size=label_result["effect_size"],
            reference_confidence=self._reference_confidence,
            batch_confidence=batch_confidence,
            confidence_status=confidence_result["status"],
            confidence_psi=confidence_result["psi"],
            confidence_statistic=confidence_result["statistic"],
            confidence_p_value=confidence_result["p_value"],
            thresholds=self.thresholds,
            notes=notes,
        )

    def _reference_distribution(self) -> dict:
        counts = self.reference_predictions.value_counts()
        total = int(counts.sum())
        return {
            label: {"count": int(count), "percent": round(float(count) / total * 100, 2)}
            for label, count in counts.items()
        }
