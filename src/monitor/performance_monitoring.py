"""Measure whether the model is still accurate, once true labels arrive.

Everything in `drift.py` and `prediction_monitoring.py` works without a
single true label: they compare distributions. This module answers the
question those two cannot - *is the model still right?* - and it can only do
that once ground truth shows up, which in a real system happens **later**,
**separately**, and **for only some rows**.

Three rules make that honest rather than merely convenient:

1. **Predictions are recorded before any label exists.** `register_batch`
   stores what the model said, with no `actual_label` field filled in. There
   is no code path that could backfill an actual label from a prediction,
   because the field simply is not set until `submit_labels` is called with
   labels that came from somewhere else.
2. **Metrics are never invented when there are no labels.** `compute_performance`
   with zero labelled rows returns `metrics=None`, not zeros - a model that
   has not yet been checked is reported as unchecked.
3. **Labels can arrive incrementally, in any number of calls, without
   duplicating anything.** Records are keyed by a stable identifier
   (`batch_id:original_row_index`), so submitting labels for the same record
   twice updates it in place rather than creating a second entry, and calling
   `compute_performance` again after more labels arrive simply recomputes
   from the current state - there is no stored history to reconcile.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .evaluation import evaluate_predictions, format_metrics_report
from .prediction import BatchPrediction, utc_now

NO_LABELS = "no_labels"
INSUFFICIENT = "insufficient_labels"
COMPUTED = "computed"

# Measured, not guessed - see docs/performance_monitoring.md. Repeated random
# samples from the real holdout set and the model's own predictions, at
# n=300, gave a standard deviation for recall of about 0.049 (recall is
# computed over only the ~24% minority-class rows, so its effective sample
# size is much smaller than n suggests). Below this, a reported recall can
# easily differ from reality by more than the usual degradation margin, by
# chance alone.
DEFAULT_MIN_LABELED_SAMPLE = 300

# A drop of this much, in absolute terms, is reported as degraded. Chosen to
# sit above the sampling noise measured at DEFAULT_MIN_LABELED_SAMPLE
# (std ~0.049 for recall, much smaller for the other metrics), so a flag at
# the minimum sample size is more likely to reflect a real change than noise -
# though at exactly that floor it is not a large margin, which is exactly why
# the report always states the labelled count next to the verdict.
DEFAULT_DEGRADATION_MARGIN = 0.05


def make_record_id(batch_id: str, row_index) -> str:
    """A stable key identifying one prediction: which batch, which row.

    Stable because it depends only on facts fixed at prediction time - it
    does not change if a label is added, changed, or never arrives.
    """
    return f"{batch_id}:{row_index}"


@dataclass
class PredictionRecord:
    """One row's prediction, with a label slot that starts empty."""

    record_id: str
    batch_id: str
    row_index: str
    predicted_label: str
    probability: float | None
    predicted_at: str
    actual_label: str | None = None
    labeled_at: str | None = None

    @property
    def is_labeled(self) -> bool:
        return self.actual_label is not None


@dataclass
class LabelSubmissionResult:
    """What happened when a batch of actual labels was submitted."""

    batch_id: str
    n_submitted: int
    n_matched: int
    n_relabeled: int
    unmatched_record_ids: list[str]

    @property
    def n_unmatched(self) -> int:
        return len(self.unmatched_record_ids)


@dataclass
class PerformanceReport:
    """Result of trying to compute performance for some slice of records."""

    status: str
    evaluated_at: str
    batch_id: str | None
    n_total: int
    n_labeled: int
    min_labeled_sample: int
    metrics: dict | None = None
    baseline: dict | None = None
    comparison: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_unlabeled(self) -> int:
        return self.n_total - self.n_labeled

    @property
    def labeled_percent(self) -> float:
        return round(self.n_labeled / self.n_total * 100, 2) if self.n_total else 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "evaluated_at": self.evaluated_at,
            "batch_id": self.batch_id,
            "n_total": self.n_total,
            "n_labeled": self.n_labeled,
            "n_unlabeled": self.n_unlabeled,
            "labeled_percent": self.labeled_percent,
            "min_labeled_sample": self.min_labeled_sample,
            "metrics": self.metrics,
            "baseline": self.baseline,
            "comparison": self.comparison,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lines = [
            f"Performance report for {self.batch_id or 'all registered batches'}",
            f"  evaluated at    {self.evaluated_at}",
            f"  status          {self.status.upper()}",
            f"  labelled        {self.n_labeled} / {self.n_total} rows "
            f"({self.labeled_percent}%)",
        ]
        if self.metrics is not None:
            lines.append("")
            lines.append(format_metrics_report(self.metrics, title="Current"))
        if self.comparison is not None:
            lines.append("")
            lines.append("  Compared with the Phase 3 baseline test metrics:")
            for metric, detail in self.comparison.items():
                arrow = "v" if detail["delta"] < 0 else "^"
                flag = " DEGRADED" if detail["degraded"] else ""
                lines.append(
                    f"    {metric:10s} baseline={detail['baseline']:.4f}  "
                    f"current={detail['current']:.4f}  "
                    f"delta={arrow}{abs(detail['delta']):.4f}{flag}"
                )
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


def load_baseline_test_metrics(path: Path = config.BASELINE_METRICS_PATH) -> dict:
    """Read the Phase 3 held-out test metrics, the fixed point to compare against."""
    if not path.exists():
        raise FileNotFoundError(
            f"No baseline metrics at {path}. Run `python train_baseline.py` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "test" not in payload:
        raise ValueError(f"{path} does not contain a 'test' section.")
    return payload["test"]


def compare_to_baseline(
    current: dict, baseline: dict, margin: float = DEFAULT_DEGRADATION_MARGIN
) -> dict:
    """Compare current metrics against the Phase 3 baseline, metric by metric.

    `degraded` means the current value fell more than `margin` below the
    baseline. This is a plain threshold, not a statistical test - it does not
    know whether the current sample is big enough to trust, which is exactly
    why `PerformanceReport` always states `n_labeled` next to the verdict
    rather than letting this comparison speak alone.
    """
    comparable = ("accuracy", "precision", "recall", "f1_score", "roc_auc")
    result = {}
    for metric in comparable:
        baseline_value = baseline.get(metric)
        current_value = current.get(metric)
        if baseline_value is None or current_value is None:
            continue
        delta = round(current_value - baseline_value, 4)
        result[metric] = {
            "baseline": round(float(baseline_value), 4),
            "current": round(float(current_value), 4),
            "delta": delta,
            "degraded": delta < -margin,
        }
    return result


class PerformanceMonitor:
    """Tracks predictions over time and scores them once labels arrive."""

    def __init__(self, min_labeled_sample: int = DEFAULT_MIN_LABELED_SAMPLE):
        if min_labeled_sample < 1:
            raise ValueError(
                f"min_labeled_sample must be at least 1, got {min_labeled_sample}"
            )
        self.min_labeled_sample = min_labeled_sample
        self._records: dict[str, PredictionRecord] = {}

    def register_batch(self, batch_prediction: BatchPrediction) -> int:
        """Record a scored batch's predictions, unlabelled.

        Idempotent: calling this again with the same batch (same batch_id,
        same row indices) updates nothing, since the record already exists
        and its prediction has not changed. Returns how many new records were
        created, so a caller can tell a fresh batch from a repeat.
        """
        has_probabilities = batch_prediction.probabilities is not None
        created = 0
        for row_index in batch_prediction.features.index:
            record_id = make_record_id(batch_prediction.batch_id, row_index)
            if record_id in self._records:
                continue
            probability = (
                float(batch_prediction.probabilities.loc[row_index])
                if has_probabilities
                else None
            )
            self._records[record_id] = PredictionRecord(
                record_id=record_id,
                batch_id=batch_prediction.batch_id,
                row_index=str(row_index),
                predicted_label=str(batch_prediction.predictions.loc[row_index]),
                probability=probability,
                predicted_at=batch_prediction.predicted_at,
            )
            created += 1
        return created

    def submit_labels(self, batch_id: str, actual_labels: pd.Series) -> LabelSubmissionResult:
        """Attach true labels to previously registered predictions.

        `actual_labels` is indexed the way the original batch features were -
        the same index `register_batch` used to build each record_id - not by
        record_id directly, so a caller never has to construct one by hand.

        Labels for rows that were never registered are reported as unmatched
        rather than silently ignored or used to create a new record: a label
        with no matching prediction is not something this system can score.
        """
        matched = 0
        relabeled = 0
        unmatched = []
        now = utc_now()
        for row_index, label in actual_labels.items():
            record_id = make_record_id(batch_id, row_index)
            record = self._records.get(record_id)
            if record is None:
                unmatched.append(record_id)
                continue
            if record.is_labeled:
                relabeled += 1
            record.actual_label = str(label)
            record.labeled_at = now
            matched += 1

        return LabelSubmissionResult(
            batch_id=batch_id,
            n_submitted=len(actual_labels),
            n_matched=matched,
            n_relabeled=relabeled,
            unmatched_record_ids=unmatched,
        )

    def records_frame(self, batch_id: str | None = None) -> pd.DataFrame:
        """All registered records as a table, labelled or not.

        `batch_id=None` (the default) covers every registered batch, so
        performance can be assessed across the whole monitoring period rather
        than one batch at a time, if that is what is wanted.
        """
        records = [
            r for r in self._records.values() if batch_id is None or r.batch_id == batch_id
        ]
        if not records:
            return pd.DataFrame(
                columns=[
                    "record_id", "batch_id", "row_index", "predicted_label",
                    "probability", "predicted_at", "actual_label", "labeled_at",
                ]
            )
        return pd.DataFrame([asdict(r) for r in records])

    def label_coverage(self, batch_id: str | None = None) -> dict:
        """How many registered predictions have a true label attached."""
        frame = self.records_frame(batch_id)
        n_total = len(frame)
        n_labeled = int(frame["actual_label"].notna().sum()) if n_total else 0
        return {
            "n_total": n_total,
            "n_labeled": n_labeled,
            "n_unlabeled": n_total - n_labeled,
            "labeled_percent": round(n_labeled / n_total * 100, 2) if n_total else 0.0,
        }

    def compute_performance(
        self,
        batch_id: str | None = None,
        baseline: dict | None = None,
        degradation_margin: float = DEFAULT_DEGRADATION_MARGIN,
    ) -> PerformanceReport:
        """Score whatever labels are currently available.

        Three outcomes, corresponding to the three rules in the module
        docstring:

        * zero labels      -> `status="no_labels"`,       `metrics=None`
        * some, below floor -> `status="insufficient_labels"`, metrics computed
          and returned, but flagged, exactly like a small drift batch in
          `drift.py` - visible, not hidden, with its reliability stated.
        * at or above floor -> `status="computed"`, metrics returned and
          compared against the baseline if one was given.
        """
        frame = self.records_frame(batch_id)
        n_total = len(frame)
        labeled = frame[frame["actual_label"].notna()]
        n_labeled = len(labeled)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if n_labeled == 0:
            return PerformanceReport(
                status=NO_LABELS,
                evaluated_at=now,
                batch_id=batch_id,
                n_total=n_total,
                n_labeled=0,
                min_labeled_sample=self.min_labeled_sample,
                metrics=None,
                notes=[
                    "No actual labels have been submitted for these records yet. "
                    "Performance cannot be calculated without them, and none has "
                    "been fabricated."
                ],
            )

        y_true = labeled["actual_label"]
        y_pred = labeled["predicted_label"]
        has_scores = labeled["probability"].notna().all()
        y_score = labeled["probability"] if has_scores else None

        metrics = evaluate_predictions(y_true, y_pred, y_score)

        notes = []
        status = COMPUTED
        if n_labeled < self.min_labeled_sample:
            status = INSUFFICIENT
            notes.append(
                f"SMALL LABELLED SAMPLE: only {n_labeled} labelled rows, below "
                f"the {self.min_labeled_sample}-row floor measured for recall's "
                "standard deviation to settle under 0.05 (see "
                "docs/performance_monitoring.md). Metrics are reported, not "
                "hidden, but should not be treated as a confident verdict."
            )
        if not has_scores:
            notes.append(
                "Some labelled rows have no recorded probability, so ROC-AUC "
                "could not be computed for this slice."
            )

        comparison = None
        if baseline is not None:
            comparison = compare_to_baseline(metrics, baseline, degradation_margin)
            degraded = [m for m, d in comparison.items() if d["degraded"]]
            if degraded:
                notes.append(
                    f"{', '.join(degraded)} fell more than {degradation_margin} "
                    "below the Phase 3 baseline. This is a plain threshold on "
                    "the current sample, not a statistical test, and it does "
                    "not by itself prove the model needs retraining - confirm "
                    "with more labelled data before acting on it."
                )

        return PerformanceReport(
            status=status,
            evaluated_at=now,
            batch_id=batch_id,
            n_total=n_total,
            n_labeled=n_labeled,
            min_labeled_sample=self.min_labeled_sample,
            metrics=metrics,
            baseline=baseline,
            comparison=comparison,
            notes=notes,
        )
