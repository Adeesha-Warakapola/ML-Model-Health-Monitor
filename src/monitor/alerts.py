"""Turn already-computed monitoring results into alerts a person can act on.

This module computes nothing new. Every alert is built by reading fields
that `validation.py`, `drift.py`, `prediction_monitoring.py` and
`performance_monitoring.py` already calculated. The job here is narrower:
decide which of those already-known facts are worth surfacing, how loudly,
and in what plain-language wording - not to re-run any statistic.

Three severities, in increasing order of urgency:

* **INFO** - worth recording, nothing to act on yet. Used for low-confidence
  signals, such as a degraded-looking metric computed from too few labels to
  trust.
* **WARNING** - a real, moderate issue. The kind of thing to look into this
  week, not drop everything for.
* **CRITICAL** - a large, confirmed issue (a severe drift, a confirmed
  performance drop, or data the pipeline cannot process at all).

Which category maps to which severity is controlled by `AlertRules`, kept
separate from the statistical thresholds in `drift.py` and
`performance_monitoring.py`. Those modules already decide *whether* something
is stable, moderate, or severe; this module only decides how that verdict
should be communicated.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import config
from .drift import DRIFT as DATA_DRIFT_SEVERE
from .drift import MODERATE as DATA_DRIFT_MODERATE
from .drift import DriftReport
from .performance_monitoring import COMPUTED, INSUFFICIENT, NO_LABELS, PerformanceReport
from .prediction_monitoring import DRIFT as PREDICTION_DRIFT_SEVERE
from .prediction_monitoring import MODERATE as PREDICTION_DRIFT_MODERATE
from .prediction_monitoring import PredictionDriftReport
from .validation import ERROR as DATA_QUALITY_ERROR
from .validation import ValidationResult

INFO = "INFO"
WARNING = "WARNING"
CRITICAL = "CRITICAL"
_SEVERITY_ORDER = {INFO: 0, WARNING: 1, CRITICAL: 2}

CATEGORY_DATA_QUALITY = "data_quality"
CATEGORY_DATA_DRIFT = "data_drift"
CATEGORY_PREDICTION_DRIFT = "prediction_drift"
CATEGORY_PERFORMANCE = "performance"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def severity_rank(severity: str) -> int:
    """Where a severity sits on the INFO < WARNING < CRITICAL scale."""
    return _SEVERITY_ORDER[severity]


def escalate(current: str, candidate: str) -> str:
    """The more urgent of two severities.

    Used when the same underlying problem is seen again: a recurrence is
    never allowed to quietly downgrade an already-known CRITICAL issue to
    WARNING just because this particular occurrence looked a little milder.
    """
    return current if severity_rank(current) >= severity_rank(candidate) else candidate


def make_dedup_key(category: str, subject: str) -> str:
    """A stable key identifying 'the same problem', independent of severity, batch or time.

    Deliberately coarser than batch_id, timestamp, *or severity*: two alerts
    about the same category and the same subject (a feature name, a metric
    name) are the same ongoing problem recurring, even if one occurrence was
    WARNING and the next is CRITICAL. Severity is excluded from the key on
    purpose - if it were included, a problem that escalated from WARNING to
    CRITICAL would get a *second*, unrelated-looking row instead of the
    existing one being escalated, which defeats the point of
    `storage.escalate`. See `storage.py` for how this key decides whether to
    insert a new alert row or update an existing one.
    """
    raw = f"{category}:{subject}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Alert:
    """One thing worth a person's attention, in plain language.

    `reason` is the precise, numeric justification (a statistic, a
    threshold); `description` is the same fact restated for someone who has
    not read the statistics documentation. Both are kept, because a
    dashboard wants the plain sentence and an engineer debugging it wants the
    exact number.
    """

    category: str
    severity: str
    title: str
    description: str
    reason: str
    batch_id: str | None
    feature: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    threshold_value: float | None = None
    triggered_at: str = field(default_factory=_utc_now)
    dedup_key: str = ""

    def __post_init__(self):
        if self.severity not in _SEVERITY_ORDER:
            raise ValueError(
                f"Unknown severity '{self.severity}'. Use one of: "
                f"{list(_SEVERITY_ORDER)}"
            )
        if not self.dedup_key:
            subject = self.feature or self.metric_name or "batch"
            self.dedup_key = make_dedup_key(self.category, subject)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        location = f"[{self.feature}] " if self.feature else ""
        return f"{self.severity:8s} {self.category:18s} {location}{self.title}"


@dataclass(frozen=True)
class AlertRules:
    """Which already-computed facts become which severity of alert.

    These are judgment calls about *communication*, separate from the
    statistical thresholds (`DriftThresholds`, `DEFAULT_DEGRADATION_MARGIN`)
    that decide the underlying stable/moderate/drift or degraded/not-degraded
    verdicts. Changing these numbers changes how loudly a given verdict is
    announced, not what the verdict is.

    `core_performance_metrics`
        If a drop in one of these specific metrics is confirmed, the alert is
        CRITICAL rather than WARNING. Accuracy, recall and F1 are chosen
        because this project's baseline (Phase 3) found recall to be the
        metric that mattered most for this imbalanced classification problem;
        precision and ROC-AUC moving alone is still worth a WARNING, not
        silence.
    """

    core_performance_metrics: tuple[str, ...] = ("accuracy", "recall", "f1_score")


def _explain_validation_issue(issue) -> str:
    """A plain-language sentence for one data-quality finding."""
    templates = {
        "required_columns": (
            "The batch is missing one or more columns the model needs, so it "
            "cannot be processed at all until this is fixed."
        ),
        "dtype": (
            f"The column '{issue.column}' contains the wrong kind of data "
            "(for example, numbers where text was expected). This usually "
            "means something upstream changed how the data is produced."
        ),
        "missing_values": (
            f"The column '{issue.column}' has more missing values than "
            "usual. The model fills these in automatically, but a rising "
            "trend is worth investigating."
        ),
        "unknown_categories": (
            f"The column '{issue.column}' contains a category the model has "
            "never seen during training. The model treats it safely, but it "
            "can be an early sign that the data source has changed."
        ),
        "value_range": (
            f"The column '{issue.column}' contains values outside the range "
            "seen during training (for example, an impossible age). This "
            "usually points to a data-entry or pipeline bug, not a real "
            "change in the population."
        ),
        "unexpected_columns": (
            "The batch contains extra columns the model does not use. "
            "Harmless, but worth checking that nothing important was renamed."
        ),
        "empty_batch": "The batch contained no rows at all.",
    }
    return templates.get(issue.check, str(issue))


def alerts_from_validation(validation: ValidationResult, batch_id: str) -> list[Alert]:
    """Data-quality alerts: malformed data, not a change in its distribution.

    Errors become CRITICAL (the batch cannot be processed reliably);
    warnings become WARNING (usable, but worth recording).
    """
    alerts = []
    for issue in validation.issues:
        severity = CRITICAL if issue.severity == DATA_QUALITY_ERROR else WARNING
        alerts.append(
            Alert(
                category=CATEGORY_DATA_QUALITY,
                severity=severity,
                title=f"Data quality: {issue.check.replace('_', ' ')}"
                + (f" ({issue.column})" if issue.column else ""),
                description=_explain_validation_issue(issue),
                reason=str(issue),
                batch_id=batch_id,
                feature=issue.column,
            )
        )
    return alerts


def alerts_from_drift(report: DriftReport) -> list[Alert]:
    """Input-distribution alerts, one per feature that moved enough to matter."""
    alerts = []
    for result in report.feature_results:
        if result.status == DATA_DRIFT_SEVERE:
            severity = CRITICAL
            scale = "a large, material shift"
        elif result.status == DATA_DRIFT_MODERATE:
            severity = WARNING
            scale = "a moderate shift worth watching"
        else:
            continue
        alerts.append(
            Alert(
                category=CATEGORY_DATA_DRIFT,
                severity=severity,
                title=f"Feature drift: {result.feature}",
                description=(
                    f"The feature '{result.feature}' in this batch looks "
                    f"different from the data the model was trained on - {scale}. "
                    "This can mean the kind of people or records now arriving "
                    "has genuinely changed."
                ),
                reason=(
                    f"PSI={result.psi:.4f} "
                    f"(moderate >= {report.thresholds.psi_moderate}, "
                    f"major >= {report.thresholds.psi_major})"
                ),
                batch_id=report.batch_id,
                feature=result.feature,
                metric_name="psi",
                metric_value=result.psi,
                threshold_value=(
                    report.thresholds.psi_major
                    if severity == CRITICAL
                    else report.thresholds.psi_moderate
                ),
            )
        )
    return alerts


def alerts_from_prediction_drift(report: PredictionDriftReport) -> list[Alert]:
    """Alerts about what the model is *saying*, as distinct from the input."""
    alerts = []
    if report.label_status in (PREDICTION_DRIFT_MODERATE, PREDICTION_DRIFT_SEVERE):
        severity = CRITICAL if report.label_status == PREDICTION_DRIFT_SEVERE else WARNING
        alerts.append(
            Alert(
                category=CATEGORY_PREDICTION_DRIFT,
                severity=severity,
                title="Predicted-label mix has shifted",
                description=(
                    "The share of predictions the model is making for each "
                    "income class has changed noticeably from its usual "
                    "pattern. This is not proof the model got worse - it can "
                    "also mean the input population genuinely changed - but "
                    "it is worth comparing against the data-drift results for "
                    "the same batch."
                ),
                reason=f"label PSI={report.label_psi:.4f}",
                batch_id=report.batch_id,
                metric_name="label_psi",
                metric_value=report.label_psi,
                threshold_value=report.thresholds.psi_moderate,
            )
        )
    if report.confidence_status in (PREDICTION_DRIFT_MODERATE, PREDICTION_DRIFT_SEVERE):
        severity = CRITICAL if report.confidence_status == PREDICTION_DRIFT_SEVERE else WARNING
        alerts.append(
            Alert(
                category=CATEGORY_PREDICTION_DRIFT,
                severity=severity,
                title="Prediction confidence has shifted",
                description=(
                    "The model's predicted probabilities have shifted, even "
                    "though the final yes/no labels may look similar so far. "
                    "This can be an earlier warning than the label mix alone, "
                    "since a probability can move well before it crosses the "
                    "decision threshold."
                ),
                reason=f"confidence PSI={report.confidence_psi:.4f}",
                batch_id=report.batch_id,
                metric_name="confidence_psi",
                metric_value=report.confidence_psi,
                threshold_value=report.thresholds.psi_moderate,
            )
        )
    return alerts


def alerts_from_performance(
    report: PerformanceReport, rules: AlertRules | None = None
) -> list[Alert]:
    """Performance alerts - the only category that requires true labels.

    Three outcomes map to three very different behaviours, which is the
    direct answer to "distinguish no labels from genuine degradation":

    * `NO_LABELS`   -> no alert at all. The system does not yet know whether
      the model is fine or not, and silence here means exactly that -
      "unknown," never "fine."
    * `INSUFFICIENT` with a degraded metric -> an alert is still raised, but
      capped at WARNING (even where a full sample would have meant CRITICAL)
      and the description says plainly that the result is unconfirmed, since
      the labelled sample is below the measured reliability floor.
    * `COMPUTED` with a degraded metric -> the real alert, at full severity.
    """
    if report.status == NO_LABELS or report.comparison is None:
        return []

    rules = rules or AlertRules()
    degraded = {m: d for m, d in report.comparison.items() if d["degraded"]}
    if not degraded:
        return []

    is_core = bool(set(degraded) & set(rules.core_performance_metrics))
    severity = CRITICAL if is_core else WARNING
    confirmed = report.status == COMPUTED
    if not confirmed:
        severity = INFO if severity == WARNING else WARNING  # cap one level down

    metric_list = ", ".join(
        f"{name} {detail['baseline']:.4f}->{detail['current']:.4f}"
        for name, detail in degraded.items()
    )
    confidence_clause = (
        "This is based on a labelled sample large enough to trust."
        if confirmed
        else (
            f"This is based on only {report.n_labeled} labelled rows, below "
            f"the {report.min_labeled_sample} considered reliable - treat this "
            "as an early, unconfirmed signal, not a confirmed problem."
        )
    )

    return [
        Alert(
            category=CATEGORY_PERFORMANCE,
            severity=severity,
            title="Model performance has dropped below baseline",
            description=(
                "Comparing recent predictions against their true outcomes shows "
                f"{', '.join(degraded)} falling compared with the Phase 3 "
                f"baseline test results. {confidence_clause}"
            ),
            reason=f"degraded metrics: {metric_list}",
            batch_id=report.batch_id,
            metric_name="performance",
            metric_value=min(d["delta"] for d in degraded.values()),
        )
    ]


def generate_alerts(
    *,
    batch_id: str,
    validation: ValidationResult | None = None,
    drift: DriftReport | None = None,
    prediction_drift: PredictionDriftReport | None = None,
    performance: PerformanceReport | None = None,
    rules: AlertRules | None = None,
) -> list[Alert]:
    """The single entry point: every monitoring result in, every alert out.

    Each argument is optional because a caller may not have all four
    available yet (most commonly: performance, before any label arrives).
    Nothing is recomputed here - each `alerts_from_*` helper only reads
    fields the corresponding module already calculated.
    """
    alerts: list[Alert] = []
    if validation is not None:
        alerts.extend(alerts_from_validation(validation, batch_id))
    if drift is not None:
        alerts.extend(alerts_from_drift(drift))
    if prediction_drift is not None:
        alerts.extend(alerts_from_prediction_drift(prediction_drift))
    if performance is not None:
        alerts.extend(alerts_from_performance(performance, rules))
    return alerts
