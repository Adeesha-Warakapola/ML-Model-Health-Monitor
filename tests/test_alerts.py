"""Tests for alert generation.

Each test builds a minimal, hand-constructed report object rather than
running the full pipeline, so the expected alert can be reasoned about
directly: this module must only *read* fields from these reports, so a
report built by hand is just as valid an input as a real one.
"""

import pytest

from monitor.alerts import (
    CRITICAL,
    INFO,
    WARNING,
    Alert,
    AlertRules,
    alerts_from_drift,
    alerts_from_performance,
    alerts_from_prediction_drift,
    alerts_from_validation,
    escalate,
    generate_alerts,
    make_dedup_key,
    severity_rank,
)
from monitor.drift import DRIFT, MODERATE, STABLE, DriftReport, DriftThresholds, FeatureDriftResult
from monitor.performance_monitoring import COMPUTED, INSUFFICIENT, NO_LABELS, PerformanceReport
from monitor.prediction_monitoring import PredictionDriftReport
from monitor.validation import ERROR, WARNING as VALIDATION_WARNING, ValidationIssue, ValidationResult


# --- helpers to build minimal report objects --------------------------------


def make_drift_report(statuses: dict[str, str], batch_id="batch_1") -> DriftReport:
    results = [
        FeatureDriftResult(
            feature=name, feature_type="numeric", method="ks+psi", status=status,
            psi={STABLE: 0.02, MODERATE: 0.15, DRIFT: 2.5}[status],
            n_reference=1000, n_batch=200,
        )
        for name, status in statuses.items()
    ]
    return DriftReport(
        batch_id=batch_id, evaluated_at="now", feature_results=results,
        validation=ValidationResult(issues=[], n_rows=200),
        thresholds=DriftThresholds(), n_reference=1000, n_batch=200,
    )


def make_prediction_report(label_status: str, confidence_status: str) -> PredictionDriftReport:
    return PredictionDriftReport(
        batch_id="batch_1", evaluated_at="now", n_reference=1000, n_batch=200,
        threshold=0.29, reference_distribution={}, batch_distribution={},
        label_status=label_status, label_psi={STABLE: 0.01, MODERATE: 0.15, DRIFT: 2.0}[label_status],
        label_statistic=None, label_p_value=None, label_effect_size=None,
        reference_confidence={"n": 0}, batch_confidence={"n": 0},
        confidence_status=confidence_status,
        confidence_psi={STABLE: 0.01, MODERATE: 0.15, DRIFT: 2.0}.get(confidence_status),
        confidence_statistic=None, confidence_p_value=None, thresholds=DriftThresholds(),
    )


def make_performance_report(
    status: str, comparison: dict | None = None, n_labeled=500, min_labeled_sample=300
) -> PerformanceReport:
    return PerformanceReport(
        status=status, evaluated_at="now", batch_id="batch_1", n_total=500,
        n_labeled=n_labeled, min_labeled_sample=min_labeled_sample,
        metrics={"accuracy": 0.7} if status != NO_LABELS else None,
        comparison=comparison,
    )


DEGRADED_ACCURACY = {
    "accuracy": {"baseline": 0.85, "current": 0.70, "delta": -0.15, "degraded": True},
    "recall": {"baseline": 0.80, "current": 0.79, "delta": -0.01, "degraded": False},
}
DEGRADED_PRECISION_ONLY = {
    "precision": {"baseline": 0.70, "current": 0.60, "delta": -0.10, "degraded": True},
    "accuracy": {"baseline": 0.85, "current": 0.84, "delta": -0.01, "degraded": False},
}
NOT_DEGRADED = {
    "accuracy": {"baseline": 0.85, "current": 0.86, "delta": 0.01, "degraded": False},
}


# --- Alert dataclass ----------------------------------------------------------


def test_alert_requires_a_known_severity():
    with pytest.raises(ValueError, match="Unknown severity"):
        Alert(category="x", severity="URGENT!!", title="t", description="d", reason="r", batch_id="b")


def test_alert_auto_generates_a_dedup_key_from_feature():
    alert = Alert(category="data_drift", severity=WARNING, title="t", description="d",
                   reason="r", batch_id="b1", feature="age")
    assert alert.dedup_key == make_dedup_key("data_drift", "age")


def test_alert_dedup_key_falls_back_to_metric_name_then_batch():
    by_metric = Alert(category="performance", severity=CRITICAL, title="t", description="d",
                       reason="r", batch_id="b1", metric_name="accuracy")
    assert by_metric.dedup_key == make_dedup_key("performance", "accuracy")

    by_batch = Alert(category="x", severity=INFO, title="t", description="d", reason="r", batch_id="b1")
    assert by_batch.dedup_key == make_dedup_key("x", "batch")


def test_two_alerts_about_the_same_feature_and_severity_share_a_dedup_key():
    a1 = Alert(category="data_drift", severity=WARNING, title="t1", description="d",
               reason="r", batch_id="batch_A", feature="age")
    a2 = Alert(category="data_drift", severity=WARNING, title="t2", description="d",
               reason="r", batch_id="batch_B", feature="age")
    assert a1.dedup_key == a2.dedup_key  # same problem, different batch/time


def test_same_feature_shares_a_dedup_key_even_if_severity_differs():
    # Severity is deliberately excluded from the key: an issue that escalates
    # from WARNING to CRITICAL must update the same row, not start a new one.
    warning = Alert(category="data_drift", severity=WARNING, title="t", description="d",
                     reason="r", batch_id="b", feature="age")
    critical = Alert(category="data_drift", severity=CRITICAL, title="t", description="d",
                      reason="r", batch_id="b", feature="age")
    assert warning.dedup_key == critical.dedup_key


def test_alerts_about_different_features_have_different_dedup_keys():
    a1 = Alert(category="data_drift", severity=WARNING, title="t", description="d",
               reason="r", batch_id="b", feature="age")
    a2 = Alert(category="data_drift", severity=WARNING, title="t", description="d",
               reason="r", batch_id="b", feature="hours_per_week")
    assert a1.dedup_key != a2.dedup_key


# --- severity helpers ----------------------------------------------------------


def test_severity_rank_orders_info_warning_critical():
    assert severity_rank(INFO) < severity_rank(WARNING) < severity_rank(CRITICAL)


def test_escalate_never_downgrades():
    assert escalate(CRITICAL, WARNING) == CRITICAL
    assert escalate(WARNING, INFO) == WARNING


def test_escalate_upgrades_when_the_new_one_is_worse():
    assert escalate(WARNING, CRITICAL) == CRITICAL
    assert escalate(INFO, WARNING) == WARNING


def test_escalate_keeps_equal_severities_unchanged():
    assert escalate(WARNING, WARNING) == WARNING


# --- data-quality alerts -------------------------------------------------------


def test_validation_error_becomes_critical():
    validation = ValidationResult(
        issues=[ValidationIssue(severity=ERROR, check="required_columns", message="missing age")],
        n_rows=0,
    )
    alerts = alerts_from_validation(validation, "batch_1")

    assert len(alerts) == 1
    assert alerts[0].severity == CRITICAL
    assert alerts[0].category == "data_quality"


def test_validation_warning_becomes_warning_not_critical():
    validation = ValidationResult(
        issues=[ValidationIssue(severity=VALIDATION_WARNING, check="missing_values",
                                 column="workclass", message="5% missing", affected_rows=25)],
        n_rows=500,
    )
    alerts = alerts_from_validation(validation, "batch_1")

    assert alerts[0].severity == WARNING
    assert alerts[0].feature == "workclass"


def test_validation_description_is_plain_language_not_jargon():
    validation = ValidationResult(
        issues=[ValidationIssue(severity=VALIDATION_WARNING, check="unknown_categories",
                                 column="occupation", message="x")],
        n_rows=500,
    )
    description = alerts_from_validation(validation, "batch_1")[0].description
    assert "PSI" not in description  # no raw statistics jargon
    assert "never seen" in description.lower()


def test_clean_validation_produces_no_alerts():
    assert alerts_from_validation(ValidationResult(issues=[], n_rows=500), "batch_1") == []


# --- data-drift alerts ---------------------------------------------------------


def test_stable_features_produce_no_drift_alerts():
    report = make_drift_report({"age": STABLE, "education": STABLE})
    assert alerts_from_drift(report) == []


def test_moderate_drift_is_warning_severe_drift_is_critical():
    report = make_drift_report({"age": DRIFT, "hours_per_week": MODERATE, "education": STABLE})
    alerts = {a.feature: a.severity for a in alerts_from_drift(report)}

    assert alerts["age"] == CRITICAL
    assert alerts["hours_per_week"] == WARNING
    assert "education" not in alerts


def test_drift_alert_names_the_feature_and_states_the_psi():
    report = make_drift_report({"age": DRIFT})
    alert = alerts_from_drift(report)[0]

    assert "age" in alert.title
    assert alert.metric_name == "psi"
    assert alert.metric_value == pytest.approx(2.5)
    assert "2.5" in alert.reason


# --- prediction-drift alerts ----------------------------------------------------


def test_stable_predictions_produce_no_alert():
    report = make_prediction_report(STABLE, STABLE)
    assert alerts_from_prediction_drift(report) == []


def test_label_and_confidence_drift_are_reported_separately():
    report = make_prediction_report(MODERATE, DRIFT)
    alerts = alerts_from_prediction_drift(report)

    assert len(alerts) == 2
    titles = {a.title for a in alerts}
    assert "Predicted-label mix has shifted" in titles
    assert "Prediction confidence has shifted" in titles
    severities = {a.title: a.severity for a in alerts}
    assert severities["Predicted-label mix has shifted"] == WARNING
    assert severities["Prediction confidence has shifted"] == CRITICAL


def test_prediction_drift_description_does_not_claim_model_failure():
    report = make_prediction_report(DRIFT, STABLE)
    description = alerts_from_prediction_drift(report)[0].description
    assert "not proof" in description.lower()


# --- performance alerts: the no-labels vs degraded distinction -----------------


def test_no_labels_produces_no_performance_alert():
    report = make_performance_report(NO_LABELS, comparison=None)
    assert alerts_from_performance(report) == []


def test_computed_with_no_degraded_metric_produces_no_alert():
    report = make_performance_report(COMPUTED, comparison=NOT_DEGRADED)
    assert alerts_from_performance(report) == []


def test_confirmed_degradation_of_a_core_metric_is_critical():
    report = make_performance_report(COMPUTED, comparison=DEGRADED_ACCURACY)
    alerts = alerts_from_performance(report)

    assert len(alerts) == 1
    assert alerts[0].severity == CRITICAL
    assert alerts[0].category == "performance"


def test_confirmed_degradation_of_a_non_core_metric_is_warning_not_critical():
    report = make_performance_report(COMPUTED, comparison=DEGRADED_PRECISION_ONLY)
    alert = alerts_from_performance(report)[0]
    assert alert.severity == WARNING


def test_insufficient_sample_degradation_is_capped_below_confirmed_severity():
    confirmed = make_performance_report(COMPUTED, comparison=DEGRADED_ACCURACY)
    unconfirmed = make_performance_report(
        INSUFFICIENT, comparison=DEGRADED_ACCURACY, n_labeled=120
    )

    confirmed_alert = alerts_from_performance(confirmed)[0]
    unconfirmed_alert = alerts_from_performance(unconfirmed)[0]

    assert confirmed_alert.severity == CRITICAL
    assert unconfirmed_alert.severity == WARNING  # capped one level down
    assert "unconfirmed" in unconfirmed_alert.description.lower()


def test_insufficient_sample_warning_level_caps_to_info():
    confirmed = make_performance_report(COMPUTED, comparison=DEGRADED_PRECISION_ONLY)
    unconfirmed = make_performance_report(
        INSUFFICIENT, comparison=DEGRADED_PRECISION_ONLY, n_labeled=50
    )
    assert alerts_from_performance(confirmed)[0].severity == WARNING
    assert alerts_from_performance(unconfirmed)[0].severity == INFO


def test_core_metrics_list_is_configurable():
    custom_rules = AlertRules(core_performance_metrics=("precision",))
    report = make_performance_report(COMPUTED, comparison=DEGRADED_PRECISION_ONLY)

    alert = alerts_from_performance(report, rules=custom_rules)[0]
    assert alert.severity == CRITICAL  # precision is now a core metric


# --- generate_alerts: the single entry point ------------------------------------


def test_generate_alerts_combines_every_category():
    validation = ValidationResult(
        issues=[ValidationIssue(severity=ERROR, check="required_columns", message="x")], n_rows=0,
    )
    drift = make_drift_report({"age": DRIFT})
    prediction_drift = make_prediction_report(MODERATE, STABLE)
    performance = make_performance_report(COMPUTED, comparison=DEGRADED_ACCURACY)

    alerts = generate_alerts(
        batch_id="batch_1", validation=validation, drift=drift,
        prediction_drift=prediction_drift, performance=performance,
    )
    categories = {a.category for a in alerts}
    assert categories == {"data_quality", "data_drift", "prediction_drift", "performance"}


def test_generate_alerts_tolerates_missing_reports():
    # performance is commonly unavailable (no labels yet); must not crash.
    alerts = generate_alerts(batch_id="batch_1", drift=make_drift_report({"age": STABLE}))
    assert alerts == []


def test_generate_alerts_with_nothing_at_all_returns_empty_list():
    assert generate_alerts(batch_id="batch_1") == []
