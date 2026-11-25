"""Tests for the pure dashboard helper functions.

These run with no model, no dataset, and no Streamlit process - every
function here only shapes plain dicts and DataFrames that a repository or a
report object would already produce.
"""

import pandas as pd
import pytest

from monitor.alerts import CRITICAL, INFO, WARNING
from monitor.dashboard_data import (
    DEFAULT_SCENARIO,
    SCENARIOS,
    drift_status_color,
    drift_status_label,
    empty_state,
    feature_kind,
    format_metric,
    format_percent,
    format_psi,
    format_timestamp,
    latest_row,
    overall_batch_status,
    overall_drift_status_from_frame,
    parse_json_field,
    performance_status_label,
    severity_counts,
    worse_status,
)
from monitor.drift import DRIFT, MODERATE, NOT_COMPUTED, STABLE
from monitor.performance_monitoring import COMPUTED, INSUFFICIENT, NO_LABELS


def test_scenarios_cover_stable_moderate_and_severe_cases():
    assert DEFAULT_SCENARIO in SCENARIOS
    assert SCENARIOS[DEFAULT_SCENARIO].modifier is None

    non_stable = [s for name, s in SCENARIOS.items() if s.modifier is not None]
    assert len(non_stable) >= 2, "expected at least a moderate and a severe scenario"


def test_each_scenario_modifier_is_callable_and_labelled():
    for scenario in SCENARIOS.values():
        assert scenario.label
        assert scenario.description
        if scenario.modifier is not None:
            assert callable(scenario.modifier)


@pytest.mark.parametrize(
    "status,expected_label",
    [(STABLE, "Stable"), (MODERATE, "Moderate"), (DRIFT, "Drift"), (NOT_COMPUTED, "Not computed")],
)
def test_drift_status_label_covers_every_known_status(status, expected_label):
    assert drift_status_label(status) == expected_label


def test_drift_status_label_falls_back_gracefully_for_unknown_status():
    assert drift_status_label("made_up_status") == "made_up_status"


def test_drift_colors_are_distinct_for_each_status():
    colors = {drift_status_color(s) for s in (STABLE, MODERATE, DRIFT)}
    assert len(colors) == 3


@pytest.mark.parametrize(
    "drift_status,prediction_status,expected",
    [
        (STABLE, STABLE, STABLE),
        (STABLE, MODERATE, MODERATE),
        (DRIFT, STABLE, DRIFT),
        (MODERATE, DRIFT, DRIFT),
        (NOT_COMPUTED, NOT_COMPUTED, NOT_COMPUTED),
    ],
)
def test_overall_batch_status_takes_the_worse_of_the_two(drift_status, prediction_status, expected):
    assert overall_batch_status(drift_status, prediction_status) == expected


def test_worse_status_combines_any_number_of_statuses():
    assert worse_status(STABLE, STABLE, MODERATE) == MODERATE
    assert worse_status(DRIFT, STABLE, MODERATE) == DRIFT


def test_worse_status_ignores_none_and_defaults_to_not_computed():
    assert worse_status(None, None) == NOT_COMPUTED
    assert worse_status(None, STABLE) == STABLE


def test_overall_drift_status_from_frame_matches_worst_feature():
    frame = pd.DataFrame({"status": [STABLE, MODERATE, STABLE]})
    assert overall_drift_status_from_frame(frame) == MODERATE

    frame_drift = pd.DataFrame({"status": [STABLE, DRIFT, MODERATE]})
    assert overall_drift_status_from_frame(frame_drift) == DRIFT


def test_overall_drift_status_from_empty_frame_is_not_computed():
    assert overall_drift_status_from_frame(pd.DataFrame()) == NOT_COMPUTED
    assert overall_drift_status_from_frame(None) == NOT_COMPUTED


def test_format_metric_distinguishes_none_from_zero():
    assert format_metric(None) == "Not available"
    assert format_metric(0.0) == "0.0000"
    assert format_metric(0.8231, digits=2) == "0.82"


def test_format_metric_treats_nan_as_unavailable_like_a_sqlite_null():
    import math

    assert format_metric(float("nan")) == "Not available"
    assert format_metric(math.nan) == "Not available"


def test_format_percent_distinguishes_none_from_zero():
    assert format_percent(None) == "Not available"
    assert format_percent(0.0) == "0.0%"
    assert format_percent(0.5, digits=0) == "50%"


def test_format_psi_handles_missing_value():
    assert format_psi(None) == "n/a"
    assert format_psi(0.1587) == "0.1587"


def test_format_timestamp_handles_missing_value():
    assert format_timestamp(None) == "Not available yet"
    assert format_timestamp("2026-01-01T00:00:00+00:00") == "2026-01-01T00:00:00+00:00"


def test_performance_status_label_distinguishes_all_three_outcomes():
    labels = {
        performance_status_label(NO_LABELS),
        performance_status_label(INSUFFICIENT),
        performance_status_label(COMPUTED),
    }
    assert len(labels) == 3
    assert "No labels" in performance_status_label(NO_LABELS)


def test_severity_counts_on_empty_dataframe_returns_zeros():
    counts = severity_counts(pd.DataFrame())
    assert counts == {INFO: 0, WARNING: 0, CRITICAL: 0}


def test_severity_counts_on_none_returns_zeros():
    assert severity_counts(None) == {INFO: 0, WARNING: 0, CRITICAL: 0}


def test_severity_counts_tallies_each_severity():
    frame = pd.DataFrame({"severity": [WARNING, WARNING, CRITICAL, INFO, WARNING]})
    counts = severity_counts(frame)
    assert counts == {INFO: 1, WARNING: 3, CRITICAL: 1}


def test_latest_row_on_empty_frame_returns_none():
    assert latest_row(pd.DataFrame(), "recorded_at") is None


def test_latest_row_picks_the_most_recent_by_timestamp_regardless_of_row_order():
    frame = pd.DataFrame(
        {
            "recorded_at": ["2026-01-01T00:00:00", "2026-01-03T00:00:00", "2026-01-02T00:00:00"],
            "batch_id": ["b1", "b3", "b2"],
        }
    )
    row = latest_row(frame, "recorded_at")
    assert row["batch_id"] == "b3"


def test_feature_kind_identifies_numeric_and_categorical_and_unknown():
    assert feature_kind("age") == "numeric"
    assert feature_kind("occupation") == "categorical"
    assert feature_kind("not_a_real_feature") == "unknown"


def test_empty_state_has_text_for_every_documented_key():
    for key in (
        "no_history", "no_labels", "no_alerts", "no_confusion_matrix", "no_raw_distribution",
    ):
        message = empty_state(key)
        assert isinstance(message, str) and message


def test_empty_state_falls_back_for_unknown_key():
    assert empty_state("totally_unknown_key") == "Nothing to show yet."


def test_parse_json_field_parses_a_real_json_string():
    assert parse_json_field('["a note"]') == ["a note"]
    assert parse_json_field('{"accuracy": 0.8}') == {"accuracy": 0.8}


def test_parse_json_field_treats_none_as_missing():
    assert parse_json_field(None) is None
    assert parse_json_field(None, default=[]) == []


def test_parse_json_field_treats_nan_as_missing_like_a_sqlite_null():
    """The exact bug this guards against: a SQLite NULL for an empty notes
    list comes back from pandas as float NaN, not None, and NaN is truthy -
    so a plain `json.loads(x) if x else default` still calls json.loads(nan)
    and crashes with a TypeError instead of returning `default`.
    """
    assert parse_json_field(float("nan")) is None
    assert parse_json_field(float("nan"), default=[]) == []
    assert parse_json_field(float("nan"), default={}) == {}
