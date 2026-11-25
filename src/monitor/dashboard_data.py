"""Pure helper functions behind the Streamlit dashboard pages.

Nothing in this module imports Streamlit or Plotly, and nothing here detects
drift, scores a model, or computes a metric - that stays in `drift.py`,
`prediction_monitoring.py`, `performance_monitoring.py` and `alerts.py`. This
module only shapes already-computed results (plain dicts, dataclasses and
`pandas.DataFrame`s) into the small pieces of data a page needs to render:
status labels, colours, formatted numbers, and empty-state text. Kept
Streamlit-free so it can be unit tested directly, without starting a
dashboard.
"""

import json
from dataclasses import dataclass

import pandas as pd

from . import schema
from .alerts import CRITICAL, INFO, WARNING
from .drift import DRIFT, MODERATE, NOT_COMPUTED, STABLE
from .performance_monitoring import COMPUTED, INSUFFICIENT, NO_LABELS
from .simulation import Modifier, overwrite_category, shift_numeric

# --- demonstration scenarios -------------------------------------------------
# Named modifiers for the dashboard's "run a new batch" control. Each one
# reuses a modifier already built and measured in `simulation.py` /
# `manage_alerts.py` - nothing new is introduced here, only labelled for a
# dropdown. PSI values are the ones measured in docs/alert_management.md.


@dataclass(frozen=True)
class Scenario:
    """One demonstration scenario: a label, a description, and a modifier."""

    label: str
    description: str
    modifier: Modifier | None  # None means "no change": a stable batch


SCENARIOS: dict[str, Scenario] = {
    "Stable batch": Scenario(
        label="Stable batch",
        description=(
            "Unmodified holdout rows. Expect only the dataset's usual "
            "missing-value warnings for workclass/occupation/native_country - "
            "nothing drift- or performance-related."
        ),
        modifier=None,
    ),
    "Moderate drift (occupation category shift)": Scenario(
        label="Moderate drift (occupation category shift)",
        description=(
            "5% of rows get an unseen 'Gig-economy' occupation. Measured PSI "
            "0.1587 - moderate band (0.10-0.25) - so expect a WARNING-level "
            "data-drift alert, not CRITICAL."
        ),
        modifier=overwrite_category("occupation", "Gig-economy", fraction=0.05),
    ),
    "Severe drift (age shifted +25 years)": Scenario(
        label="Severe drift (age shifted +25 years)",
        description=(
            "Every row's age is shifted 25 years older. This is large enough "
            "to cross the major PSI band and to measurably change accuracy, "
            "so expect CRITICAL data-drift and performance alerts."
        ),
        modifier=shift_numeric("age", 25),
    ),
}

DEFAULT_SCENARIO = "Stable batch"
DEFAULT_BATCH_SIZE = 600
MIN_BATCH_SIZE = 100
MAX_BATCH_SIZE = 2000
BATCH_SIZE_STEP = 50


# --- status/severity presentation -------------------------------------------
# Colours chosen for a dark background: desaturated enough not to clash, but
# distinct enough to scan at a glance. Centralised here so every page agrees
# on what "stable" or "CRITICAL" looks like.

DRIFT_STATUS_STYLE = {
    STABLE: ("Stable", "#3ddc97"),
    MODERATE: ("Moderate", "#e8b339"),
    DRIFT: ("Drift", "#e8545c"),
    NOT_COMPUTED: ("Not computed", "#6b7280"),
}

SEVERITY_STYLE = {
    INFO: ("#4f8ff0"),
    WARNING: ("#e8b339"),
    CRITICAL: ("#e8545c"),
}

PERFORMANCE_STATUS_STYLE = {
    NO_LABELS: ("No labels yet", "#6b7280"),
    INSUFFICIENT: ("Early signal (small sample)", "#e8b339"),
    COMPUTED: ("Confirmed", "#3ddc97"),
}


def drift_status_label(status: str) -> str:
    """Plain-language label for a `drift.py` / `prediction_monitoring.py` status."""
    return DRIFT_STATUS_STYLE.get(status, (status, "#6b7280"))[0]


def drift_status_color(status: str) -> str:
    return DRIFT_STATUS_STYLE.get(status, (status, "#6b7280"))[1]


def severity_color(severity: str) -> str:
    return SEVERITY_STYLE.get(severity, "#6b7280")


def performance_status_label(status: str) -> str:
    return PERFORMANCE_STATUS_STYLE.get(status, (status, "#6b7280"))[0]


def performance_status_color(status: str) -> str:
    return PERFORMANCE_STATUS_STYLE.get(status, (status, "#6b7280"))[1]


_STATUS_ORDER = {STABLE: 0, NOT_COMPUTED: 0, MODERATE: 1, DRIFT: 2}


def worse_status(*statuses: str | None) -> str:
    """The most severe of any number of stable/moderate_drift/drift/not_computed values.

    The same STABLE < MODERATE < DRIFT ordering `DriftReport.status` and
    `PredictionDriftReport.status` already use - this only combines
    already-known verdicts (several features, or label + confidence), never
    a new statistic. `None` entries are ignored, and an empty or all-`None`
    call reports `NOT_COMPUTED` rather than raising.
    """
    present = [s for s in statuses if s is not None]
    if not present:
        return NOT_COMPUTED
    return max(present, key=lambda s: _STATUS_ORDER.get(s, 0))


def overall_batch_status(drift_status: str, prediction_status: str) -> str:
    """Worst of the data-drift and prediction-drift verdicts for one batch.

    Performance is deliberately excluded: a `no_labels` batch must not make
    the overall picture look worse (or better) than the two signals that are
    always available.
    """
    return worse_status(drift_status, prediction_status)


def overall_drift_status_from_frame(drift_df: pd.DataFrame) -> str:
    """Reconstruct a batch's overall drift verdict from the persisted per-feature rows.

    Lets a restart-safe page (reading only `drift_results`, with no live
    `DriftReport` object) show the same overall verdict the live object's
    `.status` property would have reported for the same feature results.
    """
    if drift_df is None or drift_df.empty:
        return NOT_COMPUTED
    return worse_status(*drift_df["status"].tolist())


# --- formatting: distinguishing "unavailable" from "zero" -------------------


def _is_missing(value) -> bool:
    """True for `None` and for NaN - a SQLite NULL comes back from pandas as
    the latter, not the former, so both must be treated as "never computed".
    """
    return value is None or (isinstance(value, float) and pd.isna(value))


def format_metric(value: float | None, digits: int = 4) -> str:
    """A metric value, or an explicit "Not available" - never a bare 0.0000.

    Missing means "this was never computed," which must never be confused
    with an actually-measured value of zero.
    """
    if _is_missing(value):
        return "Not available"
    return f"{value:.{digits}f}"


def format_percent(value: float | None, digits: int = 1) -> str:
    if _is_missing(value):
        return "Not available"
    return f"{value * 100:.{digits}f}%"


def format_psi(value: float | None) -> str:
    if _is_missing(value):
        return "n/a"
    return f"{value:.4f}"


def format_timestamp(value: str | None) -> str:
    return value if value else "Not available yet"


def parse_json_field(value, default=None):
    """Parse a JSON-encoded database column, treating a missing value safely.

    A SQLite NULL comes back from `pandas.read_sql_query` as NaN (a float),
    not `None` - and `nan` is truthy in Python, so a plain
    `json.loads(x) if x else default` still calls `json.loads(nan)` and
    raises a confusing `TypeError`. This is the one place every dashboard
    page should parse a nullable JSON column, so that mistake only needs
    fixing once.
    """
    if _is_missing(value):
        return default
    return json.loads(value)


# --- empty states ------------------------------------------------------------

EMPTY_STATES = {
    "no_history": (
        "No monitoring runs yet. Use “Run monitoring batch” in the "
        "sidebar to score a batch and populate this page."
    ),
    "no_labels": (
        "No true labels have been submitted for this batch yet, so "
        "performance is unknown - this is deliberately different from "
        "reporting a zero or a failing score."
    ),
    "no_alerts": "No alerts have been raised for the selected filters.",
    "no_confusion_matrix": (
        "No confusion matrix was saved for this run - it was likely created "
        "by an older version of this project, before the full metrics were "
        "stored. The summary metrics above are still accurate."
    ),
    "no_raw_distribution": (
        "The underlying values are only available for the batch most "
        "recently run in this dashboard session. Historical batches below "
        "show the saved summary statistics instead - nothing fabricated."
    ),
}


def empty_state(key: str) -> str:
    return EMPTY_STATES.get(key, "Nothing to show yet.")


# --- summaries over repository-backed DataFrames -----------------------------


def severity_counts(alerts_df: pd.DataFrame) -> dict:
    """How many alert rows fall in each severity, defaulting missing ones to zero."""
    counts = {INFO: 0, WARNING: 0, CRITICAL: 0}
    if alerts_df is None or alerts_df.empty:
        return counts
    observed = alerts_df["severity"].value_counts().to_dict()
    for severity in counts:
        counts[severity] = int(observed.get(severity, 0))
    return counts


def latest_row(frame: pd.DataFrame, timestamp_column: str) -> dict | None:
    """The most recent row of a repository DataFrame, as a plain dict, or None.

    Repository `get_*` methods already return rows ordered newest-first, but
    this re-sorts defensively so callers do not depend on that ordering
    silently staying true.
    """
    if frame is None or frame.empty:
        return None
    sorted_frame = frame.sort_values(timestamp_column, ascending=False)
    return sorted_frame.iloc[0].to_dict()


def feature_kind(feature: str) -> str:
    """Whether a feature name is numeric or categorical, from the schema."""
    if feature in schema.NUMERIC_COLUMNS:
        return "numeric"
    if feature in schema.CATEGORICAL_COLUMNS:
        return "categorical"
    return "unknown"
