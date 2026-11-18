"""Alerts page: severity, descriptions, timestamps, affected features, history.

Reads only `MonitoringRepository.get_alerts` - the single source of truth
Phase 7 built, already deduplicated by `dedup_key` so a recurring problem
shows one row with a rising `occurrence_count` rather than flooding this
page. The date range is pushed down to the database query; severity and
category are applied in pandas afterwards so several values can be selected
at once - both are filtering already-retrieved rows, not a new calculation.
"""

import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

import dashboard_common as dc
from monitor.alerts import CATEGORY_DATA_DRIFT, CATEGORY_DATA_QUALITY, CATEGORY_PERFORMANCE, CATEGORY_PREDICTION_DRIFT
from monitor.alerts import CRITICAL, INFO, WARNING
from monitor.dashboard_data import empty_state, severity_color, severity_counts

dc.configure_page("Alerts")
context = dc.render_sidebar()
repository = dc.get_repository()

dc.page_header("Alerts", "Every distinct, ongoing problem the monitoring system has raised.")

all_alerts = repository.get_alerts()
if all_alerts.empty:
    dc.empty_panel(empty_state("no_history"))
    st.stop()

CATEGORIES = [CATEGORY_DATA_QUALITY, CATEGORY_DATA_DRIFT, CATEGORY_PREDICTION_DRIFT, CATEGORY_PERFORMANCE]

filter_cols = st.columns([1, 1, 2])
with filter_cols[0]:
    severity_filter = st.multiselect("Severity", [CRITICAL, WARNING, INFO], default=[CRITICAL, WARNING, INFO])
with filter_cols[1]:
    category_filter = st.multiselect("Category", CATEGORIES, default=CATEGORIES)
with filter_cols[2]:
    min_date = datetime.fromisoformat(all_alerts["first_seen_at"].min()).date()
    max_date = datetime.fromisoformat(all_alerts["last_seen_at"].max()).date()
    date_range = st.date_input("Last seen between", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# st.date_input returns a single date (not a tuple) while the user has only
# picked the start of the range - guard against that rather than indexing
# into a non-tuple.
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_day, end_day = date_range
else:
    start_day = end_day = date_range[0] if isinstance(date_range, tuple) else date_range

start_date = datetime.combine(start_day, time.min).isoformat()
end_date = datetime.combine(end_day, time.max).isoformat()

filtered = repository.get_alerts(start_date=start_date, end_date=end_date)
filtered = filtered[filtered["severity"].isin(severity_filter) & filtered["category"].isin(category_filter)]

counts = severity_counts(filtered)
cols = st.columns(3)
cols[0].metric("Critical", counts["CRITICAL"])
cols[1].metric("Warning", counts["WARNING"])
cols[2].metric("Info", counts["INFO"])

dc.section_label(f"{len(filtered)} alert(s) matching these filters")

if filtered.empty:
    dc.empty_panel(empty_state("no_alerts"))
    st.stop()


def _style_severity(frame):
    return frame.style.map(
        lambda value: f"background-color: {severity_color(value)}22; color: {severity_color(value)};",
        subset=["severity"],
    )


display_columns = [
    "severity", "category", "title", "feature", "batch_id",
    "occurrence_count", "first_seen_at", "last_seen_at", "description", "reason",
]
st.dataframe(
    _style_severity(filtered[display_columns].sort_values("last_seen_at", ascending=False)),
    width="stretch",
    hide_index=True,
)
