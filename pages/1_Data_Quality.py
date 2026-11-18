"""Data Quality page: incoming batch validation results.

Every validation issue - a missing-column error, a dtype mismatch, missing
values, an unknown category, an out-of-range value - becomes exactly one
alert in `alerts_from_validation` (`src/monitor/alerts.py`), regardless of
severity. That means the `alerts` table, filtered to
`category="data_quality"`, already holds the complete validation detail for
any batch ever run - this page does not need the live `ValidationResult`
object, and works the same right after a restart.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

import dashboard_common as dc
from monitor.dashboard_data import empty_state, severity_color, severity_counts

dc.configure_page("Data Quality")
context = dc.render_sidebar()
repository = dc.get_repository()

dc.page_header(
    "Data Quality",
    "Validation findings for one batch - missing values, invalid values, and schema errors.",
)

batch_id = dc.select_batch_id(repository, key="quality_batch")
if batch_id is None:
    dc.empty_panel(empty_state("no_history"))
    st.stop()

quality_alerts = repository.get_alerts(batch_id=batch_id, category="data_quality")

if quality_alerts.empty:
    st.success("No data-quality issues were raised for this batch.", icon="✅")
    st.stop()

counts = severity_counts(quality_alerts)
cols = st.columns(3)
cols[0].metric("Critical (schema errors)", counts["CRITICAL"])
cols[1].metric("Warnings", counts["WARNING"])
cols[2].metric("Info", counts["INFO"])

dc.section_label("Issues found")

for _, row in quality_alerts.sort_values("severity", ascending=False).iterrows():
    color = severity_color(row["severity"])
    with st.container(border=True):
        st.markdown(
            f"{dc.status_chip(row['severity'], color)} &nbsp; **{row['title']}**"
            + (f" &nbsp; <code>{row['feature']}</code>" if row["feature"] else ""),
            unsafe_allow_html=True,
        )
        st.write(row["description"])
        st.caption(f"Reason: {row['reason']}")
        st.caption(
            f"First seen {row['first_seen_at']} · last seen {row['last_seen_at']} · "
            f"seen in {row['occurrence_count']} run(s) so far"
        )

dc.section_label("Raw table")
st.dataframe(
    quality_alerts[
        ["severity", "title", "feature", "reason", "first_seen_at", "last_seen_at", "occurrence_count"]
    ],
    width="stretch",
    hide_index=True,
)
