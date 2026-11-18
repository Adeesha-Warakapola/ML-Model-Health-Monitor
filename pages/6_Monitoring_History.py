"""Monitoring History page: previous batches and historical results, filtered.

Everything here comes from `MonitoringRepository`, independent of whether
this dashboard process has run a single batch itself - this is the page
that most directly demonstrates "monitoring history remains available after
restarting the application."
"""

import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard_common as dc
from monitor.alerts import CRITICAL, INFO, WARNING
from monitor.dashboard_data import (
    empty_state,
    overall_drift_status_from_frame,
    performance_status_label,
    worse_status,
)
from monitor.performance_monitoring import NO_LABELS

dc.configure_page("Monitoring History")
context = dc.render_sidebar()
repository = dc.get_repository()

dc.page_header("Monitoring History", "Every monitoring run saved to this database, with filters.")

runs = repository.get_runs()
if runs.empty:
    dc.empty_panel(empty_state("no_history"))
    st.stop()

min_date = datetime.fromisoformat(runs["started_at"].min()).date()
max_date = datetime.fromisoformat(runs["started_at"].max()).date()
date_range = st.date_input("Started between", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_day, end_day = date_range
else:
    start_day = end_day = date_range[0] if isinstance(date_range, tuple) else date_range

runs = repository.get_runs(
    start_date=datetime.combine(start_day, time.min).isoformat(),
    end_date=datetime.combine(end_day, time.max).isoformat(),
)

if runs.empty:
    dc.empty_panel("No runs fall in the selected date range.")
    st.stop()

# Build one combined row per run, from each table's already-persisted summary
# - no drift, performance, or alert logic runs again here.
drift_all = repository.get_drift_results()
performance_all = repository.get_performance_metrics()
alerts_all = repository.get_alerts()

rows = []
for _, run in runs.iterrows():
    batch_id = run["batch_id"]
    batch_drift = drift_all[drift_all["batch_id"] == batch_id] if not drift_all.empty else drift_all
    batch_perf = performance_all[performance_all["batch_id"] == batch_id] if not performance_all.empty else performance_all
    batch_alerts = alerts_all[alerts_all["batch_id"] == batch_id] if not alerts_all.empty else alerts_all

    perf_status = batch_perf.iloc[0]["status"] if not batch_perf.empty else NO_LABELS
    rows.append(
        {
            "run_id": run["run_id"],
            "batch_id": batch_id,
            "started_at": run["started_at"],
            "status": run["status"],
            "data_drift": overall_drift_status_from_frame(batch_drift),
            "performance": performance_status_label(perf_status),
            "n_critical": int((batch_alerts["severity"] == CRITICAL).sum()) if not batch_alerts.empty else 0,
            "n_warning": int((batch_alerts["severity"] == WARNING).sum()) if not batch_alerts.empty else 0,
            "n_info": int((batch_alerts["severity"] == INFO).sum()) if not batch_alerts.empty else 0,
        }
    )

history_df = pd.DataFrame(rows).sort_values("started_at", ascending=False)

dc.section_label(f"{len(history_df)} run(s)")
st.dataframe(
    history_df[
        ["started_at", "batch_id", "status", "data_drift", "performance", "n_critical", "n_warning", "n_info"]
    ],
    width="stretch",
    hide_index=True,
)

dc.section_label("Alerts per run (severity breakdown)")
trend = history_df.sort_values("started_at")
fig = go.Figure()
fig.add_bar(name="Critical", x=trend["started_at"], y=trend["n_critical"], marker_color="#e8545c")
fig.add_bar(name="Warning", x=trend["started_at"], y=trend["n_warning"], marker_color="#e8b339")
fig.add_bar(name="Info", x=trend["started_at"], y=trend["n_info"], marker_color="#4f8ff0")
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig, width="stretch")

dc.section_label("Jump to a run")
chosen = st.selectbox("Batch", history_df["batch_id"].tolist(), key="history_jump_batch")
if st.button("View this batch on the Overview page", width="content"):
    st.session_state["latest_batch_id"] = chosen
    st.switch_page("app.py")
