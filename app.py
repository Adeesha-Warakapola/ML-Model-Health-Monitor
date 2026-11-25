"""Overview page - also the Streamlit entry point.

Run with: streamlit run app.py

Shows overall monitoring status, the latest batch, feature drift summary,
prediction changes, and model performance when labels are available. Every
number on this page comes from `MonitoringRepository` (survives a restart)
or, where noted, from the live report objects for the batch just run this
session - see `dashboard_common.live_result_for`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import plotly.graph_objects as go
import streamlit as st

import dashboard_common as dc
from monitor.dashboard_data import (
    drift_status_color,
    drift_status_label,
    empty_state,
    format_metric,
    format_timestamp,
    overall_drift_status_from_frame,
    parse_json_field,
    performance_status_color,
    performance_status_label,
    severity_color,
    severity_counts,
    worse_status,
)
from monitor.performance_monitoring import NO_LABELS

dc.configure_page("Overview")
context = dc.render_sidebar()
repository = dc.get_repository()

dc.page_header(
    "ML Model Health Monitor",
    "Overview — data quality, drift, predictions, and performance for one batch at a glance.",
)

batch_id = dc.select_batch_id(repository, key="overview_batch")
if batch_id is None:
    dc.empty_panel(empty_state("no_history"))
    st.stop()

runs = repository.get_runs()
run_row = runs[runs["batch_id"] == batch_id].iloc[0]
summary_df = repository.get_batch_summaries(batch_id=batch_id)
drift_df = repository.get_drift_results(batch_id=batch_id)
pred_df = repository.get_prediction_drift_results(batch_id=batch_id)
perf_df = repository.get_performance_metrics(batch_id=batch_id)
alerts_df = repository.get_alerts(batch_id=batch_id)

drift_status = overall_drift_status_from_frame(drift_df)
pred_row = pred_df.iloc[0] if not pred_df.empty else None
prediction_status = worse_status(
    pred_row["label_status"] if pred_row is not None else None,
    pred_row["confidence_status"] if pred_row is not None else None,
)
perf_row = perf_df.iloc[0] if not perf_df.empty else None
performance_status = perf_row["status"] if perf_row is not None else NO_LABELS
n_rows = int(summary_df.iloc[0]["n_rows"]) if not summary_df.empty else None

# --- top facts ---------------------------------------------------------------

st.caption(f"Batch `{batch_id}`")
cols = st.columns(3)
cols[0].metric("Monitored at (UTC)", format_timestamp(run_row["started_at"])[:19])
cols[1].metric("Rows in batch", f"{n_rows:,}" if n_rows is not None else "Not available")
severities = severity_counts(alerts_df)
cols[2].metric("Alerts for this batch", len(alerts_df), help="CRITICAL + WARNING + INFO, after deduplication")

st.markdown(
    " ".join(
        [
            dc.status_chip(f"Data drift: {drift_status_label(drift_status)}", drift_status_color(drift_status)),
            dc.status_chip(
                f"Prediction drift: {drift_status_label(prediction_status)}",
                drift_status_color(prediction_status),
            ),
            dc.status_chip(
                f"Performance: {performance_status_label(performance_status)}",
                performance_status_color(performance_status),
            ),
            dc.status_chip(f"{severities['CRITICAL']} critical", severity_color("CRITICAL")),
            dc.status_chip(f"{severities['WARNING']} warning", severity_color("WARNING")),
            dc.status_chip(f"{severities['INFO']} info", severity_color("INFO")),
        ]
    ),
    unsafe_allow_html=True,
)

st.markdown("<br/>", unsafe_allow_html=True)

left, right = st.columns([3, 2])

# --- feature drift summary ---------------------------------------------------

with left:
    dc.section_label("Feature drift summary")
    if drift_df.empty:
        dc.empty_panel("No drift results recorded for this batch.")
    else:
        ranked = drift_df.sort_values("psi", ascending=False, na_position="last").head(12)
        color_map = {status: drift_status_color(status) for status in ranked["status"].unique()}
        fig = go.Figure()
        fig.add_bar(
            x=ranked["feature"],
            y=ranked["psi"],
            marker_color=[color_map[s] for s in ranked["status"]],
            text=[drift_status_label(s) for s in ranked["status"]],
            hovertemplate="%{x}<br>PSI=%{y:.4f}<br>%{text}<extra></extra>",
        )
        fig.add_hline(y=0.10, line_dash="dot", line_color="#6c7585", annotation_text="moderate")
        fig.add_hline(y=0.25, line_dash="dot", line_color="#6c7585", annotation_text="major")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="PSI",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

# --- prediction changes -------------------------------------------------------

with right:
    dc.section_label("Prediction changes")
    if pred_row is None:
        dc.empty_panel("No prediction-monitoring result recorded for this batch.")
    else:
        reference_dist = context.prediction_monitor.reference_label_distribution()
        batch_dist = parse_json_field(pred_row["batch_label_distribution_json"], default={})
        labels = sorted(set(reference_dist) | set(batch_dist))
        fig = go.Figure()
        fig.add_bar(
            name="Reference",
            x=labels,
            y=[reference_dist.get(label, {}).get("percent", 0) for label in labels],
            marker_color="#4f8ff0",
        )
        fig.add_bar(
            name="This batch",
            x=labels,
            y=[batch_dist.get(label, {}).get("percent", 0) for label in labels],
            marker_color="#5fd0c0",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="% of predictions",
            barmode="group",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Label PSI {format_metric(pred_row['label_psi'])} · "
            f"Confidence PSI {format_metric(pred_row['confidence_psi'])}"
        )

# --- model performance ---------------------------------------------------------

dc.section_label("Model performance")
if performance_status == NO_LABELS or perf_row is None:
    dc.empty_panel(empty_state("no_labels"))
else:
    if performance_status == "insufficient_labels":
        st.info(
            f"Based on only {int(perf_row['n_labeled'])} labelled rows - below the "
            "reliability floor. Treat as an early signal, not a confirmed result.",
            icon="⚠️",
        )
    metric_cols = st.columns(5)
    comparison = parse_json_field(perf_row["comparison_json"], default={})
    for col, (label, key) in zip(
        metric_cols,
        [
            ("Accuracy", "accuracy"), ("Precision", "precision_score"),
            ("Recall", "recall"), ("F1 score", "f1_score"), ("ROC-AUC", "roc_auc"),
        ],
    ):
        value = perf_row.get(key)
        compare_key = "precision" if key == "precision_score" else key
        delta = comparison.get(compare_key, {}).get("delta") if comparison else None
        col.metric(
            label,
            format_metric(value),
            delta=f"{delta:+.4f} vs baseline" if delta is not None else None,
            delta_color="inverse" if delta is not None and delta < 0 else "normal",
        )
