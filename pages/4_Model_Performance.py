"""Model Performance page: accuracy once true labels arrive, plus history.

`performance_metrics.status` is the three-way answer requirement 14 asks
for: `no_labels` (nothing shown as a result - not a zero),
`insufficient_labels` (shown, but flagged as an early signal), or `computed`
(a confirmed result). The full confusion matrix and per-class report are
persisted as JSON, so they are available for any historical run, not only
the one most recently run in this session.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import plotly.graph_objects as go
import streamlit as st

import dashboard_common as dc
from monitor.dashboard_data import empty_state, format_metric, performance_status_color, performance_status_label
from monitor.performance_monitoring import NO_LABELS

dc.configure_page("Model Performance")
context = dc.render_sidebar()
repository = dc.get_repository()

dc.page_header(
    "Model Performance",
    "Accuracy, precision, recall, F1 and ROC-AUC once true labels are available - never fabricated.",
)

batch_id = dc.select_batch_id(repository, key="performance_batch")
if batch_id is None:
    dc.empty_panel(empty_state("no_history"))
    st.stop()

perf_df = repository.get_performance_metrics(batch_id=batch_id)
if perf_df.empty:
    dc.empty_panel("No performance result recorded for this batch.")
    st.stop()

row = perf_df.iloc[0]
status = row["status"]

st.markdown(
    dc.status_chip(performance_status_label(status), performance_status_color(status)),
    unsafe_allow_html=True,
)

if status == NO_LABELS:
    dc.empty_panel(empty_state("no_labels"))
    st.stop()

if status == "insufficient_labels":
    st.info(
        f"Based on only {int(row['n_labeled'])} labelled rows - below the reliability "
        "floor measured for this project. Treat this as an early signal, not a confirmed result.",
        icon="⚠️",
    )

comparison = json.loads(row["comparison_json"]) if row.get("comparison_json") else {}
metric_cols = st.columns(5)
for col, label, key in zip(
    metric_cols,
    ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"],
    ["accuracy", "precision_score", "recall", "f1_score", "roc_auc"],
):
    compare_key = "precision" if key == "precision_score" else key
    delta = comparison.get(compare_key, {}).get("delta") if comparison else None
    col.metric(
        label,
        format_metric(row.get(key)),
        delta=f"{delta:+.4f} vs baseline" if delta is not None else None,
        delta_color="inverse" if delta is not None and delta < 0 else "normal",
    )

metrics_payload = json.loads(row["metrics_json"]) if row.get("metrics_json") else None

dc.section_label("Confusion matrix")
if not metrics_payload or "confusion_matrix" not in metrics_payload:
    dc.empty_panel(empty_state("no_confusion_matrix"))
else:
    labels = metrics_payload["confusion_matrix_labels"]
    matrix = metrics_payload["confusion_matrix"]
    heat = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"predicted {l}" for l in labels],
            y=[f"actual {l}" for l in labels],
            colorscale=[[0, "#131824"], [1, "#5fd0c0"]],
            text=matrix,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    heat.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(heat, width="stretch")
    with st.expander("Per-class classification report"):
        st.code(metrics_payload.get("classification_report", "Not available"), language=None)

dc.section_label("Performance trend across all runs")
history = repository.get_performance_metrics()
if len(history) < 2:
    dc.empty_panel("Run at least two batches to see a trend here.")
else:
    history = history.sort_values("recorded_at")
    trend = go.Figure()
    for key, label, color in [
        ("accuracy", "Accuracy", "#5fd0c0"), ("recall", "Recall", "#e8b339"),
        ("f1_score", "F1", "#4f8ff0"), ("roc_auc", "ROC-AUC", "#e8545c"),
    ]:
        trend.add_scatter(
            x=history["recorded_at"], y=history[key], mode="lines+markers", name=label,
            line=dict(color=color), marker=dict(size=6),
        )
    trend.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(trend, width="stretch")
    st.caption(
        "Hollow or missing points indicate a batch where a metric could not be "
        "computed (for example, ROC-AUC with only one labelled class present)."
    )
