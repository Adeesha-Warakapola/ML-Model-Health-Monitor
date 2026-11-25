"""Prediction Monitoring page: what the model is saying, as distinct from the input.

The predicted-label mix and the confidence summary (mean/std/median/
quartiles/borderline rate) are both persisted per batch, so the comparisons
on this page work the same for any historical batch. Only the full
confidence histogram - built from every individual prediction's probability
- needs the live batch, so it only appears for the batch most recently run
in this session.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import plotly.graph_objects as go
import streamlit as st

import dashboard_common as dc
from monitor.dashboard_data import (
    drift_status_color,
    drift_status_label,
    empty_state,
    format_metric,
    parse_json_field,
)

dc.configure_page("Prediction Monitoring")
context = dc.render_sidebar()
repository = dc.get_repository()

dc.page_header(
    "Prediction Monitoring",
    "What the model predicted on this batch, compared with its reference behaviour.",
)

batch_id = dc.select_batch_id(repository, key="prediction_batch")
if batch_id is None:
    dc.empty_panel(empty_state("no_history"))
    st.stop()

pred_df = context.repository.get_prediction_drift_results(batch_id=batch_id)
if pred_df.empty:
    dc.empty_panel("No prediction-monitoring result recorded for this batch.")
    st.stop()

row = pred_df.iloc[0]
st.markdown(
    " ".join(
        [
            dc.status_chip(
                f"Predicted-label mix: {drift_status_label(row['label_status'])}",
                drift_status_color(row["label_status"]),
            ),
            dc.status_chip(
                f"Confidence: {drift_status_label(row['confidence_status'])}",
                drift_status_color(row["confidence_status"]),
            ),
        ]
    ),
    unsafe_allow_html=True,
)

notes = parse_json_field(row["notes_json"], default=[])
for note in notes:
    st.caption(f"📝 {note}")

reference_dist = context.prediction_monitor.reference_label_distribution()
batch_dist = parse_json_field(row["batch_label_distribution_json"], default={})

left, right = st.columns(2)

with left:
    dc.section_label("Predicted class distribution")
    labels = sorted(set(reference_dist) | set(batch_dist))
    fig = go.Figure()
    fig.add_bar(name="Reference", x=labels, y=[reference_dist.get(l, {}).get("percent", 0) for l in labels], marker_color="#4f8ff0")
    fig.add_bar(name="This batch", x=labels, y=[batch_dist.get(l, {}).get("percent", 0) for l in labels], marker_color="#5fd0c0")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="% of predictions",
        barmode="group", legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Label PSI {format_metric(row['label_psi'])}")

with right:
    dc.section_label("Prediction counts")
    counts_table = [
        {"label": label, "reference %": reference_dist.get(label, {}).get("percent"),
         "batch count": batch_dist.get(label, {}).get("count"), "batch %": batch_dist.get(label, {}).get("percent")}
        for label in labels
    ]
    st.dataframe(counts_table, width="stretch", hide_index=True)

dc.section_label("Confidence distribution (predicted probability of the positive class)")
reference_confidence = context.prediction_monitor.reference_confidence_summary()
batch_confidence = parse_json_field(row["batch_confidence_json"], default={})

if not batch_confidence or batch_confidence.get("n", 0) == 0:
    dc.empty_panel("Probabilities were unavailable for this batch, so confidence drift could not be assessed.")
else:
    compare_cols = st.columns(5)
    for col, key, label in zip(
        compare_cols,
        ["mean", "std", "median", "q25", "borderline_rate"],
        ["Mean", "Std dev", "Median", "25th pct.", "Borderline rate"],
    ):
        col.metric(
            label,
            format_metric(batch_confidence.get(key)),
            delta=(
                f"ref {format_metric(reference_confidence.get(key))}"
                if reference_confidence.get(key) is not None
                else None
            ),
            delta_color="off",
        )
    st.caption(f"Confidence PSI {format_metric(row['confidence_psi'])} · n={batch_confidence.get('n')}")

    live = dc.live_result_for(batch_id)
    if live is None or live["prediction"].probabilities is None:
        dc.empty_panel(empty_state("no_raw_distribution"))
    else:
        reference_probabilities = context.prediction_monitor.reference_probabilities
        hist = go.Figure()
        hist.add_histogram(x=reference_probabilities, name="Reference", histnorm="probability", marker_color="#4f8ff0", opacity=0.6)
        hist.add_histogram(x=live["prediction"].probabilities, name="This batch", histnorm="probability", marker_color="#5fd0c0", opacity=0.6)
        hist.add_vline(x=live["prediction"].threshold, line_dash="dot", line_color="#e8b339", annotation_text="decision threshold")
        hist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="overlay", height=320, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(hist, width="stretch")
