"""Data Drift page: feature-level numerical and categorical drift.

`drift_results` persists every feature's result for every run, so the table
and PSI chart below are restart-safe. The one thing not persisted is the raw
reference/batch values themselves - only their summary statistics - so the
per-feature value-distribution chart only appears for the batch most
recently run in this dashboard session (`dashboard_common.live_result_for`);
older batches fall back to the saved PSI/statistic/p-value table.
"""

import json
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
    format_psi,
    overall_drift_status_from_frame,
)

dc.configure_page("Data Drift")
context = dc.render_sidebar()
repository = dc.get_repository()

dc.page_header(
    "Data Drift",
    "Reference-vs-current comparison for every feature - PSI, KS/chi-square, and effect size.",
)

batch_id = dc.select_batch_id(repository, key="drift_batch")
if batch_id is None:
    dc.empty_panel(empty_state("no_history"))
    st.stop()

drift_df = repository.get_drift_results(batch_id=batch_id)
if drift_df.empty:
    dc.empty_panel("No drift results recorded for this batch.")
    st.stop()

overall = overall_drift_status_from_frame(drift_df)
st.markdown(
    dc.status_chip(f"Overall: {drift_status_label(overall)}", drift_status_color(overall)),
    unsafe_allow_html=True,
)

type_filter = st.radio("Feature type", ["All", "Numeric", "Categorical"], horizontal=True, key="drift_type_filter")
filtered = drift_df
if type_filter != "All":
    filtered = drift_df[drift_df["feature_type"] == type_filter.lower()]

dc.section_label("PSI by feature")
ranked = filtered.sort_values("psi", ascending=False, na_position="last")
color_map = {status: drift_status_color(status) for status in ranked["status"].unique()}
fig = go.Figure()
fig.add_bar(
    x=ranked["feature"],
    y=ranked["psi"],
    marker_color=[color_map[s] for s in ranked["status"]],
    text=[drift_status_label(s) for s in ranked["status"]],
    hovertemplate="%{x}<br>PSI=%{y:.4f}<br>%{text}<extra></extra>",
)
fig.add_hline(y=0.10, line_dash="dot", line_color="#6c7585", annotation_text="moderate (0.10)")
fig.add_hline(y=0.25, line_dash="dot", line_color="#6c7585", annotation_text="major (0.25)")
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="PSI", showlegend=False,
)
st.plotly_chart(fig, width="stretch")

dc.section_label("Every feature's statistical result")
display = filtered.copy()
display["psi"] = display["psi"].apply(format_psi)
display["statistic"] = display["statistic"].apply(format_metric)
display["p_value"] = display["p_value"].apply(lambda v: format_metric(v, digits=2))
display["effect_size"] = display["effect_size"].apply(format_metric)
st.dataframe(
    display[
        ["feature", "feature_type", "method", "status", "psi", "statistic", "p_value",
         "effect_size", "n_reference", "n_batch"]
    ],
    width="stretch",
    hide_index=True,
)

dc.section_label("Feature detail")
selected_feature = st.selectbox("Feature", sorted(filtered["feature"].unique()), key="drift_feature_select")
feature_row = drift_df[drift_df["feature"] == selected_feature].iloc[0]

notes = json.loads(feature_row["notes_json"]) if feature_row.get("notes_json") else []
if notes:
    for note in notes:
        st.info(note, icon="📝")
else:
    st.caption("No notes recorded for this feature on this batch.")

live = dc.live_result_for(batch_id)
if live is None:
    dc.empty_panel(empty_state("no_raw_distribution"))
else:
    reference_values = context.drift_detector.reference[selected_feature]
    batch_values = live["batch"].features[selected_feature]
    if feature_row["feature_type"] == "numeric":
        hist = go.Figure()
        hist.add_histogram(x=reference_values, name="Reference", histnorm="probability", marker_color="#4f8ff0", opacity=0.6)
        hist.add_histogram(x=batch_values, name="This batch", histnorm="probability", marker_color="#5fd0c0", opacity=0.6)
        hist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="overlay", height=320, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(hist, width="stretch")
    else:
        ref_counts = reference_values.value_counts(normalize=True).head(12)
        batch_counts = batch_values.value_counts(normalize=True).reindex(ref_counts.index, fill_value=0)
        categories = ref_counts.index.tolist()
        bar = go.Figure()
        bar.add_bar(name="Reference", x=categories, y=ref_counts.values, marker_color="#4f8ff0")
        bar.add_bar(name="This batch", x=categories, y=batch_counts.values, marker_color="#5fd0c0")
        bar.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(bar, width="stretch")
        st.caption("Top 12 reference categories shown; rarer categories are omitted from this chart only.")
