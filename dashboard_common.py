"""Streamlit/Plotly glue shared by every dashboard page.

Every monitoring calculation stays in `src/monitor/*`. This module only:

1. builds the services every page needs, once per process
   (`get_context`, cached with `st.cache_resource` so re-running a page
   never reloads the dataset or rescales the reference predictions);
2. wires the sidebar's "run a new batch" control to the existing
   `monitor.monitoring_cycle.run_monitoring_cycle` - the same function
   `manage_alerts.py` uses, so the dashboard can never run the pipeline
   differently than the CLI does;
3. holds the handful of look-and-feel helpers (theme CSS, status chips,
   page headers) every page shares, so seven page files do not each
   reinvent them.

No drift statistic, metric, or alert rule is computed here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from monitor import config
from monitor.dashboard_data import (
    BATCH_SIZE_STEP,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SCENARIO,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    SCENARIOS,
)
from monitor.monitoring_cycle import MonitoringContext, build_monitoring_context, run_monitoring_cycle
from monitor.storage import MonitoringRepository

APP_TITLE = "ML Model Health Monitor"
APP_ICON = "◆"


# --- page scaffolding ---------------------------------------------------------


def configure_page(title: str) -> None:
    """Call once, as the very first Streamlit call, at the top of every page."""
    st.set_page_config(
        page_title=f"{title} · {APP_TITLE}",
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="mhm-page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="mhm-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown("<hr class='mhm-hr'/>", unsafe_allow_html=True)


def section_label(text: str) -> None:
    """A small, uppercase section marker - the project's recurring visual rhythm."""
    st.markdown(f'<div class="mhm-section-label">{text}</div>', unsafe_allow_html=True)


def status_chip(label: str, color: str) -> str:
    """An HTML pill for a status/severity value. Compose into any `st.markdown` call."""
    return (
        '<span class="mhm-chip" style="'
        f'background:{color}1f;color:{color};border:1px solid {color}66;">'
        f"{label}</span>"
    )


def empty_panel(message: str) -> None:
    st.markdown(f'<div class="mhm-empty">{message}</div>', unsafe_allow_html=True)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        div.block-container { padding-top: 1.6rem; max-width: 1400px; }

        .mhm-page-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            color: #eef1f5;
        }
        .mhm-page-subtitle {
            color: #8a93a3;
            font-size: 0.92rem;
            margin-top: 0.15rem;
        }
        .mhm-hr {
            border: none;
            border-top: 1px solid #262c38;
            margin: 0.9rem 0 1.4rem 0;
        }
        .mhm-section-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6c7585;
            margin: 0.6rem 0 0.4rem 0;
        }
        .mhm-chip {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 3px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .mhm-empty {
            border: 1px dashed #30374;
            border-color: #323a48;
            border-radius: 6px;
            padding: 1.1rem 1.2rem;
            color: #8a93a3;
            font-size: 0.92rem;
            background: #10141c;
        }

        /* Metrics: monospace values, tighter card */
        div[data-testid="stMetric"] {
            background: #10141c;
            border: 1px solid #222838;
            border-radius: 6px;
            padding: 0.7rem 0.9rem 0.6rem 0.9rem;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.74rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #8a93a3;
        }

        /* Sidebar: uppercase small-caps labels, tighter spacing */
        section[data-testid="stSidebar"] div.block-container { padding-top: 1.2rem; }
        section[data-testid="stSidebar"] h2 {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #aab3c2;
        }

        code, .stCode, pre { font-family: 'JetBrains Mono', monospace !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- shared services and session state -----------------------------------------


@st.cache_resource(show_spinner="Loading model, reference data, and monitoring history...")
def get_context() -> MonitoringContext:
    """Build the monitoring context exactly once per running dashboard process.

    This loads the dataset and scores the reference batch once; it never
    trains or retrains the model - `PredictionService` loads the artifact
    `train_baseline.py` already saved.
    """
    return build_monitoring_context()


def get_repository() -> MonitoringRepository:
    return get_context().repository


def render_sidebar() -> MonitoringContext:
    """The "run a new batch" control, shared by every page's sidebar."""
    context = get_context()

    st.sidebar.markdown("## Run monitoring")
    scenario_names = list(SCENARIOS.keys())
    scenario_name = st.sidebar.selectbox(
        "Demonstration scenario",
        scenario_names,
        index=scenario_names.index(DEFAULT_SCENARIO),
        key="scenario_select",
    )
    st.sidebar.caption(SCENARIOS[scenario_name].description)
    batch_size = st.sidebar.slider(
        "Batch size",
        min_value=MIN_BATCH_SIZE,
        max_value=MAX_BATCH_SIZE,
        value=DEFAULT_BATCH_SIZE,
        step=BATCH_SIZE_STEP,
        key="batch_size_select",
    )

    run_clicked = st.sidebar.button("Run monitoring batch", type="primary", width="stretch")
    reset_clicked = st.sidebar.button("Reset batch simulator", width="stretch")

    if reset_clicked:
        context.simulator.reset()
        st.sidebar.success("Simulator reset to the start of the holdout pool.")

    if run_clicked:
        scenario = SCENARIOS[scenario_name]
        try:
            batch = context.simulator.next_batch(
                size=batch_size,
                modifier=scenario.modifier,
                modification_label=scenario.label if scenario.modifier else None,
            )
        except ValueError as error:
            st.sidebar.error(f"Could not draw a batch: {error}")
        else:
            with st.spinner("Scoring batch, checking drift, updating alerts..."):
                result = run_monitoring_cycle(batch, context)
            st.session_state["latest_result"] = result
            st.session_state["latest_batch_id"] = result["batch_id"]
            st.sidebar.success(f"Batch {result['batch_id']} processed and saved.")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Database: `{config.MONITORING_DB_PATH.name}`")
    st.sidebar.caption(
        f"{context.simulator.rows_remaining:,} holdout rows left before the "
        "simulator loops back to the start."
    )
    if st.session_state.get("latest_batch_id"):
        st.sidebar.caption(f"Latest batch this session: `{st.session_state['latest_batch_id']}`")

    return context


def latest_batch_id(repository: MonitoringRepository) -> str | None:
    """The batch to show by default: this session's latest run, or else the newest in the database."""
    if st.session_state.get("latest_batch_id"):
        return st.session_state["latest_batch_id"]
    runs = repository.get_runs()
    if runs.empty:
        return None
    return runs.iloc[0]["batch_id"]


def select_batch_id(repository: MonitoringRepository, key: str) -> str | None:
    """A batch-picker widget, defaulting to the most recent batch. Returns None if history is empty."""
    runs = repository.get_runs()
    if runs.empty:
        return None
    options = runs["batch_id"].tolist()
    default_batch = latest_batch_id(repository)
    default_index = options.index(default_batch) if default_batch in options else 0
    return st.selectbox("Batch", options, index=default_index, key=key)


def live_result_for(batch_id: str | None) -> dict | None:
    """This session's full in-memory result dict, if it matches the selected batch.

    Returns None for any batch not just run in this session (including after
    a dashboard restart), in which case a page should fall back to what the
    repository persisted - which is exactly what "independent of the
    dashboard" and "survives a restart" require.
    """
    result = st.session_state.get("latest_result")
    if result and batch_id and result["batch_id"] == batch_id:
        return result
    return None
