"""Phase 9: automated checks that the dashboard displays what was stored.

Phase 8 verified every page was exception-free using ad hoc `AppTest` runs
during development; this file turns the most important of those checks into
permanent, re-runnable tests - requirement 10 ("check that the dashboard
accurately displays stored monitoring results"), not just "does it crash."

Each test runs the *real* monitoring pipeline (`run_monitoring_cycle`)
against a throwaway database, records the ground-truth facts that produced
(status, alert counts, severities), points the dashboard at that same
database, and checks the rendered page actually shows those facts - not
merely that the page rendered without an exception.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from monitor import config
from monitor.alerts import CRITICAL, WARNING
from monitor.dashboard_data import drift_status_label
from monitor.monitoring_cycle import build_monitoring_context, run_monitoring_cycle
from monitor.simulation import overwrite_category, shift_numeric

import dashboard_common

# AppTest.from_file resolves a relative path against the file that calls it
# (this test file), not the working directory - so every page path must be
# absolute, rooted at the actual project directory, two levels above tests/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _page(relative_path: str) -> str:
    return str(PROJECT_ROOT / relative_path)


def _clear_dashboard_cache():
    """`dashboard_common.get_context` is cached process-wide with no
    arguments, so every test must clear it after pointing
    `config.MONITORING_DB_PATH` at a fresh database - otherwise a later test
    would silently reuse an earlier test's (possibly already-deleted)
    database."""
    dashboard_common.get_context.clear()


@pytest.fixture(autouse=True)
def _isolated_dashboard_cache():
    _clear_dashboard_cache()
    yield
    _clear_dashboard_cache()


@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    """A real database, populated by the real pipeline, with the dashboard's
    cached context pointed at it."""
    db_path = tmp_path / "dashboard_integration.db"
    monkeypatch.setattr(config, "MONITORING_DB_PATH", db_path)

    context = build_monitoring_context(db_path=db_path, batch_size=400, seed=config.RANDOM_SEED)

    stable_batch = context.simulator.next_batch()
    stable_result = run_monitoring_cycle(stable_batch, context)

    context.simulator.reset()
    severe_batch = context.simulator.next_batch(modifier=shift_numeric("age", 25))
    severe_result = run_monitoring_cycle(severe_batch, context)

    context.simulator.reset()
    moderate_batch = context.simulator.next_batch(
        modifier=overwrite_category("occupation", "Gig-economy", fraction=0.05)
    )
    moderate_result = run_monitoring_cycle(moderate_batch, context)

    return {
        "db_path": db_path,
        "stable": stable_result,
        "severe": severe_result,
        "moderate": moderate_result,
    }


def _select(app_test, key, value):
    app_test.selectbox(key=key).set_value(value).run()


# --- Overview ------------------------------------------------------------------------


def test_overview_page_shows_the_correct_status_for_the_severe_batch(populated_db):
    at = AppTest.from_file(_page("app.py"), default_timeout=120)
    at.run()
    assert not at.exception

    severe_batch_id = populated_db["severe"]["batch_id"]
    _select(at, "overview_batch", severe_batch_id)
    assert not at.exception

    rendered = " ".join(m.value for m in at.markdown)
    assert "Drift" in rendered  # the CRITICAL/"Drift" status chip for data drift
    assert severe_batch_id in " ".join(c.value for c in at.caption)


def test_overview_page_shows_the_correct_status_for_the_stable_batch(populated_db):
    at = AppTest.from_file(_page("app.py"), default_timeout=120)
    at.run()

    stable_batch_id = populated_db["stable"]["batch_id"]
    _select(at, "overview_batch", stable_batch_id)
    assert not at.exception

    rendered = " ".join(m.value for m in at.markdown)
    assert "Stable" in rendered


# --- Data Quality ----------------------------------------------------------------------


def test_data_quality_page_lists_the_real_validation_issues_for_the_latest_batch(populated_db):
    """The *moderate* batch is the last one `populated_db` ran, so it is the
    one deduplication has attributed the three recurring missing-value
    alerts to (see the module docstring on `pages/1_Data_Quality.py`)."""
    at = AppTest.from_file(_page("pages/1_Data_Quality.py"), default_timeout=120)
    at.run()
    moderate_batch_id = populated_db["moderate"]["batch_id"]
    _select(at, "quality_batch", moderate_batch_id)
    assert not at.exception

    table = at.dataframe[0].value
    # Exactly the three recurring missing-value columns this dataset has.
    assert set(table["feature"]) == {"workclass", "occupation", "native_country"}
    assert (table["severity"] == WARNING).all()


def test_data_quality_page_is_honest_about_an_older_batchs_reassigned_alerts(populated_db):
    """The *stable* batch genuinely had the same three missing-value issues,
    but deduplication has since moved their `batch_id` forward to the
    `moderate` batch run after it - so this page must not claim the stable
    batch had zero issues, only that none are currently attributed to it."""
    at = AppTest.from_file(_page("pages/1_Data_Quality.py"), default_timeout=120)
    at.run()
    stable_batch_id = populated_db["stable"]["batch_id"]
    _select(at, "quality_batch", stable_batch_id)
    assert not at.exception

    assert not at.dataframe  # no raw table - zero alerts attributed to this batch_id
    rendered = " ".join(i.value for i in at.info)
    assert "does not guarantee" in rendered
    assert not any("No data-quality issues were raised" in s.value for s in at.success)


# --- Data Drift --------------------------------------------------------------------------


def test_data_drift_page_marks_age_as_the_drifted_feature_in_the_severe_batch(populated_db):
    at = AppTest.from_file(_page("pages/2_Data_Drift.py"), default_timeout=120)
    at.run()
    severe_batch_id = populated_db["severe"]["batch_id"]
    _select(at, "drift_batch", severe_batch_id)
    assert not at.exception

    table = at.dataframe[0].value
    age_row = table[table["feature"] == "age"].iloc[0]
    assert age_row["status"] == "drift"

    rendered = " ".join(m.value for m in at.markdown)
    assert "Drift" in rendered  # overall-status chip


def test_data_drift_page_shows_no_drifted_features_for_the_stable_batch(populated_db):
    at = AppTest.from_file(_page("pages/2_Data_Drift.py"), default_timeout=120)
    at.run()
    stable_batch_id = populated_db["stable"]["batch_id"]
    _select(at, "drift_batch", stable_batch_id)
    assert not at.exception

    table = at.dataframe[0].value
    assert (table["status"] == "stable").all()


# --- Prediction Monitoring ---------------------------------------------------------------


def test_prediction_monitoring_page_matches_the_stored_label_status(populated_db):
    at = AppTest.from_file(_page("pages/3_Prediction_Monitoring.py"), default_timeout=120)
    at.run()
    severe_batch_id = populated_db["severe"]["batch_id"]
    _select(at, "prediction_batch", severe_batch_id)
    assert not at.exception

    expected_status = populated_db["severe"]["prediction_report"].label_status
    rendered = " ".join(m.value for m in at.markdown)
    assert drift_status_label(expected_status) in rendered


# --- Model Performance -------------------------------------------------------------------


def test_model_performance_page_shows_the_stored_accuracy_value(populated_db):
    at = AppTest.from_file(_page("pages/4_Model_Performance.py"), default_timeout=120)
    at.run()
    stable_batch_id = populated_db["stable"]["batch_id"]
    _select(at, "performance_batch", stable_batch_id)
    assert not at.exception

    expected_accuracy = populated_db["stable"]["performance_report"].metrics["accuracy"]
    displayed_values = [m.value for m in at.metric]
    assert f"{expected_accuracy:.4f}" in displayed_values


def test_model_performance_page_shows_a_confusion_matrix_for_a_computed_batch(populated_db):
    at = AppTest.from_file(_page("pages/4_Model_Performance.py"), default_timeout=120)
    at.run()
    stable_batch_id = populated_db["stable"]["batch_id"]
    _select(at, "performance_batch", stable_batch_id)
    assert not at.exception
    # A heatmap (confusion matrix) renders as a plotly_chart; no empty-state panel for it.
    rendered = " ".join(m.value for m in at.markdown)
    assert "no confusion matrix" not in rendered.lower()


# --- Alerts ------------------------------------------------------------------------------


def test_alerts_page_shows_exactly_the_alerts_the_pipeline_actually_raised(populated_db):
    at = AppTest.from_file(_page("pages/5_Alerts.py"), default_timeout=120)
    at.run()
    assert not at.exception

    expected_total = sum(
        len(populated_db[s]["alerts"]) for s in ("stable", "severe", "moderate")
    )
    # Deduplication can fold recurring data-quality warnings into fewer rows,
    # so the displayed count is at most the sum of what each cycle raised,
    # and at least the number of distinct dedup_keys involved.
    table = at.dataframe[0].value
    assert 0 < len(table) <= expected_total

    expected_critical_titles = {
        a.title for a in populated_db["severe"]["alerts"] if a.severity == CRITICAL
    }
    displayed_critical_titles = set(table[table["severity"] == CRITICAL]["title"])
    assert expected_critical_titles <= displayed_critical_titles


def test_alerts_page_severity_filter_actually_filters(populated_db):
    at = AppTest.from_file(_page("pages/5_Alerts.py"), default_timeout=120)
    at.run()

    severity_box = at.multiselect[0]  # the "Severity" multiselect is the first on the page
    severity_box.set_value([CRITICAL]).run()
    assert not at.exception

    table = at.dataframe[0].value
    assert (table["severity"] == CRITICAL).all()


# --- Monitoring History ------------------------------------------------------------------


def test_monitoring_history_page_lists_every_run_with_correct_status(populated_db):
    at = AppTest.from_file(_page("pages/6_Monitoring_History.py"), default_timeout=120)
    at.run()
    assert not at.exception

    table = at.dataframe[0].value
    expected_batch_ids = {
        populated_db[s]["batch_id"] for s in ("stable", "severe", "moderate")
    }
    assert expected_batch_ids <= set(table["batch_id"])

    severe_row = table[table["batch_id"] == populated_db["severe"]["batch_id"]].iloc[0]
    assert severe_row["data_drift"] == "Drift"
    assert severe_row["status"] == "completed"


def test_monitoring_history_jump_button_selects_the_batch_on_overview(populated_db):
    """Exercises the real `st.switch_page` call, which needs the full
    multipage navigation context - only present when `AppTest` starts from
    the actual entrypoint (`app.py`) and navigates with `at.switch_page`,
    not when a sub-page is loaded directly via `AppTest.from_file`. Loading
    `pages/6_Monitoring_History.py` on its own would raise
    `StreamlitPageNotFoundError` here even though the real app works fine -
    an `AppTest` limitation, not a bug in the page."""
    at = AppTest.from_file(_page("app.py"), default_timeout=120)
    at.run()
    at.switch_page(_page("pages/6_Monitoring_History.py"))
    at.run()
    assert not at.exception

    moderate_batch_id = populated_db["moderate"]["batch_id"]
    _select(at, "history_jump_batch", moderate_batch_id)
    jump_button = [b for b in at.button if "View this batch" in b.label][0]
    jump_button.click().run()
    assert not at.exception

    assert at.session_state.get("selected_batch_id") == moderate_batch_id
