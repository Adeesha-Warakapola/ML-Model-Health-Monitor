# Interactive Dashboard

A Streamlit + Plotly interface over the services and SQLite history Phases
1-7 already built. Nothing in this phase computes a metric, a drift
statistic, or an alert - every number on every page comes from
`MonitoringRepository` or from calling `monitor.monitoring_cycle` once.

## Running it

```
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar (shared by every page) is
where a new batch is simulated; the seven pages read what is already in the
database.

## Architecture

```
app.py (Overview)  ─┐
pages/1_Data_Quality.py          ├─ each page: configure_page() → render_sidebar()
pages/2_Data_Drift.py            │             → read MonitoringRepository
pages/3_Prediction_Monitoring.py │             → st.plotly_chart / st.dataframe
pages/4_Model_Performance.py     │
pages/5_Alerts.py                │
pages/6_Monitoring_History.py   ─┘
        │
        ▼
dashboard_common.py   (Streamlit glue: cached context, sidebar, theme CSS)
        │
        ▼
src/monitor/monitoring_cycle.py   (one shared predict→drift→performance→alert→persist cycle)
        │
        ▼
src/monitor/{alerts,drift,prediction_monitoring,performance_monitoring,storage}.py   (Phases 5-7, unchanged)
```

Three layers, each with one job:

1. **`src/monitor/dashboard_data.py`** - pure Python (no Streamlit, no
   Plotly). Demonstration scenarios, status-to-colour/label mappings,
   "unavailable vs. zero" formatting, empty-state text, and small
   aggregations over a repository `DataFrame` (such as combining several
   features' drift statuses into one overall verdict). Fully unit tested in
   `tests/test_dashboard_data.py` - no model, dataset, or running dashboard
   needed.
2. **`src/monitor/monitoring_cycle.py`** - the orchestration Phase 7's
   `manage_alerts.py` used to own, pulled out so the CLI script and the
   dashboard call the exact same `run_monitoring_cycle` function. Neither
   trains nor retrains anything: `build_monitoring_context` loads the
   dataset once and loads the already-trained model artifact.
3. **`dashboard_common.py` + `app.py` + `pages/*.py`** - the only files that
   import Streamlit or Plotly. `dashboard_common.get_context()` builds the
   monitoring context once per process (`st.cache_resource`), so the dataset
   and the reference predictions are loaded exactly once no matter how many
   times a page reruns or how many pages are visited - this is what "refresh
   results without losing useful state" means in practice. `render_sidebar()`
   wires the "run a new batch" control to `run_monitoring_cycle` and stores
   the full result in `st.session_state`, shared across every page in the
   same browser session.

## Why most pages work the same right after a restart

Every `drift_results` row (not just drifted features), every
`performance_metrics` row (not just degraded ones), and every
`prediction_drift_results` row is saved for *every* run, regardless of
status - Phase 7 already did this for alerting; Phase 8 adds three JSON
columns (`notes_json`, `metrics_json`, `batch_label_distribution_json`,
`batch_confidence_json`) so the richer detail each page shows - per-feature
notes, the full confusion matrix, the predicted-class and confidence
breakdowns - is also restart-safe, not just the alert that was generated
from it. `storage.py`'s migration step (`_ensure_added_columns`) adds these
columns to a pre-existing database file without losing any saved row, which
`test_opening_a_pre_existing_database_without_the_newer_columns_adds_them`
checks directly.

The one thing **not** persisted is the raw row-level data (every feature
value, every individual prediction) - only its summary statistics are. So
the few charts that need actual values rather than summaries - a
reference-vs-batch value histogram on the Data Drift page, the full
confidence histogram on the Prediction Monitoring page - only appear for the
batch most recently run in *this dashboard session*
(`dashboard_common.live_result_for`). Older batches fall back to the saved
summary statistics, with an explicit note saying why the richer chart is
not shown, rather than silently omitting it.

## Pages

| Page | What it shows | Restart-safe? |
| --- | --- | --- |
| Overview | Latest batch, overall drift/prediction/performance status, feature-drift bar chart, prediction-mix chart, performance metrics | Yes |
| Data Quality | Every validation issue for the selected batch (one alert per issue, regardless of severity - see below) | Yes |
| Data Drift | Every feature's PSI/statistic/p-value/effect size; per-feature notes; raw value histogram for the live batch | Yes (notes + table); histogram needs the live batch |
| Prediction Monitoring | Predicted-class distribution and counts; confidence summary; full confidence histogram for the live batch | Yes (summary); histogram needs the live batch |
| Model Performance | Accuracy/precision/recall/F1/ROC-AUC, confusion matrix, per-class report, and a trend line across every run | Yes |
| Alerts | Every deduplicated alert, filterable by severity, category, and last-seen date range | Yes |
| Monitoring History | One combined row per run (drift/prediction/performance status, alert counts by severity), filterable by date, with a jump-to-Overview link | Yes |

### Data Quality reuses the alerts table, not a new query - with one caveat

`alerts_from_validation` (Phase 7) already creates exactly one alert per
`ValidationIssue`, errors and warnings alike - not only the ones severe
enough to matter elsewhere. That means `get_alerts(batch_id=..., category="data_quality")`
needed no new storage and no live `ValidationResult` object for the Data
Quality page to work after a restart.

**Found during Phase 9 integration testing:** this is only exactly true for
the *latest* batch. Deduplication (Phase 7) moves a recurring alert's
`batch_id` forward to whichever batch most recently retriggered it, so
querying by an older batch's id can come back empty even though that batch
genuinely had the same issue - the row now "belongs" to a later batch. The
page accounts for this directly: its empty state only claims a batch is
clean when it is also the latest run (where this ambiguity cannot exist);
for any older batch it says plainly that zero alerts does not guarantee
zero issues, rather than asserting something that might be false. See
`test_data_quality_page_is_honest_about_an_older_batchs_reassigned_alerts`
in `tests/test_dashboard_integration.py`.

## Demonstration scenarios

The sidebar's three scenarios are the exact modifiers `manage_alerts.py`
already used and measured (see `docs/alert_management.md`), only named for a
dropdown - `src/monitor/dashboard_data.SCENARIOS`:

| Scenario | Modifier | Measured result |
| --- | --- | --- |
| Stable batch | none | Only the dataset's known ~2-6% missing-value warnings |
| Moderate drift (occupation category shift) | `overwrite_category("occupation", "Gig-economy", fraction=0.05)` | PSI 0.1587 → WARNING |
| Severe drift (age shifted +25 years) | `shift_numeric("age", 25)` | PSI > 0.25 → CRITICAL, plus a confirmed performance-degradation alert |

Batch size is a slider (100-2000 rows); a size below the measured
300-row reliability floor deliberately produces an `insufficient_labels`
performance result, so that distinction (confirmed vs. early-signal vs.
no-labels-yet) can be demonstrated on demand rather than only described.

## Verification

Every page was run headlessly with Streamlit's own `AppTest` framework
(`streamlit.testing.v1.AppTest`) - the supported way to execute a Streamlit
script without a browser and inspect it for exceptions:

- All seven pages run with zero exceptions against the real (populated)
  monitoring database.
- All seven pages run with zero exceptions against an empty database,
  rendering the intended empty-state message rather than failing.
- The sidebar's "Run monitoring batch" button was clicked under all three
  scenarios; each produced exactly the alerts described in the table above,
  with zero exceptions.
- The "Reset batch simulator" button was clicked with zero exceptions.

## Limitations

- **Single-process `st.cache_resource`.** The monitoring context (model,
  drift detector, batch simulator) is shared by every browser session
  connected to the same running dashboard process, which is appropriate for
  a one-user demo but would need per-session isolation for multiple
  concurrent users running independent simulations.
- **Raw-value charts are session-only**, as explained above - a deliberate
  trade-off (no new "raw batch values" table) rather than an oversight.
- **No authentication or multi-user history separation** - every run goes
  into the same `monitoring/monitoring_history.db`, consistent with every
  earlier phase treating this as a single-operator prototype.
