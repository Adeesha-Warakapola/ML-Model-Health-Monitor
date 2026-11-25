# System Integration and Comprehensive Testing

Phase 9 did not add a feature. It drove the already-built system
end to end - the real trained model, the real dataset, the real SQLite
database - and fixed what that revealed. Four genuine issues were found this
way; each is described below with how it was found and what changed.
Nothing in `src/monitor/validation.py`, `drift.py`, `prediction_monitoring.py`,
`performance_monitoring.py`, `alerts.py`, or the dashboard pages' core logic
needed to change - those held up under integration.

## How the system was verified

Two new test files drive the real, wired-together system rather than a
single module in isolation:

- **`tests/test_integration.py`** (22 tests) - calls
  `monitor.monitoring_cycle.build_monitoring_context` and
  `run_monitoring_cycle` directly, exactly as `manage_alerts.py` and the
  dashboard do, against the real model artifact and a throwaway database.
  Covers the full predict → drift → prediction-drift → performance → alert →
  persist chain; batch_id and record_id consistency; that inference only
  ever sees features, never labels; stable, severely-drifted,
  moderately-drifted, invalid, partially-labelled, and unlabelled batches;
  a simulated restart; repeated-batch deduplication; missing-artifact and
  bad-database-path error handling; and seed-based reproducibility.
- **`tests/test_dashboard_integration.py`** (13 tests) - runs the real
  pipeline to populate a database, points the dashboard's cached context at
  it, and checks with Streamlit's own `AppTest` framework that each page
  *displays* what was actually stored (not just that it renders without
  crashing, which Phase 8 already checked manually).

Both were run against the real `artifacts/baseline_model.joblib` and the
real UCI Adult dataset - the same way `test_performance_monitoring.py` and
`test_prediction.py` already did - because the point of an integration
suite is to exercise the real wiring, not a stand-in for it.

## Four real issues found and fixed

### 1. A failed batch left its run stuck at `status="running"` forever

`run_monitoring_cycle` saved a `monitoring_runs` row before scoring the
batch, but nothing ever updated it if scoring failed partway through (a
malformed batch failing validation, a storage error). The row stayed at
`status="running"` indefinitely - indistinguishable from a run still in
progress, long after the process had moved on or crashed.

**Found by:** `test_invalid_batch_fails_cleanly_and_the_run_is_recorded_as_failed`,
written to cover requirement 6 ("test ... invalid ... batches"), which
expected a sensible final status and got "running" instead.

**Fix:** `run_monitoring_cycle` (`src/monitor/monitoring_cycle.py`) now wraps
the scoring/persistence body in `try`/`except`. On failure it calls
`repository.complete_run(run_id, status="failed", notes=f"{type(error).__name__}: {error}")`
and re-raises - callers see the exact same exception as before, but the
database now honestly reflects what happened. `complete_run`
(`src/monitor/storage.py`) gained an optional `notes` parameter to write
this; the `monitoring_runs.notes` column already existed in the schema but
had never actually been written to.

### 2. Deduplication can make an older batch's data-quality page lie

The Data Quality page queried `get_alerts(batch_id=X, category="data_quality")`
to show batch `X`'s validation issues, documented as "the complete
validation detail for any batch." That is only true for the *latest*
occurrence of a recurring issue: Phase 7's deduplication moves an alert's
`batch_id` forward to whichever batch most recently retriggered it. Viewing
an *older* batch that also had the same recurring issue (for example, this
dataset's persistent ~2-6% missingness in three columns, present in nearly
every batch) returned zero rows - and the page's old empty state
unconditionally declared "No data-quality issues were raised for this
batch," which was false for that batch.

**Found by:** `test_data_quality_page_lists_the_real_validation_issues_for_a_batch`
in the dashboard integration suite failed with an `IndexError` (no table to
read) when checking an older batch than the one deduplication had most
recently reassigned the alerts to.

**Fix:** `pages/1_Data_Quality.py`'s empty state now only claims a batch is
clean when it is also the most recent run in the database (where this
ambiguity cannot exist); for any older batch, it explains plainly that zero
currently-attributed alerts does not guarantee zero issues, and points to
the Alerts page's `occurrence_count` for the real history. No storage or
alert-generation logic changed - deduplication's behaviour is correct and
already documented (`docs/alert_management.md`); only the Data Quality
page's claim about what it can promise was wrong, and both the page and
`docs/dashboard.md` now say so accurately.

### 3. Runs saved within the same second could "restart" in the wrong order

`MonitoringRepository.get_runs()` ordered by `started_at DESC` alone.
Every timestamp in this project is recorded to second resolution by design
(`isoformat(timespec="seconds")`, consistent across every module) - fine for
a human-readable log, but it means several monitoring cycles run back to
back (exactly what `manage_alerts.py`'s four scenarios, the dashboard's demo
button, and this phase's own integration tests all do) can share an
identical `started_at` value. SQLite does not guarantee any particular
order among tied rows, so "the most recent run" - which decides the default
batch shown on every dashboard page - could resolve to the wrong batch.

**Found by:** `test_data_quality_page_is_honest_about_an_older_batchs_reassigned_alerts`
was flaky: it passed in isolation and failed inside the full suite, because
three cycles created in the `populated_db` fixture sometimes landed in the
same wall-clock second, and `get_runs()`'s tie-break was unspecified.

**Fix:** `get_runs()` now orders by `ORDER BY started_at DESC, rowid DESC`.
SQLite's implicit `rowid` reflects true insertion order and breaks the tie
deterministically. This fixes the one place the ambiguity had a proven
correctness impact; the cosmetic-only orderings in `get_alerts`,
`get_drift_results`, `get_performance_metrics`, and `get_batch_summaries`
were left unchanged; no current page's correctness depends on their
relative order among same-second rows, and widening the fix to every
`ORDER BY` without a demonstrated need would be changing working code
without a reason. `test_runs_started_in_the_same_second_still_sort_most_recent_first`
locks this in.

### 4. A hardcoded `300` duplicated an already-correct default

`monitoring_cycle.py` called `PerformanceMonitor(min_labeled_sample=300)`,
repeating the exact number `PerformanceMonitor.__init__` already defaults to
(`DEFAULT_MIN_LABELED_SAMPLE` in `performance_monitoring.py`). Not a bug
today, but a latent one: if that measured floor were ever revised in one
place, the other would silently keep using the old number.

**Found by:** code review while verifying requirement 12 (configuration
consistency), not a failing test - this is the one fix in this phase that
is prevention rather than a cure.

**Fix:** `monitoring_cycle.py` now calls `PerformanceMonitor()` with no
argument, so there is exactly one place this number lives.

## Things checked and found already correct

- **Batch ID consistency** - `test_batch_id_is_identical_across_every_stage_and_every_stored_table`
  confirms the same `batch_id` appears on the `BatchPrediction`,
  `DriftReport`, `PredictionDriftReport`, `PerformanceReport`, every `Alert`,
  and every persisted row, for a real run.
- **Record ID consistency** - `test_record_id_is_consistent_between_registration_and_label_submission`
  confirms `PerformanceMonitor`'s `batch_id:row_index` keys match exactly
  between `register_batch` and `submit_labels` against real prediction
  output, not a hand-built fixture.
- **Inference uses the saved pipeline** - `test_prediction_service_loads_the_real_saved_artifact`
  confirms the loaded model is the exact `Pipeline` `train_baseline.py`
  saved (`preprocess` + `classify` steps bundled together), and that its
  fitted preprocessor transforms real batch features correctly.
- **Labels never reach inference** - `test_predict_simulated_calls_the_real_models_predict_with_only_features`
  spies on the real model's own `.predict`, not a mock, and confirms the
  label column is never among the columns it receives; `test_prediction_service_refuses_a_batch_that_still_carries_labels`
  confirms the explicit guard still rejects a batch that smuggles the label
  in as a feature; `test_labels_only_affect_performance_not_drift_or_prediction_outputs`
  confirms two batches with identical features but shuffled labels produce
  byte-for-byte identical predictions.
- **Stable / drifted / invalid / partially-labelled / unlabelled batches** -
  one test per case, against the real pipeline (see `test_integration.py`).
  One assumption of the author's own test needed correcting here, not the
  system: a *stable* (unmodified) batch can still legitimately trigger a
  **performance** alert, because the degradation-margin comparison is a
  plain threshold on a finite sample, and `performance_monitoring.py`
  already documents that sampling noise alone can occasionally push a
  metric like precision past that margin even on real, unmodified data. The
  test now asserts the thing that must always hold (no spurious
  *data_drift* or *prediction_drift* alert on unmodified input) rather than
  a stronger claim that is not actually guaranteed.
- **Restart correctness** - `test_history_survives_a_simulated_restart`
  drops every in-memory reference and reopens the database by path alone,
  confirming runs, alerts, drift results, and performance results are all
  recovered correctly - beyond what `manage_alerts.py`'s demo already
  showed, this asserts it rather than relying on someone reading the
  printed output.
- **No unintended duplicates on repeated processing** -
  `test_two_stable_batches_deduplicate_the_recurring_data_quality_alerts`
  and `test_rerunning_the_identical_batch_object_does_not_duplicate_storage_rows`
  confirm this against the real pipeline, not hand-built `Alert` objects.
- **Error handling** - missing model artifact, missing baseline metrics, and
  an unopenable database path (a directory where a file was expected) all
  raise clear, specific errors through the *full* `build_monitoring_context`
  call, not only when the lower-level function is called directly (which
  `test_model.py`/`test_prediction.py`/`test_storage.py` already covered).
- **Reproducibility** - `test_same_seed_produces_the_same_batches_and_the_same_alerts`
  confirms two independently built contexts with the same seed draw
  byte-identical batches and raise identically-titled alerts.
- **Dependencies** - every entry in `requirements.txt` is genuinely used
  (`tabulate` only indirectly, via `pandas.DataFrame.to_markdown`, which
  `report.py` calls); nothing is pinned to a version that isn't actually
  installed and exercised by the suite.
- **No hardcoded paths, secrets, or accidental commits** - every path in
  `src/`, `tests/`, and the dashboard files goes through `config.py` or a
  relative/`tmp_path` construction; no API keys, passwords, or tokens exist
  anywhere in the codebase; `git ls-files` contains no `.db`, `.joblib`, or
  raw dataset file - `data/`, `artifacts/`, and `monitoring/` are
  consistently git-ignored and regenerated on demand.

## A testing pitfall worth recording for its own sake

Two pre-existing tests - `test_drift_module_does_not_depend_on_streamlit`
and `test_storage_module_does_not_import_streamlit` - started failing the
moment this phase's dashboard-integration test file (which legitimately
needs `streamlit`) was added to the suite. Both checked `"streamlit" not in
sys.modules`, which is a statement about the whole *process*, not about
`drift.py` or `storage.py` specifically - once anything else in the same
pytest run imports streamlit, that check breaks regardless of whether the
module under test does. Both now parse the target module's own source with
`ast` and check its import statements directly, which is true or false
based only on that module - not on what else has run in the same process.
This was a test-isolation bug in the tests themselves, not in `drift.py` or
`storage.py`, which never imported streamlit and still don't.

## Test report

```
python -m pytest -q
```

**446 passed, 0 failed, 0 skipped.** Run four times in a row (once after
each fix, then three repeats) to confirm the same-second ordering fix
actually removed the flakiness rather than coincidentally passing once.

| File | Tests | Covers |
| --- | --- | --- |
| test_data.py | 8 | Dataset loading and cleaning |
| test_explore.py | 8 | Dataset inspection |
| test_validation.py | 18 | Schema validation |
| test_preprocessing.py | 20 | Preprocessing pipeline |
| test_model.py | 30 | Model build/train/save/load |
| test_evaluation.py | 24 | Metrics, confusion matrix, fairness |
| test_tuning.py | 16 | Cross-validation, threshold selection |
| test_prediction.py | 29 | Prediction service, label separation |
| test_simulation.py | 39 | Batch simulation, drift modifiers |
| test_drift.py | 62 | Data drift detection |
| test_prediction_monitoring.py | 21 | Prediction-distribution monitoring |
| test_performance_monitoring.py | 34 | Performance monitoring with delayed labels |
| test_alerts.py | 30 | Alert generation and severity |
| test_storage.py | 34 | SQLite persistence and deduplication |
| test_dashboard_data.py | 34 | Pure dashboard helper functions |
| test_monitoring_cycle.py | 4 | Shared orchestration (Phase 8) |
| **test_integration.py** | **22** | **End-to-end workflow (new, Phase 9)** |
| **test_dashboard_integration.py** | **13** | **Dashboard displays real stored results (new, Phase 9)** |
| **Total** | **446** | |

## Known limitations (unchanged or newly understood, not newly introduced)

- **Deduplication's `batch_id` reassignment** (issue 2 above) is an
  inherent trade-off of storing "one row per distinct ongoing problem"
  rather than "one row per occurrence" - already documented as a limitation
  in `docs/alert_management.md`. Phase 9 made the dashboard honest about it
  rather than changing the storage model, which would mean tracking full
  per-occurrence history and was judged out of scope for an integration
  phase that is explicitly not a rebuild.
- **`st.cache_resource` is process-wide**, not per-session - documented
  already in `docs/dashboard.md`; still appropriate for a single-operator
  prototype, not for concurrent independent users.
- **Raw per-row value charts remain session-only** - unchanged from Phase
  8; only summary statistics are persisted.
- **No authentication, deployment infrastructure, or automatic retraining**
  - unchanged, and explicitly out of scope for this phase.
- **`AppTest` cannot exercise `st.switch_page` from a sub-page loaded in
  isolation** - confirmed during this phase to be a limitation of the test
  tool, not the app: `at.switch_page` from a real entrypoint works
  correctly, and `test_monitoring_history_jump_button_selects_the_batch_on_overview`
  uses that pattern. Worth recording so a future contributor does not
  "fix" the real button because a narrower test harness couldn't reach it.

## Commands

| Command | What it does |
| --- | --- |
| `python -m pytest -q` | Run the complete suite (446 tests) |
| `python -m pytest tests/test_integration.py tests/test_dashboard_integration.py -q` | Run only this phase's new integration tests |
| `python manage_alerts.py --fresh` | Run the real CLI workflow end to end against a fresh database |
| `streamlit run app.py` | Start the real dashboard |
