# Alert Management and Monitoring History

How already-computed monitoring results become alerts, how repeated alerts
are prevented from flooding the system, and how monitoring history survives
a restart. All numbers below come from running `manage_alerts.py`.

## Nothing here is a new calculation

`alerts.py` reads fields that `validation.py`, `drift.py`,
`prediction_monitoring.py` and `performance_monitoring.py` already computed.
`storage.py` saves and retrieves those same results. Neither module runs a
statistical test, fits anything, or touches the model. This is deliberate:
Phase 7's job is deciding what is worth a person's attention and keeping a
record of it, not re-deriving facts Phases 2-6 already established.

## Severity levels

Three levels, in increasing urgency: **INFO**, **WARNING**, **CRITICAL**.

| Severity | Meaning | Example |
| --- | --- | --- |
| INFO | Worth recording, nothing to act on yet | A metric looks degraded, but from too few labelled rows to trust |
| WARNING | A real, moderate issue | A feature has moderately drifted (PSI 0.10-0.25) |
| CRITICAL | A large, confirmed issue | A feature has severely drifted (PSI >= 0.25), or confirmed accuracy loss |

The mapping from an existing verdict (`stable` / `moderate_drift` / `drift`,
`degraded` / not) to one of these three levels lives in `AlertRules`,
deliberately separate from the statistical thresholds in `DriftThresholds`
and `DEFAULT_DEGRADATION_MARGIN`. Those modules decide *whether* something
moved enough to matter; `AlertRules` only decides how loudly to say so.

## The four alert categories, and what counts as plain language

| Category | Built from | Needs labels? |
| --- | --- | --- |
| `data_quality` | `ValidationResult` | No |
| `data_drift` | `DriftReport` | No |
| `prediction_drift` | `PredictionDriftReport` | No |
| `performance` | `PerformanceReport` | **Yes** |

Every alert carries both a `reason` (the exact number: `"PSI=2.6811"`) and a
`description` written for someone who has not read the statistics
documentation, for example:

> The feature 'age' in this batch looks different from the data the model
> was trained on - a large, material shift. This can mean the kind of people
> or records now arriving has genuinely changed.

`test_validation_description_is_plain_language_not_jargon` checks this
directly: a validation alert's description is asserted not to contain the
word "PSI".

## Distinguishing "no labels" from "genuinely degraded" (requirement 14)

This is the one rule that most needed to be checked by code, not just by
description, because getting it wrong either way is bad: claiming
degradation with no evidence is a false alarm, and staying silent when the
model really has degraded is worse.

Three outcomes, each handled differently:

| `PerformanceReport.status` | Alert raised? | Severity cap |
| --- | --- | --- |
| `no_labels` | **Never** | - |
| `insufficient_labels`, with a degraded metric | Yes | capped one level below what a confirmed result would get |
| `computed`, with a degraded metric | Yes | full severity |

`alerts_from_performance` returns `[]` immediately whenever
`report.status == NO_LABELS` - there is no path by which "we do not know yet"
can turn into an alert that reads like "something is wrong." Separately, an
`insufficient_labels` result still gets a WARNING or INFO alert (never
silenced), but its description says explicitly:

> This is based on only 120 labelled rows, below the 300 considered
> reliable - treat this as an early, unconfirmed signal, not a confirmed
> problem.

`test_insufficient_sample_degradation_is_capped_below_confirmed_severity`
and `test_no_labels_produces_no_performance_alert` pin both halves of this.

## Preventing a flood of identical alerts (requirement 6)

This dataset's own Phase 1/2 finding - `workclass`, `occupation` and
`native_country` are missing in roughly 2-6% of rows, in *every* batch - is
exactly the scenario that would flood a naive alerting system: three
warnings, every single run, forever.

**The fix is at the storage layer, not the generation layer.**
`alerts.py` always produces the same logical alert every time the same
problem is still present - that is correct, and it should. `storage.py`
decides whether that logical alert becomes a *new row* or an *update to an
existing one*, using a `dedup_key` built from category and subject only
(`make_dedup_key(category, subject)` - deliberately **excluding severity**,
for a reason explained below).

Measured, running `manage_alerts.py` for four consecutive cycles:

```
All alerts after 4 cycles (deduplication keeps this list short):
                                        title severity  occurrence_count
     Data quality: missing values (workclass)  WARNING                 4
    Data quality: missing values (occupation)  WARNING                 5
Data quality: missing values (native_country)  WARNING                 4
                           Feature drift: age CRITICAL                 1
            Prediction confidence has shifted  WARNING                 1
 Model performance has dropped below baseline CRITICAL                 1
                    Feature drift: occupation CRITICAL                 1
```

Seven rows, not the seventeen-plus individual alerts raised across four
runs. The three recurring data-quality warnings show `occurrence_count` 4-5,
proving they were folded into the same row every time rather than
multiplying.

### Why severity is excluded from the dedup key

This was found and fixed during testing, not designed correctly the first
time, which is worth recording honestly. The first version included
severity in the key. That is wrong: it means a problem that starts as
WARNING and later escalates to CRITICAL gets a **second, unrelated-looking
row** instead of updating the first one - the opposite of deduplication. A
test (`test_deduplication_never_downgrades_severity`) caught this
immediately, because a WARNING recurrence of a previously-CRITICAL problem
came back as WARNING instead of staying CRITICAL. Removing severity from the
key and verifying `escalate()` fires correctly fixed it:

```
1st occurrence (WARNING):               created,       severity=WARNING, count=1
2nd occurrence (CRITICAL, escalation):  deduplicated,  severity=CRITICAL, count=2
3rd occurrence (WARNING again):         deduplicated,  severity=CRITICAL, count=3
```

The third line is the point: a milder recurrence never erases the fact that
the problem was, at some point, confirmed to be CRITICAL.

## Database structure

One SQLite file, five tables, each with a stable identifier and a timestamp
set once at creation (requirement 9):

| Table | Primary identity | One row per |
| --- | --- | --- |
| `monitoring_runs` | `run_id` (`run_<timestamp>_<uuid8>`) | one monitoring cycle |
| `batch_summaries` | `(run_id, batch_id)` unique | one batch's size and label coverage |
| `drift_results` | `(run_id, feature)` unique | one feature's drift result |
| `prediction_drift_results` | `run_id` unique | one batch's predicted-label and confidence drift |
| `performance_metrics` | `run_id` unique | one batch's accuracy/precision/recall/F1/ROC-AUC |
| `alerts` | `alert_id`, with `dedup_key` **unique** | one *distinct ongoing problem*, which is why this table's identity is handled differently from the other five |

`alerts` is the one table where "one row per occurrence" is explicitly *not*
the rule - see the deduplication section above for why.

```
monitoring_runs (run_id) ──┬──< batch_summaries
                            ├──< drift_results
                            ├──< prediction_drift_results
                            ├──< performance_metrics
                            └──< alerts
```

Every child table's `run_id` is a foreign key with `PRAGMA foreign_keys = ON`
enforced, so an alert can never reference a run that was never saved -
`test_saving_an_alert_for_a_nonexistent_run_is_rejected` checks this directly.

## Handling duplicate records and errors safely (requirement 11)

| Situation | Behaviour |
| --- | --- |
| Database file does not exist yet | Parent directories and the file are created automatically; this is treated as the normal first-run case, not an error |
| The same `run_id` saved twice | `INSERT OR IGNORE` - the second call is a no-op, not a duplicate row or a crash |
| The same batch's drift report saved twice | `INSERT OR REPLACE` keyed on `(run_id, feature)` - re-saving replaces, it does not duplicate |
| The same alert recurring | Handled by the dedup logic above - an update, never a duplicate |
| A path that cannot be opened as a database (e.g. a directory) | Raises `MonitoringStorageError` with the underlying `sqlite3` message, rather than a raw, unexplained `sqlite3.Error` |
| A query against an empty database | Returns an empty `DataFrame` with the right columns, not an exception |

## Retrieval (requirement 10)

`get_alerts`, `get_runs`, `get_drift_results` and `get_performance_metrics`
each accept the filters relevant to them - `batch_id`, `category`,
`severity`, and a `start_date`/`end_date` range compared against
`last_seen_at` (the question "is this problem still active in this period"
is answered by *last* seen, not first seen). Every retrieval method returns a
plain `pandas.DataFrame`, so results can be filtered, joined or plotted
further without any dependency on this module.

## Independent of any dashboard (requirement 12)

`storage.py` imports nothing from a UI framework
(`test_storage_module_does_not_import_streamlit` checks this), and every
test in `tests/test_storage.py` builds its `MonitoringRepository` against a
throwaway file under `tmp_path` with hand-constructed report objects - no
model, dataset, or live service is ever required to test persistence.

## Usage

```python
from monitor.alerts import generate_alerts
from monitor.storage import MonitoringRepository, make_run_id

repository = MonitoringRepository()          # monitoring/monitoring_history.db
run_id = make_run_id()
repository.save_run(run_id, batch.batch_id, started_at=...)

alerts = generate_alerts(
    batch_id=batch.batch_id,
    validation=prediction.validation,        # from Phase 2/4, already computed
    drift=drift_report,                      # from Phase 5, already computed
    prediction_drift=prediction_report,       # from Phase 6, already computed
    performance=performance_report,           # from Phase 6, already computed
)
repository.save_drift_report(run_id, drift_report)
repository.save_prediction_drift_report(run_id, prediction_report)
repository.save_performance_report(run_id, performance_report)
outcomes = repository.save_alerts(alerts, run_id)   # each either "created" or "deduplicated"

# Later - a new process, a restart, a dashboard reading the same file:
history = MonitoringRepository()
history.get_alerts(severity="CRITICAL")
history.get_alerts(category="data_quality", start_date="2026-01-01T00:00:00+00:00")
```

## Limitations

- **Dedup keying is coarse by design** (category + subject only). Two
  genuinely different problems that happen to share a category and subject -
  for instance, `age` drifting for two unrelated reasons in two unrelated
  periods - would still fold into one row. A richer system might reset
  `occurrence_count` after a problem is marked "resolved"; there is no
  resolution concept yet.
- **No alert routing or notification.** Alerts are generated and stored;
  nothing pages anyone or sends an email, by design (see the Phase 7
  restrictions).
- **Severity-to-category mapping is a judgment call**, documented in
  `AlertRules` but not independently validated the way the PSI and
  `min_batch_rows` thresholds were measured in Phase 5.
- **One SQLite file, opened and closed per call.** Fine at this project's
  scale (one writer, batch-at-a-time); a system with concurrent writers would
  need a different connection strategy.
