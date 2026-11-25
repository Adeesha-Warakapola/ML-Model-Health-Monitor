# Demo Guide: Stable vs. Drifted Monitoring

A short, reproducible walkthrough showing the monitoring system tell a
stable batch apart from a deliberately drifted one - once from the command
line, once from the dashboard. Every number below is exactly what the fixed
seed (`42`) produces; running the same commands yourself will print the
same numbers.

Prerequisites: the virtual environment is active and dependencies are
installed (see the root [`README.md`](../README.md#installation)), and
`python train_baseline.py` has been run at least once so
`artifacts/baseline_model.joblib` exists.

## Part 1 — command line (`manage_alerts.py`)

```bash
python manage_alerts.py --fresh
```

`--fresh` deletes any existing `monitoring/monitoring_history.db` first, so
the run starts from an empty history. The script then runs four scenarios
in sequence and prints each one's verdict and alerts:

| Scenario | What it does | What to expect |
| --- | --- | --- |
| 1. Normal batch | Unmodified holdout rows | `data drift: stable`, only the dataset's known missing-value warnings |
| 2. Another normal batch | Unmodified again | Same three warnings, `occurrence_count` rising, **not** new rows |
| 3. Moderate problem | 5% of rows get an unseen `occupation` category | `data drift: moderate_drift`, one new WARNING alert |
| 4. Severe problem | Every row's `age` shifted +25 years | `data drift: drift`, CRITICAL alerts for both drift and performance |

After the four scenarios, the script **simulates a restart**: it discards
its in-memory repository object, opens a brand-new one against the same
database file, and reads everything back - proving the history survived
without needing the original process still running.

```
All alerts after 4 cycles (deduplication keeps this list short):
                                        title severity  occurrence_count
     Data quality: missing values (workclass)  WARNING                 4
    Data quality: missing values (occupation)  WARNING                 5
Data quality: missing values (native_country)  WARNING                 4
                           Feature drift: age CRITICAL                 1
            Prediction confidence has shifted  WARNING                 1
 Model performance has dropped below baseline CRITICAL                 1
                    Feature drift: occupation  WARNING                 1
```

Seven alert rows, not the dozen-plus individual alerts raised across four
runs - that collapse is deduplication working as designed (see
[`docs/alert_management.md`](alert_management.md)).

## Part 2 — the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar is the same on every page:

1. **Choose a scenario** from the dropdown - *Stable batch*, *Moderate
   drift (occupation category shift)*, or *Severe drift (age shifted +25
   years)*. These are the exact three modifiers used above, named for a
   dropdown.
2. **Pick a batch size** with the slider (100-2000 rows). A size at or
   above 300 produces a confident (`computed`) performance result; below
   300 produces a flagged (`insufficient_labels`) one - both are worth
   trying, to see the dashboard distinguish "confirmed" from "early
   signal."
3. Click **Run monitoring batch**. The sidebar shows a success message with
   the new batch's id once the cycle completes.

### A stable run

- **Overview**: status chips read *Stable* for both data drift and
  prediction drift; the feature-drift bar chart's tallest bars still sit
  under the PSI 0.10 line.
- **Data Quality**: the three recurring missing-value warnings, nothing
  else.
- **Alerts**: no new CRITICAL rows; existing WARNING rows' occurrence count
  goes up by one.

### A severe-drift run

- **Overview**: the *Data drift* chip turns red (*Drift*); the `age` bar in
  the feature-drift chart is far above both threshold lines.
- **Data Drift**: select `age` in the feature detail dropdown to see the
  reference-vs-batch histogram - two visibly non-overlapping distributions,
  because every value was shifted by +25.
- **Model Performance**: accuracy and precision show a visible drop against
  the Phase-3 baseline (the delta is shown directly under each metric).
- **Alerts**: a CRITICAL `data_drift` row and a CRITICAL `performance` row
  appear, both referencing this run's batch id.
- **Monitoring History**: the new run appears at the top, marked
  `completed`, with `data_drift = Drift`.

### Comparing the two side by side

Switch the **Batch** selector at the top of any page between the stable
run's batch id and the severe run's batch id (both are still in the
dropdown - nothing is overwritten) to see the same page render two very
different pictures of the same model, using the same code path, from the
same stored history.

## Why this is a fair demonstration, not a cherry-picked one

- The "severe drift" scenario does not retrain, adjust, or special-case
  anything - it draws real holdout rows and adds 25 to one real column.
  Every number the dashboard shows afterward (PSI, accuracy, the alert
  severities) is computed by the same code that scores an unmodified batch.
- The thresholds that decide *stable* vs. *moderate* vs. *drift*
  (`psi_moderate=0.10`, `psi_major=0.25`) and the performance floors
  (`min_batch_rows=400`, `min_labeled_sample=300`) were measured from
  false-alarm rates on **unmodified** data, not chosen to make this demo
  look good - see [`docs/drift_detection.md`](drift_detection.md) and
  [`docs/performance_monitoring.md`](performance_monitoring.md).
- Running `python manage_alerts.py --fresh` or the dashboard's "Stable
  batch" scenario repeatedly will occasionally - rarely - still produce a
  WARNING-level performance alert on a genuinely unmodified batch, purely
  from sampling noise on a finite sample. That is expected and documented,
  not a bug; see "Known limitations" in the root README.
