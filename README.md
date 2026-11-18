# ML Model Health Monitor

An educational prototype that trains a classifier on the UCI Adult Census
Income dataset and then monitors it for data-quality problems, feature
drift, and performance degradation over time.

Built as an undergraduate portfolio project. It is **not** a production
system and must not be used to make decisions about real people.

---

## Status: Phase 6 complete — prediction distribution and performance monitoring

| Phase | Scope | State |
| --- | --- | --- |
| 1 | Project setup, dataset loading and exploration | Done |
| 2 | Schema, batch validation, preprocessing pipeline | Done |
| 3 | Baseline model training and evaluation | Done |
| 4 | Prediction service and reproducible batch simulation | Done |
| 5 | Numerical and categorical drift detection | Done |
| 6 | Prediction-distribution monitoring and performance monitoring with delayed labels | Done |
| 7+ | Alerting, monitoring history persistence, dashboard | Not started |

Baseline results on the held-out test set (9,769 rows), at the
cross-validated decision threshold of **0.29**: accuracy **0.8251**,
precision **0.6028**, recall **0.7887**, F1 **0.6833**, ROC-AUC **0.9059**.
For context, always predicting the majority class scores **0.7607** accuracy.

5-fold cross-validation puts ROC-AUC at **0.9067 ±0.0013**, so the scores are
stable. The threshold was chosen from out-of-fold training predictions, never
from the test set; it trades precision for recall deliberately. See
[`docs/baseline_model.md`](docs/baseline_model.md) for the full discussion,
including per-group behaviour by sex and race.

---

## Setup

**Requires Python 3.10 or newer.** The code uses `float | None` style type
hints, which earlier versions do not understand at runtime. Check with
`python --version`.

Create and activate a virtual environment, then install the dependencies.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows (Git Bash)**

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To leave the virtual environment later, run `deactivate`.

---

## Verify the setup

Run these commands in order. Expected results are listed beside each.

```bash
python -c "import pandas, numpy, sklearn, ucimlrepo, tabulate, joblib; print('dependencies ok')"
python explore_dataset.py
python validate_and_preprocess.py
python train_baseline.py
python simulate_batches.py
python detect_drift.py
python monitor_performance.py
python -m pytest -q
```

| Command | What it proves | Expected output |
| --- | --- | --- |
| `python -c "import ..."` | Dependencies installed | `dependencies ok` |
| `python explore_dataset.py` | Dataset downloads and is inspectable | A summary table, then `Report written to .../docs/dataset_exploration.md` |
| `python validate_and_preprocess.py` | Validation and preprocessing work end to end | Validation summaries, then `Fitted preprocessor saved to .../artifacts/preprocessor.joblib` |
| `python train_baseline.py` | The baseline trains, scores and saves | Metrics tables, then three `saved to` lines |
| `python simulate_batches.py` | Batches are drawn from unseen data and scored | `Overlap 0 rows`, per-batch summaries, a variation table |
| `python detect_drift.py` | Drift is detected in changed batches but not stable ones | Scenario 1 `STABLE`, scenarios 2–4 `DRIFT`, scenario 5 a quality failure |
| `python monitor_performance.py` | Data drift, prediction drift and performance move independently | Stable scenario all-clear; shifted scenario shows drift + degraded accuracy with ROC-AUC unchanged |
| `python -m pytest -q` | All code works | `309 passed` |

Extra flags for `train_baseline.py`:

| Flag | Effect |
| --- | --- |
| `--search` | Also cross-validate the regularisation strength `C` |
| `--compare` | Also train a Random Forest and print a side-by-side table |
| `--threshold 0.5` | Use a fixed threshold instead of selecting one |

Extra flags for `simulate_batches.py`:

| Flag | Effect |
| --- | --- |
| `--batches N --size N` | How many batches, and how many rows each |
| `--seed N` | Change the sampling seed |
| `--demo-modifier` | Show the controlled-change hook drift detection uses |

Extra flags for `detect_drift.py`:

| Flag | Effect |
| --- | --- |
| `--size N` | Rows per batch (default 500) |
| `--small-sample-demo` | Measured false-alarm rates at small batch sizes |
| `--json` | Print one report as JSON and exit |

Extra flags for `monitor_performance.py`:

| Flag | Effect |
| --- | --- |
| `--delayed-labels` | Show labels arriving in two separate waves instead of the drift scenarios |

Run `explore_dataset.py` **before** `pytest` the first time. It downloads the
dataset from UCI (this can take a minute) and caches it in `data/`, so later
runs — including the tests — are fast and work offline. If you run the tests
first they will simply do the download themselves. Use
`python explore_dataset.py --refresh` to force a re-download.

---

## Project layout

| Path | Purpose |
| --- | --- |
| `src/monitor/config.py` | Single source of truth for file paths, column names, the target column, and the random seed |
| `src/monitor/data.py` | Downloads the dataset from UCI, caches it locally, and cleans it |
| `src/monitor/explore.py` | Inspection helpers — schema, missing values, duplicates, class distribution, summary statistics |
| `src/monitor/report.py` | Formats the inspection results into the Markdown report |
| `src/monitor/schema.py` | The expected shape of a batch — column kinds, allowed categories, plausibility bounds |
| `src/monitor/validation.py` | Checks a batch against the schema and reports issues by severity |
| `src/monitor/preprocessing.py` | The `ColumnTransformer` pipeline: imputing, scaling, one-hot encoding |
| `src/monitor/model.py` | Builds, trains, persists the model and records its metadata |
| `src/monitor/evaluation.py` | Reusable metric functions, including per-group breakdowns — reused later to score monitoring batches |
| `src/monitor/tuning.py` | Cross-validation, decision-threshold selection, hyperparameter check |
| `src/monitor/prediction.py` | Loads the saved model and scores incoming batches; refuses batches containing labels |
| `src/monitor/simulation.py` | Draws reproducible batches from the held-out data, with hooks for drift experiments |
| `src/monitor/drift.py` | Compares batches against the fixed training distribution using KS, chi-square and PSI |
| `src/monitor/prediction_monitoring.py` | Compares a batch's predicted labels and confidence against the model's reference predictions (reuses `drift.py`'s PSI machinery) |
| `src/monitor/performance_monitoring.py` | Scores accuracy once true labels arrive, with stable record IDs so labels can arrive late and in parts |
| `explore_dataset.py` | Entry point: run the inspection and write the report |
| `validate_and_preprocess.py` | Entry point: validate, fit the preprocessor on training data, save it |
| `train_baseline.py` | Entry point: train the baseline, evaluate it, save model + metadata + metrics |
| `simulate_batches.py` | Entry point: simulate incoming batches and score them |
| `detect_drift.py` | Entry point: compare stable and deliberately drifted batches against the reference |
| `monitor_performance.py` | Entry point: prediction and performance monitoring, including delayed/partial labels |
| `tests/` | 309 tests across loading, exploration, validation, preprocessing, model, evaluation, tuning, prediction, simulation, drift, prediction monitoring and performance monitoring |
| `docs/dataset_exploration.md` | **Phase 1 deliverable** — findings, generated from real data |
| `docs/preprocessing.md` | **Phase 2 deliverable** — validation rules, pipeline design, and how it is reused for monitoring batches |
| `docs/baseline_model.md` | **Phase 3 deliverable** — results, every metric explained, and limitations |
| `docs/prediction_and_batches.md` | **Phase 4 deliverable** — the five kinds of data, label separation, batch sampling |
| `docs/drift_detection.md` | **Phase 5 deliverable** — the statistical methods, every threshold's provenance, and two bugs found by measurement |
| `docs/performance_monitoring.md` | **Phase 6 deliverable** — data drift vs. prediction drift vs. performance degradation, measured as three different numbers on the same batch |
| `reports/baseline_metrics.json` | Measured train and test metrics (committed — small and readable) |
| `data/` | Local dataset cache (git-ignored, regenerated on demand) |
| `artifacts/` | Fitted preprocessor and model binaries (git-ignored, regenerated on demand) |
| `pyproject.toml` | pytest configuration (puts `src/` on the import path) |
| `requirements.txt` | Python dependencies |

The root-level scripts exist so the project can be run with a plain
`python explore_dataset.py`, without needing to install it as a package or
set `PYTHONPATH` by hand.

## Validation and preprocessing

See [`docs/preprocessing.md`](docs/preprocessing.md) for the full design. In short:

```python
from monitor.preprocessing import load_preprocessor, transform_batch
from monitor.validation import validate_batch

preprocessor = load_preprocessor()       # fitted on training data only
result = validate_batch(incoming_batch)  # always validate first
print(result.summary())

if result.is_valid:
    features = transform_batch(preprocessor, incoming_batch)
```

- The preprocessor is **fitted once on training data** and only ever
  *transforms* later batches. Refitting on a batch would scale away the very
  shift the monitor is meant to detect, so there is deliberately no
  `fit_transform` helper for batches.
- Unknown categories encode as all-zeros instead of raising, so the output is
  always the same 108 columns and any batch can be scored.
- Validation **reports** rather than repairs. Errors (missing columns, wrong
  types) block processing; warnings (missing values, unknown categories,
  out-of-range numbers) do not.

## Using the trained model

The saved artifact bundles preprocessing and the classifier, so one load is
enough to score raw Adult-shaped rows:

```python
from monitor.model import load_model, get_threshold

model = load_model()                  # artifacts/baseline_model.joblib
get_threshold(model)                  # 0.29
predictions = model.predict(raw_dataframe)
```

The artifact bundles preprocessing, the classifier **and** the decision
threshold, so `predict` can never disagree with the documented metrics.

Because `Pipeline.fit(X_train, y_train)` fits the preprocessor inside the same
call, the preprocessor never sees the test set — there is no point at which
preprocessing could be fitted on the full dataset before splitting. The
threshold is likewise chosen from out-of-fold predictions within the training
half only.

## Scoring incoming batches

See [`docs/prediction_and_batches.md`](docs/prediction_and_batches.md) for the
full design, including what "training data", "reference data", "incoming
batch", "prediction" and "actual label" each mean in this project.

```python
from monitor.data import load_adult_data
from monitor.prediction import PredictionService
from monitor.simulation import BatchSimulator, split_reference_and_holdout

df = load_adult_data()
reference, holdout = split_reference_and_holdout(df)

service = PredictionService()                 # loads the saved model
simulator = BatchSimulator(holdout, batch_size=500, seed=42)

batch = simulator.next_batch()                # features + labels, kept apart
result = service.predict_simulated(batch)     # only features are passed in
record = result.to_frame()                    # features + predictions, for monitoring
truth = batch.labels                          # available later, never used above
```

- **Batches come only from the held-out 20%** — the same split from Phase 3,
  rebuilt with the same seed, so no row the model trained on can appear.
- **Labels are structurally separated.** `predict_batch` *raises* if the frame
  still contains the income column, because at real prediction time it would
  not exist yet.
- **Sampling is random, not stratified**, so batches vary naturally around the
  24% rate instead of being pinned to it. Nothing is balanced.
- **Batches are disjoint and reproducible** — the same seed replays the same
  sequence, and each record arrives at most once.

## Detecting drift

See [`docs/drift_detection.md`](docs/drift_detection.md) for the methods, every
threshold's provenance, and the limitations.

```python
from monitor.drift import DriftDetector

detector = DriftDetector(reference)                  # fixed training distribution
report = detector.detect(batch.features, batch_id=batch.batch_id)

print(report.status)        # "stable" | "moderate_drift" | "drift"
report.to_frame()           # one row per feature, sorted by PSI
report.to_dict()            # JSON-ready
report.validation           # data quality, kept separate from drift
```

- **Numerical** features use a KS test plus PSI; **categorical** features use
  chi-square plus PSI, with Cramér's V as the effect size.
- **PSI decides the verdict, not the p-value.** With 39,073 reference rows,
  adding 1 year to every age gives p = 1.5e-13 while PSI is 0.0081 — the report
  says so explicitly instead of crying drift.
- **Data-quality failures are separated from drift.** Impossible values are a
  pipeline bug; a distribution shift is a reason to retrain.
- **No feature is ever silently dropped.** If a statistic cannot be computed,
  the feature is reported as `not_computed` with the reason.
- Thresholds are configurable and recorded in every report. Two defaults
  (`min_batch_rows=400`, `min_category_share=0.01`) were **measured** from
  false-alarm rates on unmodified data, not guessed.

## Prediction and performance monitoring

See [`docs/performance_monitoring.md`](docs/performance_monitoring.md) for the
full picture. **Data drift, prediction drift, and performance degradation are
three different questions** and were measured to disagree on the same batch
(600 rows, `age + 25`, true labels untouched):

| | Stable batch | Age +25 batch |
| --- | --- | --- |
| Data drift (age PSI) | 0.0373 | **5.1124** |
| Prediction drift (label PSI) | 0.0001 | 0.0324 (still stable) |
| Prediction drift (confidence PSI) | 0.0112 | **0.1181** |
| Performance (accuracy) | 0.8317 | **0.7683** |
| Performance (ROC-AUC) | 0.9094 | 0.9094 (barely moved) |

The input changed severely; the hard predicted-label mix barely moved because
the threshold absorbed it; the underlying confidence scores moved more and
earlier; and real accuracy genuinely fell while the model's ranking ability
(ROC-AUC) stayed intact. No single one of these tells the whole story.

```python
from monitor.prediction_monitoring import PredictionMonitor
from monitor.performance_monitoring import PerformanceMonitor, load_baseline_test_metrics

prediction_monitor = PredictionMonitor.from_reference_features(service, reference_features)
performance_monitor = PerformanceMonitor(min_labeled_sample=300)   # measured, not guessed

prediction = service.predict_simulated(batch)           # no labels involved
print(prediction_monitor.evaluate(prediction).summary())  # available immediately

performance_monitor.register_batch(prediction)           # still no labels
# ... later, maybe in several waves, maybe only partially ...
performance_monitor.submit_labels(batch.batch_id, batch.labels)

report = performance_monitor.compute_performance(
    batch.batch_id, baseline=load_baseline_test_metrics()
)
```

- **Predictions are recorded before any label exists**, and labels can arrive
  later in any number of separate calls without duplicating a single record -
  both are keyed by a stable `batch_id:row_index` identifier.
- **Zero labels never produces a metric.** `compute_performance` returns
  `metrics=None` rather than a fabricated number when nothing has been
  labelled yet.
- A labelled sample below the measured floor (`min_labeled_sample=300`,
  chosen from how recall's sampling noise behaves on this dataset) is still
  computed and returned, flagged rather than hidden.

---

## Dataset

| Field | Value |
| --- | --- |
| Name | Adult (also called Census Income) |
| Source | https://archive.ics.uci.edu/dataset/2/adult |
| UCI id | 2 |
| DOI | 10.24432/C5XW20 |
| Creators | Barry Becker, Ronny Kohavi |
| Accessed via | the `ucimlrepo` package |
| Size | 48,842 rows x 15 columns |

The dataset is **not committed to this repository**. It is downloaded on
first run and cached in `data/`, which keeps the repo small and makes the
data acquisition step reproducible and visible in code.

See [`docs/dataset_exploration.md`](docs/dataset_exploration.md) for the full
findings. The headline points:

- **Class imbalance of about 3.18:1** (76.07% `<=50K`, 23.93% `>50K`). This
  natural distribution is kept everywhere in the project; rebalancing it
  would misrepresent real incoming data.
- **Missing values are encoded inconsistently.** Rows from the original
  training split use the string `"?"`; rows from the test split use real
  empty values. Loading the data naively hides about two thirds of the
  missing values and invents a phantom `"?"` category.
- **Three columns have missing values**: `occupation` (5.75%),
  `workclass` (5.73%), `native_country` (1.75%).
- **52 fully duplicated rows** (0.11%), left in place because anonymous
  census records can legitimately coincide.

### Usage terms

The UCI ML Repository publishes this dataset for public research and
educational use. The `ucimlrepo` metadata response carries no
machine-readable licence field, so the licence shown on the
[dataset page](https://archive.ics.uci.edu/dataset/2/adult) is the
authoritative statement of terms. This project uses the data for
non-commercial educational purposes only and redistributes no part of it.

---

## Limitations

- The data is a 1994 US census extract and does not describe the present day.
- It contains sensitive attributes (`race`, `sex`, `native_country`) and the
  label rate differs sharply across groups. Any model trained on it will
  reproduce those patterns.
- This project demonstrates monitoring techniques. It has not been assessed
  for fairness, and no part of it should inform real decisions about people.

---
